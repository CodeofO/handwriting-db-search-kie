"""디스크 보호 — 서버가 디스크 부족으로 죽지 않게 단계별로 정리 · 보류 · 중단한다. scripts/watch.sh가 2분마다 1회 실행.

사용률 u = used / (used + avail) (df의 Use%와 같은 정의). 단계 기준(LEVELS, 환경변수 FORMREC_DISK_LEVELS="93,94,95,94.5,96,97"로 바꿀 수 있음):
  u ≥ L1 (93)   파생 정리(다시 만들 수 있는 것만): 최선이 아니고 val 채점이 끝난 VLM 에폭 내보내기, 학습이 끝난 데이터셋의 OvisOCR2 base64 학습 파일 ·
                전처리 캐시(cache/hfds/{ds}), 학습이 끝난 PP-OCRv5 옵티마이저 상태(.pdopt), 실제 실험 시작 뒤의 스모크 잔여물(스모크가 도는 중이면 제외)
  u ≥ L2 (94)   체크포인트 정리: VLM 체크포인트 중, 그 데이터셋의 loss 기준 선정(best.json)과 지표 기준 선정(모든 체크포인트의 val 완전 일치)이
                모두 기록된 경우에만 두 선정 체크포인트를 남기고 나머지 삭제 → runs/{ds}/{k}_ft/pruned.json 에 기록
  u ≥ HOLD (95) 시작 보류: logs/HOLD_DISK 생성 → 스케줄러가 새 학습(큰 쓰기)을 시작하지 않음(돌던 작업 · 평가는 계속). u < HOLD_OFF (94.5)면 해제
  u ≥ KILL (96) 쓰기 중단: 돌고 있는 GPU 작업을 종료(kill_job.sh). 학습은 모두 이어 학습(--resume)을 지원 → 보류 해제 뒤 자동 재시도
  u ≥ EMERG (97) 비상(다른 사용자 원인 포함): 이 프로젝트의 돌고 있는 작업(X_ · S_)을 모두 종료하고 보류
기록: logs/disk_guard.log, logs/events.log("EVENT disk_guard: …"), logs/disk_guard_state.json(리포트용)
--dry: 아무것도 지우거나 끄지 않고 할 일만 기록.
사용: python -m formrec.ops.disk_guard [--dry]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from formrec.common import workspace as cfg

ROOT = cfg.ROOT
LOG, EV, HOLD, STATE = ROOT / "logs/disk_guard.log", ROOT / "logs/events.log", ROOT / "logs/HOLD_DISK", ROOT / "logs/disk_guard_state.json"
_lv = [float(x) for x in os.environ.get("FORMREC_DISK_LEVELS", "93,94,95,94.5,96,97").split(",")]
T1, T2, T3, T3_OFF, T4, T5 = _lv
DSS = ("xa", "xh", "xi", "xc", "xk")
PLAN = "x"                          # 실제 실험 계획의 기록 폴더 이름(logs/sched/x)
DRY = False


def pct():
    st = os.statvfs(ROOT)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    return 100.0 * used / (used + avail), avail / 2 ** 30


def note(msg, event=True):
    line = time.strftime("%F %T ") + msg
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    if event:
        with open(EV, "a") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + "EVENT disk_guard: " + msg + "\n")
    print(line, flush=True)


def size_gb(p):
    try:
        out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True, timeout=600).stdout.split()
        return int(out[0]) / 2 ** 30 if out else 0.0
    except Exception:
        return 0.0


def rm(p, why):
    """작업 공간 안의 경로만 지운다(DRY면 기록만)."""
    p = Path(p)
    if not (p.exists() or p.is_symlink()):
        return 0.0
    assert ROOT in p.absolute().parents, f"작업 공간 밖 경로: {p}"
    g = size_gb(p)
    if not DRY:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    note(f"{'(dry) ' if DRY else ''}삭제 {p.relative_to(ROOT)} ({g:.1f} GB) — {why}")
    return g


def done(job):
    return (ROOT / f"logs/sched/{PLAN}/{job}.done").exists()


def smoke_running():
    return subprocess.run(["pgrep", "-f", "formrec.ops.jobsched .*plan_smoke"], capture_output=True).returncode == 0


def level1():
    freed = 0.0
    for ds in DSS:
        for k in ("pvl", "ovis"):
            for rd in ROOT.glob(f"runs/{ds}/{k}_ft*"):
                best = rd / "export/best"
                tgt = os.readlink(best) if best.is_symlink() else None
                sx = rd.name[len(f"{k}_ft"):]
                for e in rd.glob("export/ep*"):
                    i = e.name[2:]
                    if e.name != tgt and (ROOT / f"results/{ds}/g1/{k}ft{sx}_ep{i}_val.parquet").exists():
                        freed += rm(e, "VLM 에폭 내보내기, val 채점 끝남(체크포인트에서 다시 만들 수 있음)")
        if done(f"X_{ds}_ovis"):
            freed += rm(ROOT / f"ft/{ds}/ovis/train.jsonl", "OvisOCR2 base64 학습 파일, 학습 끝남(다시 만들 수 있음)")
            freed += rm(ROOT / f"cache/hfds/{ds}", "OvisOCR2 전처리 캐시, 학습 끝남")
        if done(f"X_{ds}_ppocr"):
            for p in ROOT.glob(f"runs/{ds}/ppocrv5_ft/*.pdopt"):
                freed += rm(p, "PP-OCRv5 옵티마이저 상태, 학습 끝남(가중치 .pdparams는 보존)")
    if (ROOT / "logs/x_production_started").exists() and not smoke_running():
        for p in list(ROOT.glob("runs/x/*_smoke")) + list(ROOT.glob("runs/x[ahi]/*_smoke")):
            freed += rm(p, "스모크 잔여물")
    return freed


def val_em(ds, k, sx, i):
    import pandas as pd
    p = ROOT / f"results/{ds}/g1/{k}ft{sx}_ep{i}_val.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["pred", "text"])
    return float((d.pred == d.text).mean())


def level2():
    """두 선정(loss · 지표)이 모두 기록된 데이터셋의 VLM 체크포인트만 정리."""
    freed = 0.0
    for ds in DSS:
        for k in ("pvl", "ovis"):
            rd = ROOT / f"runs/{ds}/{k}_ft"
            if not (rd.exists() and done(f"X_{ds}_{k}") and (rd / "best.json").exists()):
                continue
            cks = sorted(rd.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            if len(cks) <= 2:
                continue
            ems = {i: val_em(ds, k, "", i) for i in range(1, len(cks) + 1)}
            if any(v is None for v in ems.values()):
                continue                                   # 지표 기준 선정이 아직 → 지우지 않음
            lb = int(json.load(open(rd / "best.json"))["best_epoch"])
            mb = max(ems, key=lambda i: (ems[i], -i))
            keep = {lb, mb}
            dels = [cks[i - 1] for i in range(1, len(cks) + 1) if i not in keep]
            for c in dels:
                freed += rm(c, f"VLM 체크포인트, 선정 기록 완료(loss 최선 {lb} · 지표 최선 {mb} 보존)")
            rec = rd / "pruned.json"
            old = json.load(open(rec)) if rec.exists() else {}
            old.update(dict(loss_best=lb, metric_best=mb, val_em=ems, deleted=sorted(set(old.get("deleted", [])) | {c.name for c in dels}), time=time.strftime("%F %T")))
            if not DRY:
                json.dump(old, open(rec, "w"), indent=1)
    return freed


def running_jobs(gpu_only=True):
    """이 프로젝트 스케줄러가 띄운 작업(X_ · S_) 중 살아 있는 것: events.log의 START pid 기준."""
    alive = {}
    if not EV.exists():
        return []
    for line in open(EV, errors="ignore").readlines()[-5000:]:
        m = re.search(r"START (\S+) pid=(\d+) gpu=(\S+)", line)
        if m:
            n, pid, g = m.group(1), int(m.group(2)), m.group(3)
            if not n.startswith(("X_", "S_")):
                continue
            try:
                os.kill(pid, 0)
                alive[n] = g
            except OSError:
                alive.pop(n, None)
    return [n for n, g in alive.items() if (g != "none" if gpu_only else True)]


def kill(names, why):
    for n in names:
        note(f"{'(dry) ' if DRY else ''}작업 종료 {n} — {why}")
        if not DRY:
            subprocess.run(["bash", str(cfg.REPO / "scripts/kill_job.sh"), n], capture_output=True, timeout=120)


def guard(usage=pct):
    """한 번 점검 · 조치. usage = () -> (사용률 %, 남은 GB). -> 상태 dict"""
    u, av = usage()
    acts = []
    if u >= T1:
        f = level1(); u, av = usage(); acts.append(f"L1 {f:.1f} GB")
    if u >= T2:
        f = level2(); u, av = usage(); acts.append(f"L2 {f:.1f} GB")
    if u >= T3 and not HOLD.exists():
        if not DRY:
            HOLD.write_text(time.strftime("%F %T"))
        note(f"시작 보류 ON (사용률 {u:.1f}%) — 새 학습 작업 대기")
        acts.append("HOLD on")
    elif u < T3_OFF and HOLD.exists():
        if not DRY:
            HOLD.unlink(missing_ok=True)
        note(f"시작 보류 OFF (사용률 {u:.1f}%)")
        acts.append("HOLD off")
    if u >= T5:
        kill(running_jobs(gpu_only=False), f"비상: 사용률 {u:.1f}% ≥ {T5}%"); acts.append("L5 kill all")
    elif u >= T4:
        kill(running_jobs(gpu_only=True), f"쓰기 중단: 사용률 {u:.1f}% ≥ {T4}%"); acts.append("L4 kill gpu")
    st = dict(time=time.strftime("%F %T"), pct=round(u, 2), avail_gb=round(av, 1), hold=HOLD.exists(), actions=acts,
              thresholds=dict(L1=T1, L2=T2, hold=T3, hold_off=T3_OFF, kill=T4, emergency=T5))
    if not DRY:
        json.dump(st, open(STATE, "w"), indent=1)
    return st


def main(argv=None):
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    DRY = a.dry
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    print(json.dumps(guard(), ensure_ascii=False))


if __name__ == "__main__":
    main()
