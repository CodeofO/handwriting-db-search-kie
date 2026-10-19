"""GPU 인지 DAG 스케줄러. 계획(JOBS 목록)의 작업을 의존성 · GPU 수 · 디스크 여유에 맞춰 scripts/run_job.sh로 실행한다.

규칙
- 학습 = GPU NGPU장(4), 평가 · 추론 = GPU 1장(빈 GPU마다 하나씩 동시 실행), CPU 작업(gpus=0)은 동시에 MAX_CPU개.
- 목록 순서가 우선순위. 준비된 NGPU장 작업이 GPU를 기다리면 그 뒤의 GPU 작업은 새로 시작하지 않는다(예약, 굶김 방지).
  short=True(수 분 안에 끝나는 1장 작업)는 예약 중에도 빈 GPU를 채운다(긴 1장 작업 하나 때문에 나머지 GPU가 노는 것 방지).
- 대용량 IO(io=True)는 run_job.sh의 io 잠금으로 한 번에 하나.
- deps = 산출물이 필요한 의존(실패하면 SKIP), after = 순서만(성공 여부 무관 — 평가 실패가 다음 학습을 막지 않게).
- 성공한 작업은 logs/sched/<계획>/<이름>.done → 다시 켜도 건너뜀(이어 하기). 실패하면 RETRY_SEC 뒤 AUTO_RETRY회 자동 재시도.
  손으로 다시: logs/sched/<계획>/<이름>.retry 파일을 만들면 실패 · 건너뜀 기록을 지우고 의존 작업과 함께 다시 대기.
- 디스크 보호: 큰 쓰기 작업(disk_gb ≥ GROW_GB, 학습)은 보류(logs/HOLD_DISK, disk_guard가 관리) 중이면 대기, 시작 후 예측 사용률이
  DISK_LIMIT를 넘으면 대기. 작게 쓰고 지우는 작업(평가 · 내보내기)은 보류 중에도 예측이 SOFT_MAX 아래면 실행 —
  보류가 체크포인트별 val 채점(= 체크포인트 정리의 전제)까지 막으면 공간이 영영 안 생기는 교착이 있었다.
- 계획은 매 주기마다 다시 읽는다 → 돌고 있는 스케줄러를 끄지 않고 작업을 추가 · 수정할 수 있다.
- 스케줄러 재시작 전에 시작돼 아직 도는 작업(run_job.sh 프로세스)은 이어받아(ADOPT) 끝나면 events.log의 END/FAIL로 판정.
이벤트: logs/events.log 에 QUEUE_START / QUEUE_ADOPT / QUEUE_SKIP / QUEUE_AUTORETRY / QUEUE_RETRY / WAIT / QUEUE_DONE / QUEUE_STOP.
계획: .py 경로 또는 모듈 이름(예: formrec.ops.plan_full). 계획의 NAME(없으면 파일 이름에서 'plan_'을 뺀 것)이 기록 폴더 이름.
사용: scripts/run_job.sh sched_x --stall-min 60 -- python -m formrec.ops.jobsched formrec.ops.plan_full
    (스케줄러도 run_job.sh로 실행 → 죽으면 FAIL, 멈추면 STALL 알림. 10분마다 심장박동 줄을 찍는다)
"""
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

from formrec.common import workspace as cfg


def _env(name, default, cast=str):
    """FORMREC_<name> (옛 이름 KIE_<name>도 받음)."""
    v = os.environ.get(f"FORMREC_{name}", os.environ.get(f"KIE_{name}"))
    return cast(v) if v is not None else default


ROOT = cfg.ROOT
NGPU = _env("NGPU", 4, int)
MAX_CPU = _env("MAX_CPU", 2, int)
POLL = _env("SCHED_POLL", 5.0, float)
AUTO_RETRY = _env("SCHED_RETRY", 1, int)
RETRY_SEC = _env("SCHED_RETRY_SEC", 600, int)
RUNJOB = _env("RUNJOB", str(cfg.REPO / "scripts/run_job.sh"))      # 시험용으로 바꿀 수 있음
DISK_LIMIT = _env("DISK_LIMIT", 95.0, float)
SOFT_MAX = _env("DISK_SOFT", 95.5, float)
GROW_GB = 5.0                     # 이 이상 쓰는 작업(학습) = 디스크를 키움
HB_SEC = 600                      # 심장박동 간격
EVENTS = ROOT / "logs/events.log"
HOLD = ROOT / "logs/HOLD_DISK"    # 디스크 보호(formrec.ops.disk_guard)가 만들고 지운다


def ev(msg):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a") as f:
        f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")


def alive_gpus(name):
    """run_job.sh <name> 프로세스가 살아 있으면 그 작업의 GPU 목록(없으면 []), 없으면 None."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    for a in out:
        t = a.split()
        if len(t) >= 3 and t[0] == "bash" and t[1].endswith("run_job.sh") and t[2] == name:
            opts = t[:t.index("--")] if "--" in t else t
            if "--gpu" in opts:
                return list(range(NGPU))
            if "--gpus" in opts:
                return [int(x) for x in t[t.index("--gpus") + 1].split(",")]
            return []
    return None


def last_status(name):
    """events.log에서 그 작업의 마지막 END/FAIL."""
    st = None
    if not EVENTS.exists():
        return st
    with open(EVENTS) as f:
        for line in f:
            p = line.split("] ", 1)[-1].split(" ", 2)
            if len(p) >= 2 and p[1] == name and p[0] in ("END", "FAIL"):
                st = p[0]
    return st


def disk_usage():
    """(사용 바이트, 사용 + 남은 바이트) — df의 Use%와 같은 정의."""
    st = os.statvfs(ROOT)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return used, used + st.f_bavail * st.f_frsize


def disk_ok(need_gb):
    used, total = disk_usage()
    after = 100 * (used + need_gb * 2 ** 30) / total
    if need_gb >= GROW_GB:
        return not HOLD.exists() and after < DISK_LIMIT
    return after < SOFT_MAX


def load_plan(plan):
    """계획 → (이름, JOBS). 경로면 run_path, 아니면 모듈로 실행(매번 새로 — 계획 수정 반영)."""
    p = Path(plan)
    g = runpy.run_path(str(p.resolve())) if p.suffix == ".py" or p.exists() else runpy.run_module(plan, run_name="__plan__")
    stem = p.stem if p.suffix == ".py" else plan.rsplit(".", 1)[-1]
    return g.get("NAME", stem.replace("plan_", "")), g["JOBS"]


def normalize(jobs):
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "중복 작업 이름"
    for j in jobs:
        j.setdefault("gpus", 0); j.setdefault("io", False); j.setdefault("deps", []); j.setdefault("stall", 20); j.setdefault("short", False)
        j.setdefault("after", [])
        assert j["gpus"] in (0, 1, NGPU), j
        for d in j["deps"] + j["after"]:
            assert d in names, f"{j['name']}: 없는 의존 {d}"
    return jobs


def child_env():
    """작업 환경: 저장소 코드(PYTHONPATH) · 작업 공간 경로를 넘긴다."""
    e = dict(os.environ)
    e["FORMREC_REPO"], e["FORMREC_WORKSPACE"] = str(cfg.REPO), str(ROOT)
    src = str(cfg.REPO / "src")
    e["PYTHONPATH"] = src + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


class Scheduler:
    def __init__(self, plan):
        self.plan = plan
        self.pname, jobs = load_plan(plan)
        jobs = normalize(jobs)
        self.sd = ROOT / "logs/sched" / self.pname
        self.sd.mkdir(parents=True, exist_ok=True)
        self.running = {}            # 이름 -> (Popen, GPU 목록, 작업)
        self.external = {}           # 이어받은 작업: 이름 -> GPU 목록
        self.failed, self.skipped = set(), set()
        self.tries, self.retry_at = {}, {}
        self.disk_wait = set()
        self.last_hb = 0
        for j in jobs:
            gs = alive_gpus(j["name"])
            if gs is not None:
                self.external[j["name"]] = gs
                ev(f"QUEUE_ADOPT {self.pname} {j['name']} (이미 실행 중, gpus={gs})")
        ev(f"QUEUE_START {self.pname} ({len(jobs)} jobs, plan={Path(plan).name})")

    def launch(self, j, gs):
        cmd = [RUNJOB, j["name"]]
        if gs:
            cmd += ["--gpus", ",".join(map(str, gs))]
        if j["io"]:
            cmd += ["--io"]
        cmd += ["--stall-min", str(j["stall"]), "--", "bash", "-c", j["cmd"]]
        return subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def tick(self):
        """한 주기. 끝났으면 False."""
        try:
            jobs = normalize(load_plan(self.plan)[1])
        except Exception as e:       # 계획 수정 중 오류: 돌던 작업은 두고 알린 뒤 다시 시도
            ev(f"ALERT sched_{self.pname}: 계획 파일 오류 {e!r}"[:300])
            time.sleep(60)
            return True
        sd, running, external, failed, skipped = self.sd, self.running, self.external, self.failed, self.skipped
        done = {p.stem for p in sd.glob("*.done")}
        for n, (p, gs, j) in list(running.items()):          # 끝난 작업 회수
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            if rc == 0:
                (sd / f"{n}.done").write_text(time.strftime("%F %T"))
                done.add(n)
            else:
                failed.add(n)
                if self.tries.get(n, 0) < AUTO_RETRY:
                    self.retry_at[n] = time.time() + RETRY_SEC
                    ev(f"QUEUE_AUTORETRY {self.pname} {n}: {RETRY_SEC // 60}분 뒤 자동 재시도 ({self.tries.get(n, 0) + 1}/{AUTO_RETRY})")
        for n in list(external):                              # 이어받은 작업: 프로세스가 끝나면 마지막 END/FAIL로 판정
            if alive_gpus(n) is None:
                external.pop(n)
                if last_status(n) == "END":
                    (sd / f"{n}.done").write_text(time.strftime("%F %T"))
                    done.add(n)
                else:
                    failed.add(n)
        for n, t in list(self.retry_at.items()):
            if time.time() >= t:
                (sd / f"{n}.retry").write_text("auto")
                self.tries[n] = self.tries.get(n, 0) + 1
                del self.retry_at[n]
        for rp in sd.glob("*.retry"):                         # 재시도 표시 → 실패 · 건너뜀 기록에서 지우고 다시 대기(의존 작업도 함께)
            n = rp.stem
            if n in failed or n in skipped:
                failed.discard(n)
                skipped.discard(n)
                dep, changed = {n}, True
                while changed:
                    changed = False
                    for j in jobs:
                        if j["name"] in skipped and any(d in dep for d in j["deps"]):
                            skipped.discard(j["name"])
                            dep.add(j["name"])
                            changed = True
                ev(f"QUEUE_RETRY {self.pname} {n} (+{len(dep) - 1} 의존 작업)")
            rp.unlink()
        changed = True                                        # 실패 전파
        while changed:
            changed = False
            for j in jobs:
                n = j["name"]
                if n in skipped or n in done or n in failed or n in running or n in external:
                    continue
                bad = [d for d in j["deps"] if d in failed or d in skipped]
                if bad:
                    skipped.add(n)
                    changed = True
                    ev(f"QUEUE_SKIP {self.pname} {n} (의존 실패: {','.join(bad)})")
        busy = {g for (_, gs, _) in running.values() for g in gs} | {g for gs in external.values() for g in gs}
        free = [g for g in range(NGPU) if g not in busy]
        ncpu = sum(1 for (_, gs, j) in running.values() if j["gpus"] == 0)
        reserve = False
        for j in jobs:
            n = j["name"]
            if n in done or n in failed or n in skipped or n in running or n in external:
                continue
            if not all(d in done for d in j["deps"]):
                continue
            if not all(d in done or d in failed or d in skipped for d in j["after"]) or any(d in self.retry_at for d in j["after"]):
                continue
            if (j["gpus"] or j["io"]) and not disk_ok(j.get("disk_gb", 0)):
                if n not in self.disk_wait:
                    self.disk_wait.add(n)
                    ev(f"WAIT {n}: 디스크 보류/예측 초과 (필요 {j.get('disk_gb', 0)} GB)")
                continue
            self.disk_wait.discard(n)
            need = j["gpus"]
            if need == 0:
                if ncpu >= MAX_CPU:
                    continue
                gs = []
                ncpu += 1
            else:
                if (reserve and not j["short"]) or len(free) < need:
                    if need == NGPU:
                        reserve = True
                    continue
                gs, free = free[:need], free[need:]
            running[n] = (self.launch(j, gs), gs, j)
        pending = [j["name"] for j in jobs if j["name"] not in done | failed | skipped]
        if not pending and not running and not external and not self.retry_at:
            ev(f"QUEUE_DONE {self.pname} (done {len(done)}, failed {sorted(failed)}, skipped {len(skipped)})")
            return False
        if not running and not external and not self.retry_at and all(not all(d in done for d in j["deps"]) for j in jobs if j["name"] in pending):
            ev(f"QUEUE_STOP {self.pname}: 진행 불가 (대기 {pending[:5]})")
            return False
        if time.time() - self.last_hb > HB_SEC:               # 심장박동: run_job.sh의 STALL 감시가 스케줄러 정지를 잡도록
            print(time.strftime("%F %T"), "running", sorted(running), "done", len(done), "pending", len(pending), flush=True)
            self.last_hb = time.time()
        return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    s = Scheduler(argv[0])
    while s.tick():
        time.sleep(POLL)


if __name__ == "__main__":
    main()
