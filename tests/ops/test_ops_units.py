"""formrec.ops 단위 · 스모크 테스트 — 임시 작업 공간에서 가짜 작업만(실제 GPU 작업 · 실제 디스크 정리 없음).

jobsched 스모크는 실제 scripts/run_job.sh로 CPU 명령(echo · exit)만 돌린다. GPU 작업은 잠금 · CUDA_VISIBLE_DEVICES 배정만 확인
(FORMREC_UTIL_POLL=0으로 nvidia-smi도 부르지 않음). disk_guard는 임시 디렉터리의 가짜 산출물만 지운다.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from formrec.common import workspace as cfg
from formrec.ops import disk_guard, jobsched, plan

PATH_RE = re.compile(r"(^|[\s;&|(])(work/|envs/)")


# ── 계획
@pytest.mark.parametrize("smoke", [False, True])
def test_plan_structure(smoke):
    jobs = jobsched.normalize(plan.build(smoke=smoke))
    names = [j["name"] for j in jobs]
    assert all(n.startswith("S_" if smoke else "X_") for n in names)
    assert not any(n.endswith("_report") for n in names)
    for j in jobs:
        assert not PATH_RE.search(j["cmd"]), j["cmd"]           # 옛 work/ · envs/ 경로 없음
        assert "/srv" not in j["cmd"] and "/home" not in j["cmd"]
        for mod in re.findall(r"-m (formrec\.[\w.]+)", j["cmd"]):
            assert mod in plan.MODULES.values() or mod.startswith("formrec.retrieval"), mod
        assert j["disk_gb"] >= 0
    if smoke:
        assert names[-1] == "S_cleanup" and set(jobs[-1]["after"]) == set(names[:-1])
    gp = {j["name"]: j["gpus"] for j in jobs}
    assert gp[("S_" if smoke else "X_") + "pretrain"] == 4


def test_plan_only_and_plan_files(monkeypatch):
    sub = plan.only(plan.build(smoke=True), "S_xc")
    names = {j["name"] for j in sub}
    assert names and all(n.startswith("S_xc") for n in names)
    assert all(set(j["deps"]) | set(j["after"]) <= names for j in sub)
    n, jobs = jobsched.load_plan("formrec.ops.plan_full")
    assert n == "x" and len(jobs) == len(plan.build(smoke=False))
    monkeypatch.setenv("FORMREC_SMOKE_ONLY", "S_xc")
    n, jobs = jobsched.load_plan("formrec.ops.plan_smoke")
    assert n == "smoke" and {j["name"] for j in jobs} == names


# ── 스케줄러 스모크 (실제 run_job.sh, 가짜 작업)
@pytest.fixture
def sched_ws(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "logs").mkdir(parents=True)
    for k, v in dict(ROOT=ws, EVENTS=ws / "logs/events.log", HOLD=ws / "logs/HOLD_DISK", POLL=0.2, AUTO_RETRY=0, RETRY_SEC=1, HB_SEC=0,
                     RUNJOB=str(cfg.REPO / "scripts/run_job.sh")).items():
        monkeypatch.setattr(jobsched, k, v)
    monkeypatch.setattr(jobsched, "disk_usage", lambda: (50, 100))
    monkeypatch.setenv("FORMREC_UTIL_POLL", "0")
    monkeypatch.delenv("FORMREC_SLICE", raising=False)
    return ws


def run_until_done(s, timeout=120):
    t0 = time.time()
    while s.tick():
        assert time.time() - t0 < timeout, "스케줄러가 끝나지 않음"
        time.sleep(0.2)


def test_jobsched_smoke(sched_ws):
    ws = sched_ws
    pf = ws / "plan_t.py"
    pf.write_text('''JOBS = [
    dict(name="T_a", cmd="echo DONE a > a.txt"),
    dict(name="T_b", cmd="echo $CUDA_VISIBLE_DEVICES > b.txt", gpus=1, deps=["T_a"]),
    dict(name="T_c", cmd="echo $CUDA_VISIBLE_DEVICES > c.txt", gpus=4, after=["T_b"]),
    dict(name="T_d", cmd="exit 3"),
    dict(name="T_e", cmd="touch e.txt", deps=["T_d"]),
    dict(name="T_f", cmd="touch f.txt", after=["T_d"]),
    dict(name="T_g", cmd="echo $FORMREC_JOB > g.txt", io=True, deps=["T_a"]),
]
''')
    s = jobsched.Scheduler(str(pf))
    assert s.pname == "t"
    run_until_done(s)
    done = {p.stem for p in (ws / "logs/sched/t").glob("*.done")}
    assert done == {"T_a", "T_b", "T_c", "T_f", "T_g"}
    assert (ws / "c.txt").read_text().strip() == "0,1,2,3" and re.fullmatch(r"[0-3]", (ws / "b.txt").read_text().strip())
    assert (ws / "g.txt").read_text().strip() == "T_g" and not (ws / "e.txt").exists()
    ev = (ws / "logs/events.log").read_text()
    assert "QUEUE_SKIP t T_e (의존 실패: T_d)" in ev and "FAIL T_d rc=3" in ev and "QUEUE_DONE t (done 5, failed ['T_d'], skipped 1)" in ev
    assert re.search(r"START T_c pid=\d+ gpu=0,1,2,3", ev) and "END T_a rc=0" in ev
    assert (ws / "logs/jobs/T_a.log").exists() and list((ws / "logs/snap").glob("*.tgz"))
    # 손으로 다시: .retry → 실패 · 건너뜀이 풀리고 다시 실행(명령을 고친 계획으로)
    pf.write_text(pf.read_text().replace('cmd="exit 3"', 'cmd="true"'))
    (ws / "logs/sched/t/T_d.retry").write_text("")
    s = jobsched.Scheduler(str(pf))
    s.failed.add("T_d"); s.skipped.add("T_e")
    run_until_done(s)
    assert {"T_d", "T_e"} <= {p.stem for p in (ws / "logs/sched/t").glob("*.done")} and (ws / "e.txt").exists()


def test_disk_ok(sched_ws, monkeypatch):
    G = 2 ** 30
    monkeypatch.setattr(jobsched, "disk_usage", lambda: (94 * G, 100 * G))
    assert jobsched.disk_ok(0) and jobsched.disk_ok(1) and not jobsched.disk_ok(2)    # 작은 쓰기: 예측 < 95.5
    assert not jobsched.disk_ok(6)                                                     # 학습: 예측 100 > 95
    monkeypatch.setattr(jobsched, "disk_usage", lambda: (50 * G, 100 * G))
    assert jobsched.disk_ok(6)
    jobsched.HOLD.write_text("x")
    assert not jobsched.disk_ok(6) and jobsched.disk_ok(1)                              # 보류 중: 학습만 대기


def test_disk_wait_event(sched_ws, monkeypatch):
    ws = sched_ws
    monkeypatch.setattr(jobsched, "disk_usage", lambda: (99, 100))
    pf = ws / "plan_w.py"
    pf.write_text('JOBS = [dict(name="T_w", cmd="true", gpus=1, disk_gb=6)]\n')
    s = jobsched.Scheduler(str(pf))
    assert s.tick() and s.tick() and not s.running
    assert (ws / "logs/events.log").read_text().count("WAIT T_w: 디스크") == 1


# ── 디스크 보호 (임시 작업 공간의 가짜 산출물)
@pytest.fixture
def guard_ws(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "logs/sched/x").mkdir(parents=True)
    for k, v in dict(ROOT=ws, LOG=ws / "logs/disk_guard.log", EV=ws / "logs/events.log", HOLD=ws / "logs/HOLD_DISK", STATE=ws / "logs/disk_guard_state.json").items():
        monkeypatch.setattr(disk_guard, k, v)
    monkeypatch.setattr(disk_guard, "smoke_running", lambda: False)
    monkeypatch.setattr(disk_guard, "DRY", False)
    rd = ws / "runs/xa/pvl_ft"
    for e in ("export/ep1", "export/ep2", "checkpoint-1", "checkpoint-2", "checkpoint-3"):
        (rd / e).mkdir(parents=True)
        (rd / e / "w.bin").write_bytes(b"0" * 1024)
    os.symlink("ep2", rd / "export/best")
    (rd / "best.json").write_text(json.dumps({"best_epoch": 1}))
    g = ws / "results/xa/g1"
    g.mkdir(parents=True)
    for i, em in ((1, 0.5), (2, 0.9), (3, 0.7)):
        n = 10
        pd.DataFrame(dict(pred=["a"] * n, text=["a"] * int(em * n) + ["b"] * (n - int(em * n)))).to_parquet(g / f"pvlft_ep{i}_val.parquet")
    (ws / "ft/xa/ovis").mkdir(parents=True)
    (ws / "ft/xa/ovis/train.jsonl").write_text("{}\n")
    for j in ("X_xa_ovis", "X_xa_pvl"):
        (ws / f"logs/sched/x/{j}.done").write_text("")
    return ws


def test_disk_guard_dry(guard_ws, monkeypatch):
    ws = guard_ws
    monkeypatch.setattr(disk_guard, "DRY", True)
    def files():
        return sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if not str(p.relative_to(ws)).startswith("logs/"))
    before = files()
    disk_guard.level1(); disk_guard.level2()
    st = disk_guard.guard(lambda: (97.5, 1.0))
    assert files() == before and "runs/xa/pvl_ft/checkpoint-3" in before
    log = (ws / "logs/disk_guard.log").read_text()
    assert "(dry) 삭제 runs/xa/pvl_ft/export/ep1" in log and "(dry) 삭제 runs/xa/pvl_ft/checkpoint-3" in log
    assert not (ws / "logs/HOLD_DISK").exists() and st["actions"][-1] == "L5 kill all"


def test_disk_guard_levels(guard_ws, monkeypatch):
    ws = guard_ws
    killed = []
    monkeypatch.setattr(disk_guard, "kill", lambda names, why: killed.append((tuple(names), why)))
    monkeypatch.setattr(disk_guard, "running_jobs", lambda gpu_only=True: ["X_gpu"] if gpu_only else ["X_gpu", "X_cpu"])
    st = disk_guard.guard(lambda: (92.0, 100.0))
    assert st["actions"] == [] and (ws / "runs/xa/pvl_ft/export/ep1").exists()
    disk_guard.guard(lambda: (93.5, 50.0))                                             # L1만
    rd = ws / "runs/xa/pvl_ft"
    assert not (rd / "export/ep1").exists() and (rd / "export/ep2").exists() and not (ws / "ft/xa/ovis/train.jsonl").exists()
    assert (rd / "checkpoint-3").exists()
    st = disk_guard.guard(lambda: (95.2, 20.0))                                        # L1 · L2 · 보류
    assert sorted(p.name for p in rd.glob("checkpoint-*")) == ["checkpoint-1", "checkpoint-2"]   # loss 최선 1 · 지표 최선 2
    pr = json.loads((rd / "pruned.json").read_text())
    assert pr["loss_best"] == 1 and pr["metric_best"] == 2 and pr["deleted"] == ["checkpoint-3"]
    assert (ws / "logs/HOLD_DISK").exists() and "HOLD on" in st["actions"] and not killed
    disk_guard.guard(lambda: (96.5, 5.0))
    assert killed[-1][0] == ("X_gpu",)
    disk_guard.guard(lambda: (97.2, 1.0))
    assert killed[-1][0] == ("X_gpu", "X_cpu")
    st = disk_guard.guard(lambda: (94.0, 30.0))
    assert not (ws / "logs/HOLD_DISK").exists() and "HOLD off" in st["actions"]
    assert json.loads((ws / "logs/disk_guard_state.json").read_text())["pct"] == 94.0


def test_disk_guard_rm_outside(guard_ws, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    with pytest.raises(AssertionError):
        disk_guard.rm(outside, "시험")
    assert outside.exists()


def test_running_jobs(guard_ws):
    ws = guard_ws
    dead = 2 ** 22 + 12345
    (ws / "logs/events.log").write_text(
        f"[10-01 00:00:00] START X_a pid={os.getpid()} gpu=0 io=0 :: x\n[10-01 00:00:00] START X_b pid={os.getpid()} gpu=none io=0 :: x\n"
        f"[10-01 00:00:00] START X_c pid={dead} gpu=1 io=0 :: x\n[10-01 00:00:00] START K_z pid={os.getpid()} gpu=2 io=0 :: x\n")
    assert disk_guard.running_jobs(True) == ["X_a"] and sorted(disk_guard.running_jobs(False)) == ["X_a", "X_b"]


def test_scripts_syntax():
    for sh in ("run_job.sh", "watch.sh", "launch.sh", "kill_job.sh", "smoke_clean.sh"):
        p = cfg.REPO / "scripts" / sh
        assert p.exists() and os.access(p, os.X_OK)
        assert os.system(f"bash -n {p}") == 0
        assert "/srv" not in p.read_text() and "/home" not in p.read_text()


def test_smoke_clean(tmp_path):
    ws = tmp_path / "ws"
    keep = [ws / "runs/xa/pvl_ft/a", ws / "results/xa/g1/pvlft_test.parquet"]
    gone = [ws / "runs/xa/pvl_ft_smoke/a", ws / "results/xa/g1/pvlft_smoke_test.parquet", ws / "ft/xa/pvl/train_smoke.jsonl"]
    for p in keep + gone:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert os.system(f"FORMREC_WORKSPACE={ws} bash {cfg.REPO / 'scripts/smoke_clean.sh'} >/dev/null") == 0
    assert all(p.exists() for p in keep) and not any(p.exists() for p in gone)
    shutil.rmtree(ws)


def test_kill_job_tree(tmp_path):
    """run_job.sh로 띄운 가짜 작업(자식 · 손자 sleep)을 kill_job.sh가 트리째 끝내고 FAIL이 기록되는지."""
    import subprocess
    ws = tmp_path / "ws"
    (ws / "logs").mkdir(parents=True)
    env = dict(os.environ, FORMREC_WORKSPACE=str(ws), FORMREC_UTIL_POLL="0", FORMREC_KILL_WAIT="1")
    env.pop("FORMREC_SLICE", None)
    p = subprocess.Popen([str(cfg.REPO / "scripts/run_job.sh"), "T_kill", "--", "bash", "-c", "sleep 300 & sleep 300; wait"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    ev = ws / "logs/events.log"
    for _ in range(100):
        if ev.exists() and "START T_kill" in ev.read_text():
            break
        time.sleep(0.1)
    pid = int(re.search(r"START T_kill pid=(\d+)", ev.read_text()).group(1))
    r = subprocess.run(["bash", str(cfg.REPO / "scripts/kill_job.sh"), "T_kill"], env=env, capture_output=True, text=True, timeout=60)
    assert "종료: T_kill" in r.stdout
    p.wait(timeout=30)
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert re.search(r"FAIL T_kill rc=\d+", ev.read_text())
