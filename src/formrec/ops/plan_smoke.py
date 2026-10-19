"""스모크 계획 — 실제 계획과 같은 생성기(build(smoke=True)): 최소 데이터 · 몇 스텝으로 모든 단계.
일부 단계만: 환경변수 FORMREC_SMOKE_ONLY=S_xc (이름 접두) — 그 단계 작업만, 의존은 그 안에서만."""
import importlib
import os

from formrec.ops import plan

importlib.reload(plan)
NAME = "smoke"
JOBS = plan.build(smoke=True)
if os.environ.get("FORMREC_SMOKE_ONLY"):
    JOBS = plan.only(JOBS, os.environ["FORMREC_SMOKE_ONLY"])
