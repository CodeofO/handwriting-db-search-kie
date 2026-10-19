"""실제 실험 계획 — 생성기 formrec.ops.plan.build(smoke=False). jobsched가 매 주기 다시 읽는다(생성기 수정도 반영되게 reload)."""
import importlib

from formrec.ops import plan

importlib.reload(plan)
NAME = "x"                      # 기록 폴더 logs/sched/x (디스크 보호 · 감시 루프가 이 이름을 본다)
JOBS = plan.build(smoke=False)
