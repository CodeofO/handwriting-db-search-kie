#!/usr/bin/env bash
# 실행기: 감시 루프 → 스모크(모든 단계, 최소 데이터) → 실패 없으면 실제 실험 시작(logs/x_production_started 표시).
# 스모크가 실패하면 실제 실험을 시작하지 않고 ALERT를 남긴다(고친 뒤 다시 실행 → 끝난 스모크 단계는 .done으로 건너뜀).
# 사용: nohup bash scripts/launch.sh >> $FORMREC_WORKSPACE/logs/launch.out 2>&1 &
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${FORMREC_WORKSPACE:-$REPO/workspace}" || exit 1
export FORMREC_WORKSPACE="$PWD" PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PB="${FORMREC_PY_BASE:-$PWD/envs/base/bin/python}"
mkdir -p logs
ev() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> logs/events.log; }
pgrep -f "bash $REPO/scripts/watch.sh" >/dev/null || { nohup bash "$REPO/scripts/watch.sh" >> logs/watch.out 2>&1 & ev "EVENT launch: 감시 루프 시작"; }
if [[ ! -f logs/x_production_started ]]; then
  ev "EVENT launch: 스모크 시작"
  n0=$(wc -l < logs/events.log)                         # 이번 실행에서 나온 기록만 본다(지난 스모크 결과를 통과로 읽지 않게)
  "$REPO/scripts/run_job.sh" sched_smoke --stall-min 60 -- "$PB" -m formrec.ops.jobsched formrec.ops.plan_smoke; rc=$?
  last=$(tail -n +"$((n0 + 1))" logs/events.log | grep "QUEUE_DONE smoke " | tail -1)
  if [[ $rc -ne 0 || "$last" != *"failed []"* || "$last" != *"skipped 0)"* ]]; then
    ev "ALERT launch: 스모크 실패 — 실제 실험 시작 안 함 :: ${last:-QUEUE_DONE 없음}"; exit 1
  fi
  ev "EVENT launch: 스모크 통과 → 실제 실험 시작"
  touch logs/x_production_started
fi
pgrep -f "formrec.ops.jobsched formrec.ops.plan_full" >/dev/null || nohup "$REPO/scripts/run_job.sh" sched_x --stall-min 60 -- "$PB" -m formrec.ops.jobsched formrec.ops.plan_full >/dev/null 2>&1 &
ev "EVENT launch: 실제 실험 스케줄러 실행"
