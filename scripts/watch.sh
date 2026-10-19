#!/usr/bin/env bash
# 무인 운영 감시 루프: 2분마다 디스크 보호, 6분마다 실제 실험 스케줄러 생존 확인 · 재기동(실제 실험 시작 뒤 · 계획이 끝나지 않았을 때만).
# 사용: nohup bash scripts/watch.sh >> $FORMREC_WORKSPACE/logs/watch.out 2>&1 &
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${FORMREC_WORKSPACE:-$REPO/workspace}" || exit 1
export FORMREC_WORKSPACE="$PWD" PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PB="${FORMREC_PY_BASE:-$PWD/envs/base/bin/python}"; n=0
mkdir -p logs
while true; do
  nice -n 10 "$PB" -m formrec.ops.disk_guard >> logs/disk_guard.out 2>&1 || true
  if (( n % 3 == 0 )) && [[ -f logs/x_production_started ]]; then
    ls_=$(grep -an "QUEUE_START x " logs/events.log | tail -1 | cut -d: -f1); ld_=$(grep -an "QUEUE_DONE x " logs/events.log | tail -1 | cut -d: -f1)
    if ! pgrep -f "formrec.ops.jobsched formrec.ops.plan_full" >/dev/null && [[ -z "$ld_" || "${ls_:-0}" -gt "${ld_:-0}" ]]; then
      echo "[$(date +%m-%d\ %H:%M:%S)] EVENT watch: 실제 실험 스케줄러가 없어 다시 띄움" >> logs/events.log
      nohup "$REPO/scripts/run_job.sh" sched_x --stall-min 60 -- "$PB" -m formrec.ops.jobsched formrec.ops.plan_full >/dev/null 2>&1 &
    fi
  fi
  n=$((n + 1)); sleep "${FORMREC_WATCH_SEC:-120}"
done
