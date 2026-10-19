#!/usr/bin/env bash
# 스케줄러 작업 하나를 손자 프로세스(분산 학습 워커 · 데이터로더)까지 모두 종료한다.
# 자식만 끝내면(pkill -P) 학습 워커가 GPU 메모리를 쥔 채 남아 다음 vLLM 평가가 시작하지 못한다 → 프로세스 트리 전체를 끈다.
# 사용: scripts/kill_job.sh <작업 이름>   (logs/events.log의 마지막 START 줄 pid 기준)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
N="$1"
P=$(grep "START $N " "$ROOT/logs/events.log" | tail -1 | grep -o "pid=[0-9]*" | cut -d= -f2)
[[ -n "$P" ]] || { echo "작업 $N 의 START 기록 없음"; exit 1; }
tree() { local p=$1; echo "$p"; for c in $(pgrep -P "$p"); do tree "$c"; done; }
PIDS=$(tree "$P")
echo "종료: $N (pid $P) 프로세스 $(echo "$PIDS" | wc -w)개"
kill $PIDS 2>/dev/null; sleep "${FORMREC_KILL_WAIT:-10}"
LEFT=$(for p in $PIDS; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
[[ -n "$LEFT" ]] && { echo "남은 프로세스 강제 종료: $LEFT"; kill -9 $LEFT 2>/dev/null; }
command -v nvidia-smi >/dev/null && [[ "${FORMREC_UTIL_POLL:-1}" != 0 ]] && { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo; }
exit 0
