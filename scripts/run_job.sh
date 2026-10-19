#!/usr/bin/env bash
# 작업 래퍼. 스케줄러(formrec.ops.jobsched)가 띄우는 모든 작업은 이것으로 실행한다.
#   scripts/run_job.sh <name> [--gpu | --gpus 0,1] [--io] [--stall-min N] [--cores 0-63] -- <command...>
# - 작업 공간 = $FORMREC_WORKSPACE (기본: 저장소/workspace). 기록은 그 아래 logs/.
# - Python 환경 기본값: FORMREC_PY_{BASE,TRAIN,PADDLE,VLLM,PF,SWIFT} = $FORMREC_WORKSPACE/envs/<이름>/bin/python. PYTHONPATH에 저장소 src를 더한다.
# - CPU 상한(선택): FORMREC_SLICE=<systemd 사용자 slice 이름>이면 그 slice 안에서 실행(CPUQuota · IOWeight · MemoryMax는 slice 설정, docs/operations.md).
# - nice 10. --io 작업은 ionice idle(-c3), 그 외 best-effort 최저(-c2 -n7).
# - GPU 잠금은 GPU별(logs/locks/gpu{k}.lock, fd 10+k, 번호 오름차순으로 잡아 교착 방지).
#   --gpu = --gpus 0,1,2,3 (학습: 4장). --gpus k = 1장(평가 · 추론, GPU별로 동시 실행 가능). CUDA_VISIBLE_DEVICES를 잡은 GPU로 설정한다.
# - --io : 대용량 IO 잠금(logs/locks/io.lock). 대용량 읽기 · 쓰기 작업은 한 번에 하나.
# - 자식 환경변수 KIE_JOB=<name>(FORMREC_JOB도 같음). 분산 실행(paddle.distributed.launch)은 --log_dir logs/pdist/$KIE_JOB 로 두면
#   rank별 workerlog.0~3 도 에러 감시 대상이 된다.
# - FORMREC_UTIL_POLL=0 이면 GPU 사용률 기록(nvidia-smi)을 끈다(시험용).
#
# logs/events.log 알림
#   START / WAIT(잠금 대기) / END / FAIL(rc≠0, 마지막 로그 3줄)
#   ALERT    : 실시간 에러 줄(OOM, CUDA/NCCL/Xid, Traceback, NaN loss 등). 작업당 최대 8줄
#   EVENT    : 작업이 스스로 찍는 중간 보고(로그의 "EVENT ..." 줄, Paddle "best metric" 줄)
#   PROGRESS : 30분마다 경과 시간 · GPU util · 로그 마지막 줄
#   ALERT util: GPU 작업인데 시작 10분 뒤부터 5분 평균 util이 5% 미만인 GPU가 있으면 (멈춘 rank, 데이터로더 정지 등)
#   STALL    : 로그가 stall-min분 동안 그대로
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"; ROOT="$(cd "$ROOT" && pwd)"
export FORMREC_REPO="$REPO" FORMREC_WORKSPACE="$ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
for e in base train paddle vllm pf swift; do
  v="FORMREC_PY_${e^^}"; [[ -z "${!v:-}" ]] && export "$v=$ROOT/envs/$e/bin/python"
done
EV="$ROOT/logs/events.log"
mkdir -p "$ROOT/logs/jobs" "$ROOT/logs/util" "$ROOT/logs/locks" "$ROOT/logs/pdist"

name="$1"; shift
use_gpu=0; gpus=""; use_io=0; stall_min=20; cores=""
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --gpu) use_gpu=1; gpus="0,1,2,3"; shift ;;
    --gpus) use_gpu=1; gpus="$2"; shift 2 ;;
    --io) use_io=1; shift ;;
    --stall-min) stall_min="$2"; shift 2 ;;
    --cores) cores="$2"; shift 2 ;;
    *) echo "unknown opt $1" >&2; exit 2 ;;
  esac
done
shift  # --
LOG="$ROOT/logs/jobs/$name.log"
PD="$ROOT/logs/pdist/$name"
ev() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$EV"; }

# 잠금 (필요한 것만). fd 10+k=gpu k, 9=io. 대기하면 WAIT 이벤트.
GLIST=()
if (( use_gpu )); then
  IFS=',' read -r -a GLIST <<< "$gpus"
  mapfile -t GLIST < <(printf '%s\n' "${GLIST[@]}" | sort -n)
  for g in "${GLIST[@]}"; do
    fd=$((10 + g))
    eval "exec $fd>\"$ROOT/logs/locks/gpu$g.lock\""
    flock -n $fd || { ev "WAIT $name: gpu$g lock"; flock $fd; }
  done
  export CUDA_VISIBLE_DEVICES="$gpus"
fi
if (( use_io )); then
  exec 9>"$ROOT/logs/locks/io.lock"
  flock -n 9 || { ev "WAIT $name: io lock"; flock 9; }
fi

export PYTHONUNBUFFERED=1 KIE_JOB="$name" FORMREC_JOB="$name"
# 코드 스냅샷: 이 작업이 쓴 코드(src · scripts · configs)의 해시와 사본 (결과 ↔ 코드 버전 추적; 대기 중 코드가 바뀌어도 기록이 남음)
mkdir -p "$ROOT/logs/snap"
CODE_SHA=$(cd "$REPO" && find src scripts configs -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' \) -not -path '*/__pycache__/*' 2>/dev/null \
  | LC_ALL=C sort | xargs cat 2>/dev/null | sha1sum | cut -c1-10)
[[ -f "$ROOT/logs/snap/$CODE_SHA.tgz" ]] || tar czf "$ROOT/logs/snap/$CODE_SHA.tgz" -C "$REPO" --exclude=__pycache__ src scripts configs 2>/dev/null
export KIE_CODE_SHA="$CODE_SHA" FORMREC_CODE_SHA="$CODE_SHA"
[[ -z "${OMP_NUM_THREADS:-}" ]] && export OMP_NUM_THREADS=8
if (( use_io )); then IONICE=(ionice -c3); else IONICE=(ionice -c2 -n7); fi
: > "$LOG"; rm -rf "$PD"; mkdir -p "$PD"
# 잠금 fd는 자식에게 넘기지 않는다: 자식이 남아 잠금을 잡고 있는 문제 방지
TS=(); [[ -n "$cores" ]] && TS=(taskset -c "$cores")
SCOPE="formrec-$name-$$"
if [[ -n "${FORMREC_SLICE:-}" ]]; then
  systemd-run --user --scope --quiet --slice="$FORMREC_SLICE" --unit="$SCOPE" -- "${TS[@]}" nice -n 10 "${IONICE[@]}" "$@" >> "$LOG" 2>&1 9>&- 10>&- 11>&- 12>&- 13>&- &
else
  "${TS[@]}" nice -n 10 "${IONICE[@]}" "$@" >> "$LOG" 2>&1 9>&- 10>&- 11>&- 12>&- 13>&- &
fi
pid=$!
t0=$(date +%s)
# 래퍼가 신호로 죽을 때도 알림 (조용히 사라지지 않게) + 자식 정리
on_kill() {
  ev "FAIL $name rc=killed dur=$(( $(date +%s) - t0 ))s :: 래퍼가 신호를 받음 → 작업 종료"
  [[ -n "${FORMREC_SLICE:-}" ]] && systemctl --user stop "$SCOPE.scope" 2>/dev/null
  kill $pid 2>/dev/null
  [[ -n "${util_pid:-}" ]] && kill $util_pid 2>/dev/null; [[ -n "${watch_pid:-}" ]] && kill $watch_pid 2>/dev/null
  exit 143
}
trap on_kill TERM INT HUP
ev "START $name pid=$pid gpu=${gpus:-none} io=$use_io code=$CODE_SHA${FORMREC_SLICE:+ slice=$FORMREC_SLICE}${cores:+ cores=$cores} :: $*"

UTIL="$ROOT/logs/util/$name.csv"; : > "$UTIL"; util_pid=""
if (( use_gpu )) && [[ "${FORMREC_UTIL_POLL:-1}" != 0 ]]; then
  nvidia-smi -i "$gpus" --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 >> "$UTIL" 2>/dev/null 9>&- 10>&- 11>&- 12>&- 13>&- &
  util_pid=$!
fi

# 실시간 에러 감시: 작업 로그 + 분산 rank 로그. 종류와 무관하게 작업당 최대 8줄
ERR_RE="out of memory|OutOfMemory|ResourceExhausted|CUDA error|cudaError|CUBLAS_STATUS|CUDNN_STATUS|illegal memory access|device-side assert|Xid|ECC error|NCCL (WARN|error)|nccl(System|Internal|Unhandled)|collective operation timeout|Traceback|Error:|Exception:|nan loss|loss[^a-z]*\bnan\b|Killed|Segmentation fault|core dumped|Aborted|error happened|EngineDeadError|died unexpectedly|FAILED"
( tail -n +1 -q -F --pid=$pid "$LOG" "$PD"/workerlog.{0,1,2,3} 2>/dev/null \
  | grep --line-buffered -E -i "$ERR_RE" | grep --line-buffered -v -E "NCCL INFO|UserWarning|FutureWarning|DeprecationWarning" \
  | awk -v n="$name" 'BEGIN{c=0} {c++; if (c<=8) {cmd="date \"+%m-%d %H:%M:%S\""; cmd | getline t; close(cmd); print "[" t "] ALERT " n ": " substr($0,1,300); fflush()}}' >> "$EV" ) 9>&- 10>&- 11>&- 12>&- 13>&- &

# 중간 보고 전달: 작업이 찍는 "EVENT ..." 줄, Paddle 평가 결과 줄
( tail -n +1 -F --pid=$pid "$LOG" 2>/dev/null \
  | grep --line-buffered -E "^EVENT |best metric, " \
  | awk -v n="$name" '{cmd="date \"+%m-%d %H:%M:%S\""; cmd | getline t; close(cmd); sub(/^EVENT /,""); print "[" t "] EVENT " n ": " substr($0,1,300); fflush()}' >> "$EV" ) 9>&- 10>&- 11>&- 12>&- 13>&- &

# 정지 감시(STALL) + 30분마다 PROGRESS + GPU 저사용 ALERT
( last_prog=$(date +%s); last_util_alert=0; PROG=${KIE_PROG_SEC:-1800}; GRACE=${KIE_UTIL_GRACE:-600}
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    age=$(( now - $(stat -c %Y "$LOG") ))
    if (( age > stall_min * 60 )); then ev "STALL $name: log idle ${age}s"; touch "$LOG"; fi
    ugpu=""
    if (( use_gpu )); then
      ng=${#GLIST[@]}
      ugpu=$(tail -n $((60 * ng)) "$UTIL" 2>/dev/null | awk -F', *' 'NF>=3 {n[$1]++; u[$1]+=$2} END {for (g in n) printf "g%s:%d%% ", g, u[g]/n[g]}')
      if (( now - t0 > GRACE && now - last_util_alert > 1800 )); then
        low=$(tail -n $((60 * ng)) "$UTIL" 2>/dev/null | awk -F', *' 'NF>=3 {n[$1]++; u[$1]+=$2} END {for (g in n) if (n[g] >= 30 && u[g]/n[g] < 5) printf "g%s ", g}')
        if [[ -n "$low" ]]; then ev "ALERT $name: GPU util <5% for 5min on ${low}(${ugpu}) — 정지 · 데이터로더 병목 의심"; last_util_alert=$now; fi
      fi
    fi
    if (( now - last_prog >= PROG )); then
      ev "PROGRESS $name: $(( (now - t0) / 60 ))min ${ugpu:+util[${ugpu% }] }:: $(grep -v '^\s*$' "$LOG" | tail -n 1 | tr -d '\r' | cut -c1-200)"
      last_prog=$now
    fi
  done ) 9>&- 10>&- 11>&- 12>&- 13>&- &
watch_pid=$!

wait $pid; rc=$?
dur=$(( $(date +%s) - t0 ))
kill $watch_pid 2>/dev/null; [[ -n "$util_pid" ]] && kill $util_pid 2>/dev/null
for g in "${GLIST[@]}"; do flock -u $((10 + g)); done; (( use_io )) && flock -u 9
usum=$(awk -F', *' 'NF>=3 {n[$1]++; u[$1]+=$2; if ($3>m[$1]) m[$1]=$3} END {for (g in n) printf "g%s:%d%%/%.0fG ", g, u[g]/n[g], m[g]/1024}' "$UTIL" 2>/dev/null)
if (( rc == 0 )); then
  last=$(grep -E '^DONE' "$LOG" | tail -n 1); [[ -z "$last" ]] && last=$(grep -v '^\s*$' "$LOG" | tail -n 1)
  ev "END $name rc=0 dur=${dur}s util[${usum}] :: $(echo "$last" | cut -c1-250)"
else
  ev "FAIL $name rc=$rc dur=${dur}s util[${usum}] :: $(grep -v '^\s*$' "$LOG" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"
fi
exit $rc
