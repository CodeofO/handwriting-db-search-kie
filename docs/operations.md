# 실험 운영 (스케줄러 · 작업 래퍼 · 디스크 보호)

긴 실험(학습 30개 · 평가 100여 개)을 사람 없이 돌리기 위한 도구입니다. 코드는 `src/formrec/ops/`, 셸 도구는 `scripts/`.

| 구성 | 역할 |
|---|---|
| `formrec.ops.plan` | 실험 계획 생성기 `build(smoke)` — 작업 이름 · 명령 · GPU 수 · 의존(deps / after) · 예상 쓰기 GB · 정지 알림 기준 |
| `formrec.ops.plan_full` / `plan_smoke` | 실제 계획(기록 이름 `x`) / 스모크 계획(`smoke`, 최소 데이터 · 몇 스텝). `FORMREC_SMOKE_ONLY=S_xc` = 그 단계만 |
| `formrec.ops.jobsched` | GPU 인지 DAG 스케줄러. 계획을 매 주기 다시 읽고, `logs/sched/<계획>/<작업>.done`으로 이어 하기 |
| `scripts/run_job.sh` | 작업 래퍼: GPU별 · IO 잠금, nice/ionice, 코드 스냅샷, `logs/events.log` 알림(START · END · FAIL · ALERT · STALL · PROGRESS) |
| `formrec.ops.disk_guard` | 디스크 사용률 단계별 정리 · 보류 · 중단 |
| `scripts/watch.sh` | 감시 루프: 2분마다 디스크 보호, 6분마다 실제 스케줄러 생존 확인 · 재기동 |
| `scripts/launch.sh` | 감시 루프 → 스모크 → (실패 없으면) 실제 실험 |
| `scripts/kill_job.sh` | 작업 하나를 프로세스 트리째 종료(분산 학습 워커가 GPU 메모리를 쥐고 남지 않게) |
| `scripts/smoke_clean.sh` | 스모크 산출물만 정리(스모크 계획의 마지막 작업) |

## 시작

```bash
export FORMREC_WORKSPACE=/path/to/workspace          # 데이터 · 결과 · 가중치 · logs/
# (선택) Python 환경이 workspace/envs/<이름>/bin/python 이 아니면
export FORMREC_PY_BASE=... FORMREC_PY_TRAIN=... FORMREC_PY_PADDLE=... FORMREC_PY_VLLM=... FORMREC_PY_PF=... FORMREC_PY_SWIFT=...
nohup bash scripts/launch.sh >> $FORMREC_WORKSPACE/logs/launch.out 2>&1 &
tail -f $FORMREC_WORKSPACE/logs/events.log
```

- 작업 하나만: `scripts/run_job.sh <이름> [--gpus 0 | --gpu] [--io] [--stall-min N] -- <명령…>`
- 실패한 작업 다시: `touch $FORMREC_WORKSPACE/logs/sched/x/<작업>.retry` (그 작업 때문에 건너뛴 작업도 함께 풀림)
- 계획 수정: `src/formrec/ops/plan.py`를 고치면 돌고 있는 스케줄러가 다음 주기에 반영(이미 시작한 이름은 그대로)

## 스케줄링 규칙

- 학습 = GPU 4장, 평가 · 추론 = GPU 1장(빈 GPU마다 동시), CPU 작업은 동시에 2개, 대용량 IO는 한 번에 하나.
- 목록 순서 = 우선순위. 4장 작업이 기다리면 뒤의 GPU 작업은 새로 시작하지 않는다(예약). `short=True` 1장 작업만 예외(백필).
- `deps` = 산출물 의존(실패 → 건너뜀), `after` = 순서만(성공 여부 무관). 평가 실패가 다음 학습을 막지 않는다.
- 실패하면 10분 뒤 1회 자동 재시도(`FORMREC_SCHED_RETRY`, `FORMREC_SCHED_RETRY_SEC`).
- 학습은 모두 이어 학습(`--resume`, 체크포인트)을 지원 — 디스크 보호가 중단해도 재시도로 이어진다.

## 디스크 보호 단계 (`FORMREC_DISK_LEVELS`, 기본 93,94,95,94.5,96,97)

| 사용률 | 조치 |
|---|---|
| ≥ 93 % | 다시 만들 수 있는 파생물 정리: val 채점이 끝난 VLM 에폭 내보내기(최선 제외), 학습이 끝난 데이터셋의 OvisOCR2 base64 학습 파일 · 전처리 캐시, PP-OCRv5 옵티마이저 상태, 실제 실험 시작 뒤의 스모크 잔여물 |
| ≥ 94 % | VLM 체크포인트 정리 — loss 기준 선정과 지표 기준 선정(모든 체크포인트 val 완전 일치)이 모두 기록된 경우에만, 두 선정본을 남기고 삭제(`pruned.json`) |
| ≥ 95 % | 시작 보류(`logs/HOLD_DISK`): 새 학습(예상 쓰기 ≥ 5 GB)만 대기, 평가 · 내보내기는 예측 사용률 < 95.5 %면 계속 (94.5 % 아래로 내려가면 해제) |
| ≥ 96 % | 돌고 있는 GPU 작업 종료(보류 해제 뒤 자동 재시도로 이어 학습) |
| ≥ 97 % | 이 프로젝트의 돌고 있는 작업 모두 종료 |

보류가 평가까지 막으면 체크포인트 정리의 전제(체크포인트별 val 채점)가 영영 안 끝나 공간이 생기지 않는다 → 보류는 큰 쓰기 작업만 막는다.
`python -m formrec.ops.disk_guard --dry` = 지우거나 끄지 않고 할 일만 기록.

## 서버 전용 설정 (저장소에 넣지 않음 — 필요하면 직접)

**CPU · 메모리 상한 (systemd 사용자 slice).** 공용 서버에서 다른 사용자를 굶기지 않도록 작업을 한 slice에 묶는다.
`run_job.sh`는 `FORMREC_SLICE`가 있을 때만 `systemd-run --user --scope --slice=$FORMREC_SLICE`로 실행한다.

```ini
# ~/.config/systemd/user/formrec.slice   (예: 128코어의 50 %, 메모리 200G)
[Slice]
CPUQuota=6400%
IOWeight=50
MemoryMax=200G
```
```bash
systemctl --user daemon-reload && export FORMREC_SLICE=formrec.slice
```

**재부팅 뒤 자동 재개 (crontab).** 감시 루프 · 스케줄러는 프로세스이므로 재부팅하면 사라진다. 이어 하기는 `.done` 기록으로 된다.

```cron
@reboot sleep 180 && cd /path/to/repo && FORMREC_WORKSPACE=/path/to/workspace nohup bash scripts/launch.sh >> /path/to/workspace/logs/launch.out 2>&1
```

**알림 구독.** `logs/events.log`의 START / END / FAIL / ALERT / STALL / WAIT / QUEUE_* 줄을 원하는 알림 도구로 구독한다(예: `tail -F | grep -E "FAIL|ALERT|STALL"`).

## 환경변수

| 변수 | 기본 | 뜻 |
|---|---|---|
| `FORMREC_WORKSPACE` | 저장소/workspace | 작업 공간 |
| `FORMREC_PY_{BASE,TRAIN,PADDLE,VLLM,PF,SWIFT}` | `$FORMREC_WORKSPACE/envs/<이름>/bin/python` | 작업별 Python |
| `FORMREC_NGPU` · `FORMREC_MAX_CPU` | 4 · 2 | GPU 수 · 동시 CPU 작업 수 |
| `FORMREC_SCHED_POLL` · `FORMREC_SCHED_RETRY` · `FORMREC_SCHED_RETRY_SEC` | 5 · 1 · 600 | 주기(초) · 자동 재시도 횟수 · 대기(초) |
| `FORMREC_DISK_LIMIT` · `FORMREC_DISK_SOFT` | 95 · 95.5 | 학습 · 작은 쓰기 작업의 시작 후 예측 사용률 상한 |
| `FORMREC_DISK_LEVELS` | 93,94,95,94.5,96,97 | 디스크 보호 단계(정리 · 체크포인트 정리 · 보류 · 보류 해제 · GPU 작업 중단 · 비상) |
| `FORMREC_SLICE` | (없음) | systemd 사용자 slice |
| `FORMREC_UTIL_POLL` | 1 | 0이면 GPU 사용률 기록(nvidia-smi)을 끈다(시험용) |
| `FORMREC_RUNJOB` | scripts/run_job.sh | 스케줄러가 쓰는 래퍼(시험용으로 바꿀 수 있음) |

옛 이름(`KIE_SCHED_*`, `KIE_DISK_*`, `KIE_RUNJOB`)도 받는다. 작업 안에서는 `KIE_JOB`(= `FORMREC_JOB`, 작업 이름) · `KIE_CODE_SHA`(코드 스냅샷 해시)를 쓸 수 있다.
