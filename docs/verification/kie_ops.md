# 검증 기록 — 영역 C (손글씨 KIE · 일반 KIE · 실험 운영)

원본 연구 코드(`work/xk_*.py`, `xkg_*.py`, `jobsched.py`, `x_disk_guard.py`, `xplan.py`, 셸 도구)를 `formrec.kie`, `formrec.kie_general`, `formrec.ops`, `scripts/`로 옮긴 뒤
같은 입력에서 같은 출력이 나오는지 대조했다. 모두 CPU(`CUDA_VISIBLE_DEVICES=""`), 원본 작업 공간은 읽기만 했다(쓰는 CLI는 임시 작업 공간에서만).

재현: `LEGACY_WORK=<원본 work> FORMREC_WORKSPACE=<원본 작업 공간> pytest tests/legacy/test_kie_legacy.py tests/legacy/test_ops_legacy.py`
(일반 KIE 대조 `tests/legacy/test_kie_general_legacy.py`는 cv2가 있는 paddle 환경). 둘 중 하나라도 없으면 건너뛴다.

## 1. 손글씨 KIE (`formrec.kie`)

| 대상 | 입력 | 비교 | 결과 |
|---|---|---|---|
| 채점 `evaluate.run(--db --auto)`, xkp | 저장 예측 7종(crop+OCR 3 · VLM 4) + `meta/x/xkp_queries_{tune,test}` | 원본 `results/xk/summary_xkp_test.csv` | 26행 × 수치 15열, 최대 \|차이\| 1.1e-16 (CSV 부동소수 읽기 오차), NaN 위치 · 확신도 종류(conf) 문자열 동일 |
| 같음, xk | 저장 예측 5종 + `xk_queries` | 원본 `summary_xk_test.csv` | 20행 × 15열, 최대 \|차이\| 1.1e-16, NaN · conf 동일 |
| 레코드 DB `record_db()` (시드 0, 이웃 6) | `meta/x/xk_items.parquet` | 원본 `xk_eval.record_db()` | 182,217행(실제 27,301) `DataFrame.equals` = True |
| DB 밖 `oov.run` | 위 + 빠짐 0 · 5 · 20 % (시드 1) | 원본 `results/xk/oov_xkp.csv` | 9행, 최대 \|차이\| 1.1e-16, τ 동일, NaN 동일 |
| 키 순서 `order.order_metrics` | `pvlkiep_test` · `pvlkiepperm_test` | 원본 `order_pvlkiep.json` | dict 완전 일치 (n 442, 일관성 0.9457, 결합 오류 0.0, 0.7534 / 0.7421) |
| 우도 기반 검증 후보 `verify.candidates` (K 8, 풀 64) | `ovkiep_{tune,test}` + 레코드 DB | 원본 `results/xk/lrv/cands_xkp_ovkiep_{tune,test}.parquet` | tune 864 · test 585 질의 — qid · keys · free · cand · cand_src · cand_d 모두 동일 |
| 질의 생성 `data.queries` | 저장 분할표 `xk_items` · `xkp_items` | 저장 `{xk,xkp}_queries_{train,val,tune,test}` | xk 53,223 / 2,199 / 2,262 / 1,335, xkp 11,586 / 732 / 864 / 585 — 8개 파일 모두 `equals` |
| xk 분할표(세로 범위 · 중간선 · clean) | `kie_persons` + 라벨 json 27,301개 | 저장 `xk_items.parquet` | 27,301행 `equals` |
| 이미지 묶음 바이트 | 원본 PNG, 무작위 표본 | `crops/xk/{all,all3,page}.blob` | all(폭 1240) 30/30 · all3(폭 1860) 30/30 · page(폭 1240) 20/20 바이트 동일 |
| xkp 분할표 | `kie_persons` + 라벨 json 13,551개 | 저장 `xkp_items.parquet` | 아래 「xkp 질의 대상」 참고 |
| PaddleOCR-VL 학습 데이터 `vlm.pvl_line` · `pvl_config` | 저장 질의 | `ft/xk/{pvl,pvl3,pvlp,pvlp10}/{train,val}.jsonl` · `pvl16_full.yaml` | 4묶음 × (train · val) 줄 단위 동일(53,223 · 2,199 / 53,223 · 800 / 11,586 · 300 / 1,158 · 300), yaml dict 동일 |
| OvisOCR2 학습 데이터(base64) | xkp val 표본 300 | `ft/xk/ovisp/val.jsonl` 앞 20줄 | 동일 |
| 생성 출력 파싱 `fields.parse` | 저장 raw | 저장 pred | pvlkie_test 1,335/1,335 · pvlkiep_test 585/585 · ovkiep_test 585/585 · pvlkiepperm_test 442/442 · pvlkie2_tune 2,262/2,262 |
| 체크포인트 선정용 val 표 `vlm.val_em_frame` | `pvlkie_ep{1,2,3}_val` | `results/xk/g1/pvlft_ep{1,2,3}_val.parquet` | 3개 모두 `equals` (2,199행) |
| 영점 시험 표본 · 재채점 `zeroshot` | 저장 raw (20질의) | `zs_qwen20_test.parquet` | 표본 qid 동일, pred · ok · all_ok 재채점 `equals` |

### xkp 질의 대상(clean) — 원본 코드에 없던 정제 단계

저장된 `xkp_items.parquet`에는 원본 `xk_data.py page`가 만들지 않는 열(`n_rrn`, `n_sido`)과 clean 판정이 있고, 질의는 사람마다 train 6 · 그 외 3개였다
(원본 코드는 clean=True 고정 · 3/1개). 이 단계의 코드는 남아 있지 않아 규칙을 복원했다:

- `n_rrn` = 6자리 숫자 토큰 다음 토큰이 6~7자리 숫자인 쌍의 수(손글씨 주민등록번호 형태), `n_sido` = 시도 이름 토큰 수 → **13,551페이지 모두 일치**.
- clean = `n_rrn == 1` · `n_sido == 1` · 주소 칸 전체 = 파싱 주소(마지막 주소 토큰 오른쪽 같은 줄 200 px 안에 이어지는 토큰 없음) → 페이지 단위 **98.97 % 일치**
  (새 2,619 / 저장 2,658, 공통 2,569). 주소 칸 판정은 근사다.
- 저장 분할표에서 질의만 다시 만드는 `python -m formrec.kie.data requery --qset xkp`는 저장 질의와 **완전히 같다**(위 표). 공개 결과는 저장된 분할표 · 질의 기준이다.
- 그 밖 열은 같고, `values`는 clean 행에서 같다(clean이 아닌 422행은 원본 정제 단계가 주소를 칸 전체로 늘려 둔 것 — 질의에 쓰이지 않음).

### 동작이 바뀐 곳

- `order make`: 원본은 `hash(qid)`(프로세스마다 달라지는 문자열 해시)로 순서를 섞어 다시 만들 수 없었다 → md5 시드로 바꿨다(실행마다 같은 세트).
  저장된 `xkpperm_queries_test.parquet`는 원본이 만든 것 그대로 쓰고, `order score` 결과는 위 표처럼 같다.
- `zeroshot`: 접속 정보를 환경변수 `VLM_*` 또는 `.env`(FORMREC_ENV_FILE → 작업 공간 → 저장소 순)에서 읽는다. 값은 출력하지 않는다.
- GPU 단계(`vlm infer`, `verify score`)는 실행하지 않았다(코드 경로만 옮김). `verify decide`는 우도 파일이 아직 없어 합성 입력 단위 테스트만.

## 2. 일반 KIE (`formrec.kie_general`, paddle 환경 · 프로세스 8)

| 대상 | 입력 | 비교 | 결과 |
|---|---|---|---|
| 칸 기반 짝 `cells.pairs_cell` | `cache/xkg/ocr_*.parquet` 무작위 50 · 500페이지 | 원본 `xkg_cells.pairs_cell` | 50/50 · 500/500 완전 일치 (짝 4,751 · 빈 칸 4,434, 서식 이름 포함) |
| 빌드 전체 흐름 `build.main --limit 1500 --no_images` | 같은 OCR 1,500페이지 | 원본 `xkg_build.py` 사본(작업 공간만 임시 경로) | `xkg_items`(1,500 × 10) · `xkg_queries_{train,val,tune,test,test_tpl}` · `xkg_vocab.json` 모두 동일 |

(원본 작업 공간에는 `xkg_items`가 아직 없다 — 전체 빌드는 실행된 적 없음.)

## 3. 실험 운영 (`formrec.ops`, `scripts/`)

### 계획: 새 `plan.build()` vs 원본 `xplan.build()`

| | 원본 | 리포트 작업 제거 | 새 | 이름 · 순서 | deps | after | gpus | disk_gb | stall | io | 명령(치환 규칙) |
|---|---:|---:|---:|---|---|---|---|---|---|---|---|
| 실제 | 235 | 5 | 230 | 같음 | 230/230 | 230/230 | 230/230 | 230/230 | 230/230 | 230/230 | 230/230 |
| 스모크 | 136 | 5 | 131 | 같음 | 131/131 | 131/131 | 131/131 | 131/131 | 131/131 | 131/131 | 131/131 |

실제 계획: GPU 4장 30 · 1장 99 · CPU 101, 예상 쓰기 합 311 GB. 원본 `plan_smoke_xc`(스모크 중 xc만) = 새 `plan_smoke` + `FORMREC_SMOKE_ONLY=S_xc` (이름 · 의존 동일).
제거: `X_{xa,xh,xi,xa_scratch,xc}_report`(HTML 리포트 생성 — 공개 저장소에서 뺌). 그 작업을 after로 쓰던 곳은 스모크 정리 작업뿐(나머지 목록 그대로).

명령 대응(원본 조각 → 새 조각, 이 규칙만 적용하면 모든 명령 문자열이 같다):

| 원본 | 새 |
|---|---|
| `envs/{base,train,paddle,vllm,pf}/bin/python` | `"$FORMREC_PY_{BASE,TRAIN,PADDLE,VLLM,PF}"` (run_job.sh가 기본값을 채움) |
| `envs/train/bin/torchrun … work/x_train.py` / `work/x_pretrain.py` | `"$FORMREC_PY_TRAIN" -m torch.distributed.run … -m formrec.retrieval.train` / `.pretrain` |
| `work/x_eval.py` · `work/kie_summary.py` | `-m formrec.retrieval.evaluate` · `-m formrec.retrieval.summary` |
| `work/x_data.py` · `work/x_ft_data.py` | `-m formrec.data_prep.build` · `-m formrec.baselines.vlm.ft_data` |
| `work/pf_train.py` · `work/vlm_ckpt.py` · `work/g1_vlm.py` | `-m formrec.baselines.vlm.pf_train` · `.ckpt` · `.infer` |
| `work/ppocr_bn_select.py` · `work/g1_ppocr.py` · `work/x_base_eval.py` | `-m formrec.baselines.ppocr.bn_select` · `formrec.baselines.ppocr.infer` · `formrec.baselines.evaluate` |
| `work/xk_vlm.py` · `xk_eval.py` · `xk_oov.py` · `xk_order.py` · `xkg_ocr.py` | `-m formrec.kie.vlm` · `kie.evaluate` · `kie.oov` · `kie.order` · `kie_general.ocr` |
| `bash work/{ppocr_ft,ppocr_ft_post,ovis_train}.sh` · `work/x_smoke_clean.sh` | `bash "$FORMREC_REPO"/scripts/{…}.sh` · `scripts/smoke_clean.sh` |

### 스케줄러 · 래퍼 · 디스크 보호 스모크 (`tests/ops`, 임시 작업 공간 · 가짜 작업)

- `jobsched` + 실제 `scripts/run_job.sh`: CPU · GPU 1장 · GPU 4장 · IO · 실패 · 의존 실패 · 순서만 의존 7개 작업 →
  .done 5개(T_a · T_b · T_c · T_f · T_g), T_d FAIL rc=3, T_e QUEUE_SKIP, `QUEUE_DONE t (done 5, failed ['T_d'], skipped 1)`,
  4장 작업 `CUDA_VISIBLE_DEVICES=0,1,2,3`, 1장 작업 0~3 중 하나, IO 잠금 작업 `FORMREC_JOB` 전달, 코드 스냅샷 생성.
  `.retry` 표시 → 실패 · 건너뜀이 풀리고 고친 계획으로 다시 실행돼 T_d · T_e 완료. (GPU는 잠금 · 변수 배정만 — nvidia-smi 끔, 실제 GPU 사용 없음)
- 디스크 대기: 예측 초과 GPU 작업은 시작하지 않고 WAIT 이벤트를 한 번만 남김. `disk_ok` 기준(학습 ≥ 5 GB: 보류 · 95 %, 작은 쓰기: 95.5 %) 확인.
- `disk_guard` (가짜 runs · results · ft): `--dry`면 파일 변화 0, 할 일만 기록. 실제 모드에서 93.5 % → 에폭 내보내기 ep1 · Ovis 학습 파일만 삭제(최선 ep2 보존),
  95.2 % → 체크포인트 3개 중 loss 최선 1 · 지표 최선 2 보존 · 3 삭제(`pruned.json`) + 보류 ON, 96.5 % → GPU 작업만 종료 호출, 97.2 % → 전체 종료 호출,
  94.0 % → 보류 OFF. 작업 공간 밖 경로는 삭제를 거부. `running_jobs`는 살아 있는 X_/S_ 작업만.
- `kill_job.sh`: run_job.sh로 띄운 가짜 작업(자식 · 손자 sleep)을 트리째 종료, 래퍼가 FAIL 기록. `smoke_clean.sh`는 *_smoke만 지움.

## 4. 단위 테스트

`tests/kie`(11) · `tests/kie_general`(7, cv2 칸 검출 1개는 paddle 환경) · `tests/ops`(13). 실행:
`PYTHONPATH=src:<pytest 위치> <env>/bin/python -m pytest -q tests/kie tests/kie_general tests/ops`

## 5. 실행 기록

- `tests/legacy/test_kie_legacy.py` 11개 통과(23분 27초, base 환경 · OMP 8) · `test_ops_legacy.py` 3개 통과 · `test_kie_general_legacy.py` 1개 통과(paddle, 200페이지).
- `LEGACY_WORK` · `FORMREC_WORKSPACE`가 없으면 15개 모두 skip 확인.
