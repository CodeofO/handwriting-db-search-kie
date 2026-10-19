# 검증 — 검색형 제안 모델 (영역 A, `formrec.retrieval`)

원본 연구 코드(`x_common` · `x_pretrain` · `x_train` · `x_eval` · `kie_summary` + `kie_model3` · `kie_model` · `g4_model` · `kie_train` · `kie_train3` · `kie_eval3` · `g4_train`)를
`formrec.retrieval.{tokenizer, model, data, augment, distributed, pretrain, train, evaluate, summary}`로 옮긴 뒤 같은 입력에서 같은 출력이 나오는지 대조했다.
모두 CPU(`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=8`, `MKL_DYNAMIC=FALSE`), fp32(autocast는 CUDA에서만 켜짐). 원본 작업 공간은 읽기만 했다.

두 방식
- (가) 함수 단위: 원본 모듈을 `LEGACY_WORK`에서 불러(바이트코드 쓰지 않음) 같은 입력으로 호출. `x_train` · `x_pretrain`의 `main()` 안 닫힘 함수
  (`build` · `batch_loss` · `val_loss` · 배치 생산자 본문)는 ast로 꺼내 같은 전역(원본 모듈 · 원본 모델)으로 컴파일해 불렀다.
- (나) CLI 단위: 임시 작업 공간 두 개(입력 `meta` · `cache/*` · 체크포인트만 심볼릭 링크)를 만들고, 한쪽은 원본 스크립트 사본(`cfg.ROOT` = 임시 루트),
  다른 쪽은 `FORMREC_WORKSPACE`로 새 모듈을 같은 인자로 실행한 뒤 산출물을 비교. `sel.json`처럼 쓰는 파일은 링크하지 않았다.

재현: `LEGACY_WORK=<원본 work> FORMREC_WORKSPACE=<원본과 같은 입력이 있는 작업 공간> pytest tests/legacy/test_retrieval_*.py` (약 1분, 10개),
단위 테스트 `pytest tests/retrieval` (12개, 작업 공간 불필요).

## 1. 모델 추론 경로 (가)

체크포인트 `runs/x/{xa,xh,xi,xc}_prop/final.pt`를 원본 `RecordEncoder3`와 새 `RecordEncoder3`에 **둘 다 strict 로드**(키 312개, 누락 · 남는 키 없음).
입력: 캐시(`cache/kie/test_64x1280.npy` 등, mmap)의 test 레코드 4개 — xa는 F3 2개 · F2 1개 · L1 1개(이미지 3 · 3 · 2 · 1장), xh · xi · xc는 단계 앞 4개.
후보: 레코드마다 4개(정답 · 마지막 글자 바꿈 · 다른 레코드 · 글자 덧붙임) = 16개.

| 비교 항목 | xa_prop | xh_prop | xi_prop | xc_prop |
|---|---|---|---|---|
| 평가 정규화 `augment(train=False)` | 0 | 0 | 0 | 0 |
| `image_tokens` (B, 3, 320, 384) | 0 | 0 | 0 | 0 |
| `slots_image` · `text_tokens` · `slots_text` | 0 | 0 | 0 | 0 |
| 토큰화 `batch` 4줄 · `field_units` · 필드 글자 수(`flens`) | 같음 | 같음 | 같음 | 같음 |
| `ink_extent` | 0 | 0 | 0 | 0 |
| `pli` (점수 · 필드별 · 마스크, per_slot 유무) | 0 | 0 | 0 | 0 |
| `cand_ctc` (점수 · 필드별 · 마스크) | 0 | 0 | 0 | 0 |
| `final_score` + mix_ctc · `verify_feats`(14) · `verify` | 0 | 0 | 0 | 0 |
| `ctc_loss` · `read_feats` | 0 | 0 | 0 | 0 |

값은 최대 절대 차이(max|diff|). 초기 실행 몇 번에서 `pli` 첫 호출에 2.4e-6 (fp32 반올림 수준) 차이가 났는데, 시스템 부하에 따라 MKL 스레드 수가 바뀌는
`MKL_DYNAMIC` 때문이었다(같은 실행 안 이후 호출 · 다른 실행에서는 0). `MKL_DYNAMIC=FALSE`로 고정한 뒤 4개 체크포인트 · 반복 14회 모두 0.
원본 테스트(`test_retrieval_model.py`)는 이 때문에 허용 오차 1e-5로 둔다.

## 2. 학습 (가) + (나)

### 2.1 함수 단위 (가)

| 항목 | 입력 | 결과 |
|---|---|---|
| 레코드 집합 `XData` (pids · by) | xa · xi · xc train | 같음 |
| 레코드 표집 `sample` | 시드 7, 64 × 50배치 (xa 복수 필드 · xi/xc 단일) | 50배치 모두 같음, 이후 난수 상태도 같음 |
| 글자 풀 `char_pools` · 토큰 사전 `token_vocab` | xa · xi · xc train | 같음 |
| 합성 음성 `synth_negatives` (원본 build 안 루프와 비교) | 시드 11, 3,200레코드 × 3음성, 데이터셋 3개 | 같음 |
| `build_batch` (정답 · 음성 문자열 · 증강 이미지 · 마스크) | xa_prop · xi_prop, bs 8, train/val, 같은 numpy · torch 시드 | 문자열 같음, 이미지 max|diff| 0, 마스크 같음 |
| `batch_loss` (전체 · l1 · pli · ctc · ver · vacc · div · scale) | 위 배치, train(dropout · 슬롯 드롭 · BN 학습 모드) / val | 모두 0 (xa train 0.163233, val 0.072792 · xi train 7.031502, val 0.49296) |
| 기울기 (268개 텐서) | 위 train 배치 backward | xa 0, xi 1.5e-8 |
| 사전학습 배치 표집 (원본 producer 본문) | 시드 42, 16 × 5배치, xa · xh · xi | 같음 |
| 사전학습 `val_loss` (데이터셋별 loss · EM · n) | pretrain_ctc/final.pt, val_cap 48 | 0 (xa 0.02081 · xh 0.06799 · xi 0.25864) |
| 사전학습 증강(학습, keep_ink) · CTC loss · 기울기 | bs 16 | 0 · 0 · 0 |

xi 기울기 1.5e-8은 CPU 역전파의 스레드 누적 순서 차이(손실 값은 비트 단위로 같음).

### 2.2 CLI 단위 (나) — FT 3스텝

`--data xa --init runs/x/pretrain_ctc/final.pt --cpu --steps 3 --spe 1 --bs 4 --val_recs 16` (사전학습 가중치 로드 · mix 초기화 · 옵티마이저 · 스케줄러 · 입력 점검 · 에폭별 val · 저장 전부 포함)

| 비교 | 원본 vs 새 코드 | 원본 vs 원본(같은 명령 2번) |
|---|---|---|
| 화면 로그(손실 4자리) | 3스텝 · 3에폭 모두 같은 숫자 | — |
| `log.jsonl` 수치(시간 제외) | 4.8e-7 | 9.5e-7 |
| `epochs/ep1·2·3.pt` 가중치 312개 | 1.4e-7 · 6.8e-7 · 2.6e-6 | 2.6e-7 · 6.9e-7 · 3.8e-6 |
| `best.json` 선정 에폭 | 3 = 3 | 3 = 3 |
| `args.json` | 원본에만 `vfeat`(14, 제거한 인자) | — |

원본끼리 다시 돌려도 같은 크기의 차이가 나므로(CPU 비결정성) 새 코드는 원본과 구별되지 않는다. 사전학습 CLI는 배치 생산 스레드 2개가 난수 생성기를
나눠 써서 원본끼리도 재현되지 않으므로 함수 단위(2.1)로만 대조했다.

## 3. 평가 (나) + (가)

### 3.1 CLI 단위 — `x_eval` vs `formrec.retrieval.evaluate`

xi_prop: `--mode select --smoke 40 --cap 200` → `--mode eval --smoke 40` → `--mode epochs --smoke 40 --cap 200`(에폭 2개만 링크).
xa_prop: `--mode select --smoke 40 --cap 200` → `--mode eval --smoke 40` (서브 단계 F3@20k(DB 보정 `fix20k` 포함) · F3@2M · F3#d0~2 포함).
스모크 DB = 단계 DB 앞 400개 + 질의 정답이라 정확도가 대부분 1.0이다 — 격자 선택보다 점수 · 확신도 열이 같은지가 이 대조의 요점.

| 산출물 | xi | xa |
|---|---|---|
| `sel.json` K · λ · 격자 44칸 | (8, 0.0) = (8, 0.0), 격자 차이 0 | (8, 0.0) = (8, 0.0), 격자 차이 0 |
| `results/{ds}/scores/*.parquet` (최종 + `_t1`) | 20개 파일 800행, 모든 열(pred · correct · conf · margin_conf · s1 · fused_s1 · p1 · g1 · open_* …) 차이 0 | 52개 파일(기본 8단계 + 서브 5단계 × tune · test × final · t1) 2,080행, 모든 열 차이 0 |
| `latency/P-*.json` 키 · recall_at_K | 같음 | 같음 |
| `epochs_metric.csv` | 같음 | — |

### 3.2 함수 단위 — `run_level`

원본 `kie_eval3.run_level`(DB 조회를 원본 x_eval처럼 바꿔 끼움) vs 새 `run_level`, xi_prop tune 단계 5개 × 질의 8개, K 8 · λ 0.5 · keep_raw:
결과 프레임 두 개(final · t1)의 모든 열 같음(실수 열 허용 1e-5 안), λ 선택 원점수(base · s3 · top · gt) 같음, 질의(`XQueries.get`) pid · 정답 · 이미지 행 같음.
(`tests/legacy/test_retrieval_eval.py::test_run_level`)

## 4. 요약 표 — `kie_summary` vs `formrec.retrieval.summary`

입력: 원본 `results/{xa,xh,xi,xc}/scores/*.parquet`(제안 모델 · 비교군 전부, 읽기만). 원본 스크립트 사본을 임시 루트(점수 폴더만 링크)에서 실행한 `summary.csv`와 새 모듈 출력을 비교.

| 데이터셋 | 행 × 열 | 방법 수 | 원본 재실행 vs 새 | 원본 작업 공간의 기존 summary.csv vs 새 |
|---|---|---|---|---|
| xa | 92 × 41 | 7 | 바이트 같음 | 바이트 같음 |
| xh | 25 × 41 | 5 | 바이트 같음 | 바이트 같음 |
| xi | 30 × 41 | 5 | 바이트 같음 | 바이트 같음 |
| xc | 15 × 41 | 5 | 바이트 같음 | 바이트 같음 |

채점 · 확신도 지표(통합 임계값 · 확신도 선택 · DB 밖 오수락 · AUC)는 이 표의 모든 열에 들어 있으므로 함께 확인된다.

## 5. 확인하지 않은 것

- GPU 경로(bf16 autocast · channels_last · nccl all_gather 다중 GPU InfoNCE · fused AdamW). 코드는 원본 그대로 옮겼고 CPU 경로만 실행했다.
- 4 GPU 학습 전체 재현 · 전체 세트 평가(시간 · GPU 금지). 저장된 점수(GPU bf16)와 CPU fp32 재계산은 원래 비트 단위로 같지 않다.
