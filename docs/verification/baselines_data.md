# 검증 — 비교군 · 데이터 준비 (영역 B)

원본 코드(이식 전)가 만든 산출물과 새 모듈의 출력을 비교했다. 모두 CPU(`CUDA_VISIBLE_DEVICES=""`), 원본 작업 공간은 읽기만 했다.
두 방식: (가) 읽기 전용 함수 호출(`FORMREC_WORKSPACE` = 원본 작업 공간), (나) 임시 작업 공간(필요한 입력만 심볼릭 링크)에서 CLI 실행 후 파일 비교.
비교는 `pandas.testing.assert_frame_equal(check_exact=True)` 또는 바이트 비교. 작업 공간 경로가 들어가는 열(`img`, `path`, 설정의 절대 경로)은 앞부분(작업 공간 루트)만 바꿔 비교했다.

## 1. 비교군

| 대상 | 입력 | 방식 | 결과 |
|---|---|---|---|
| 채점 `baselines.evaluate` | 저장된 생성 결과 `results/{xa,xh,xi,xc}/g1/{ppocrv5ft,pvlft,ovisft}_{tune,test}` | (가) score_levels vs `results/{ds}/scores/G1-*` | 132개 파일(xa 66 · xh 24 · xi 30 · xc 12), 642,309행 모두 같음, 파일 집합도 같음 |
| 채점 CLI | xa ppocrv5ft · pvlft | (나) `python -m formrec.baselines.evaluate --tag …` | 44개 점수 파일 **바이트까지 같음** |
| 생성 결과 정리 `baselines.normalize` (+ VLM score) | 저장된 g1 파일 62개의 raw | (가) 새 정리 함수 → pred, exp(lp_mean) → score | 1,352,645행 차이 0 |
| PP-OCR FT 데이터 `ppocr.ft_data` | 분할표 4종, 데이터셋별 인자(아래) | (가) 목록 · dict · 설정 텍스트 | `{train,val,tune,test}_list.txt` · `dict.txt` · `{korean,en}_v5_ft.yml`(YAML 텍스트 그대로) 4/4 데이터셋 같음 |
| VLM FT 데이터 `vlm.ft_data` | 분할표 4종, 계획 인자 | (가) jsonl 줄 · 설정 | PaddleOCR-VL `train.jsonl`(xa 108,828 · xh 308,247 · xi 5,825 · xc 40,886줄) · `val.jsonl` 전체 줄 같음, `pvl16_full.yaml` 텍스트 · 등록 파일 바이트 같음, Ovis `epochs.txt` · `swift_args.txt` 같음, Ovis `val.jsonl` 표본(xa 234 · xh 299 · xi 109 · xc 150줄, base64 PNG 포함) 같음 |
| VLM FT 데이터 CLI | xi (`vlm --vlm_epochs 10`, `ppocr --epochs 100 --max_len 100 --width 2560 --bs 8 --save_every 5`) | (나) | 원본에 남아 있는 13개 파일 모두 같음(Ovis train.jsonl은 원본이 학습 뒤 지워 비교 불가) |
| PP-OCR 내보내기 `ppocr/train.py export` | 원본 `runs/xa/ppocrv5_ft/bn_best`, `KIE_PPOCR_POOL=prop`, CPU | (나) 새 실행기로 export | `inference.pdiparams` · `inference.json` 바이트 같음, `inference.yml`(경로 제외) 같음 |
| PP-OCR 전처리 `bn_select.load_imgs` | val 목록 앞 64장, 데이터셋 폭(xh · xi는 가로 2배) | (가) 원본 `ppocr_bn_select.load_imgs`와 배열 비교 | 4/4 데이터셋 최대 절대 차이 0.0 |
| VLM 선정 `vlm.ckpt` val loss 해석 | 학습 로그 workerlog.0 / 남은 체크포인트 trainer_state.json | (가) vs 원본 `runs/{ds}/{pvl,ovis}_ft/best.json` | PaddleOCR-VL 4/4 같음(선정 에폭 xa 3 · xh 3 · xi 3 · xc 2 일치), Ovis 4/4 기록 순서 같음 |
| PaddleFormers 패치 `vlm.pf_train` | pf 환경, CPU | 패치 뒤 `_file_download("…/ft/xa/pvl/blob:src_train/5")` | `blob.read`와 같은 바이트 |

데이터셋별 PP-OCR FT 인자(원본 설정 파일에서 역산, 위 비교로 확인): xa 기본 · xc `--max_len 48 --width 1280` · xh `--epochs 3 --max_len 80 --width 640` ·
xi `--epochs 100 --max_len 100 --width 2560 --bs 8 --save_every 5`. VLM: xa 기본 · xh `--save_splits 3` · xi `--vlm_epochs 10` · xc `--vlm_epochs 3`.

GPU가 필요한 부분(PP-OCR 학습 · `ppocr.infer`의 TextRecognition, `bn_select`의 에폭 평가, `vlm.infer`의 vLLM 생성, PaddleFormers · ms-swift 학습, `ckpt.export`)은
가져오기 · 인자 파싱(계획의 실제 인자) · 전처리 함수만 확인했다.

## 2. 데이터 준비

| 대상 | 방식 | 결과 |
|---|---|---|
| `aihub_bbox` | (나) 전체 JSON 96,019개 | `meta/bbox.parquet` 4,379,632행 같음 |
| `names_v1` | (나) | 242,708행 같음 |
| `kie_persons` | (나) 전체 서식 페이지 55,695개 | 27,302행 같음(`img`는 루트 제외) |
| `kie_crops` | (나) `--limit 40`(세트마다 40명) | index 600행 같음, PNG 600개 바이트 같음(원본 blob의 같은 item) |
| `kie_cache.load` | (가) train · tune · test 각 100행 | `cache/kie/*_64x1280.npy`와 300/300 같음 |
| `kie_db` (+ `_2M`, `kie_sub_db`) | (나) 전체 | `kie_db_{L1…L6,F2,F3}` 8개 · `kie_db_F3_2M`(200만행) · `kie_db_F3_20k` · `kie_levels`(173,102행) 모두 같음 |
| `hana` | (나) | `hana_cols` 1,105,904행 · `hana_db` 703,802행 같음 |
| `build split` | (나) | `xa_items` 145,801 · `xh_items` 1,105,904 · `xi_items` 10,373행 같음(`xi.path`는 루트 제외) |
| `build.split_xc` | (가) | `xc_items` 54,602행 같음(목록 열 포함) |
| `build links` | (나) | 링크 6개 대상 같음 |
| `build iam_blob` · `hana_blob` · `cache --ds xi/xh` | (나) 분할표 앞 100행 | blob 바이트 · 오프셋 · 캐시 배열 앞 100행 같음 |
| `iam_png` · `hana_png` · `xc_compose` | (가) 무작위 100행씩 | 원본 blob과 100/100 · 100/100 · 100/100 바이트 같음 |
| 캐시 `load` | (가) xh · xi · xc 무작위 100행씩 | 원본 npy와 100/100 같음 |
| `search_db` | (나) | `xh_db` 703,802 · `xi_db` 204,147행 같음 |
| `vocab` | (나) train 환경 | `vocab.json` 텍스트 같음(138 단위) |

전체 crop · 캐시 재생성(crops/xh 31GB, cache/xh 30GB 등)은 디스크 사정으로 하지 않았고 표본으로 대신했다.

## 3. 테스트

- 단위: `tests/baselines/test_baselines_unit.py`(16), `tests/data_prep/test_data_prep_unit.py`(9) — 합성 입력, 작업 공간 불필요.
- 원본 대조: `tests/legacy/test_baselines_legacy.py`(20), `tests/legacy/test_data_prep_legacy.py`(6) — `LEGACY_WORK` · `FORMREC_WORKSPACE`가 없으면 건너뜀.
  위 표의 (가) 항목을 줄인 형태(분할표 · DB · 어휘 전체, 원천 파싱 · 이미지 묶음은 표본). 원본 작업 공간 기준 26개 통과(약 110초).

```
PYTHONPATH=src:<pytest 경로> <base 환경 python> -m pytest -q tests/baselines tests/data_prep
LEGACY_WORK=<원본 work> FORMREC_WORKSPACE=<원본 작업 공간> PYTHONPATH=src:<pytest 경로> <base 환경 python> -m pytest -q tests/legacy/test_baselines_legacy.py tests/legacy/test_data_prep_legacy.py
```
