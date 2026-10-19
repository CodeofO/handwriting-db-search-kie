# 데이터 (초안)

이 저장소는 **데이터를 포함하지 않으며 재배포하지 않습니다.** 세 원천 데이터는 각 배포처의 이용 약관에 따라 직접 신청 · 내려받아야 하고,
이 저장소가 만드는 산출물(분할표 · crop 묶음 · 캐시 · 검색 DB · 학습 데이터)도 원천 데이터의 파생물이므로 같은 약관을 따릅니다.
분할표(`meta/x/*_items.parquet`)에는 정답 문자열과 원본 파일 이름이 들어 있어 원천 데이터 없이 공개할 수 없습니다.

## 1. 원천 데이터와 배치

모든 경로는 작업 공간(`FORMREC_WORKSPACE`, 기본 `workspace/`) 기준입니다.

| 데이터 | 배포처 · 받는 것 | 작업 공간 배치 |
|---|---|---|
| AI Hub 053 대용량 손글씨 OCR (한국어) | AI Hub(aihub.or.kr) 신청 · 승인 후 내려받아 압축 해제. 서식(O.Form) 페이지 PNG + bbox 라벨 JSON | `data/aihub053/01.데이터/{1.Training,2.Validation}/` |
| HANA (덴마크 손글씨 이름) | HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition (Dahl et al.) 공개본의 `archive.zip` (약 47GB, 압축을 풀지 않는다) | `data/hana/archive.zip` |
| IAM Handwriting Database 3.0 (영어) | 베른 대학교 FKI 연구실(등록 필요): `ascii.tgz`(lines.txt · forms.txt), `lines.tgz`(줄 이미지). RWTH Aachen 분할 목록(`train/validation/test.uttlist`) | `data/iam/raw/` |

```
data/
  aihub053/01.데이터/
    1.Training/  원천데이터/HW-OCR/4.Validation/{P.Paper,T.Tablet}/{O.Form,R.Free}/{image_id}.png
                 라벨링데이터/라벨/HW-OCR/4.Validation/{P.Paper,T.Tablet}/{O.Form,R.Free}/{image_id}.json
    2.Validation/ (같은 구조 — AI Hub 원본 구조라 Training 쪽 폴더 이름도 4.Validation)
  hana/archive.zip            HANA/labels/{train,test}/HANA.csv, HANA/minipics/HANA/{image_id}
  iam/raw/
    lines.txt  forms.txt      (ascii.tgz)
    splits/{train,validation,test}.uttlist   (RWTH Aachen 분할: 서식 ID 목록)
    a01/a01-000u/a01-000u-00.png …          (lines.tgz를 그대로 풀기)
```

PP-OCRv5 · PaddleOCR-VL-1.6 · OvisOCR2 가중치는 `models/`에 둡니다
(`models/{korean,en}_PP-OCRv5_mobile_rec/{infer, *_pretrained.pdparams}`, `models/PaddleOCR-VL-1.6`, `models/OvisOCR2`).
PaddleOCR · PaddleFormers 공식 코드는 `bash scripts/setup_third_party.sh`, Python 환경은 `bash scripts/setup_envs.sh <이름>`.

## 2. 만드는 순서

명령은 작업 공간에서 `PYTHONPATH=<저장소>/src`로 실행합니다. 이미지 묶음 · 캐시는 IO가 커서 한 번에 하나씩, 프로세스 8개 이하로 돕니다.

| 단계 | 명령 (`python -m formrec.data_prep.…`) | 입력 → 출력 |
|---|---|---|
| A1 | `aihub_bbox` | 라벨 JSON 9.6만 개 → `meta/bbox.parquet` (bbox 438만) |
| A2 | `names_v1` | bbox의 PR 페이지 이름 구간 → `meta/names_v1.parquet` (실제 한국어 이름 24만, 고유 22.8만 — DB 방해 항목용) |
| A3 | `kie_persons` | 서식(PO · TO) 페이지 5.6만 → 사람 블록 `meta/kie_persons.parquet` (2.7만 명) |
| A4 | `kie_crops` | 페이지 PNG → 필드 crop `crops/kie/{train,tune,test}.blob` + `cache/kie/{…}_index.parquet` |
| A5 | `kie_cache --H 64 --W 1280` | blob → `cache/kie/{…}_64x1280.npy` (제안 모델 입력) |
| A6 | `kie_db` · `kie_db --levels F3 --N 2000000 --suffix _2M --no_levels_file` · `kie_sub_db` | → `meta/kie_db_{L1…L6,F2,F3}.parquet`(단계마다 20만, 절반은 토큰 1개만 다른 이웃), `kie_db_F3_2M`, `kie_db_F3_20k`, 단계 질의 `meta/kie_levels.parquet` |
| H1 | `hana` | zip의 CSV 2개 → `meta/hana_cols.parquet`(성 · 이름 · 중간이름, 110.6만), `meta/hana_db.parquet` |
| X1 | `build split` | → `meta/x/{xa,xh,xi}_items.parquet` (최종 분할표) |
| X2 | `build links` | `crops/xa/src_{train,tune,test}.blob` → `crops/kie/…` 심볼릭 링크 |
| X3 | `build iam_blob` · `build hana_blob` | → `crops/xi/all.blob`(원본 해상도) · `crops/xh/all.blob`(높이 96, zip 순차 1회) |
| X4 | `build cache --ds xi` · `build cache --ds xh` | → `cache/xi/all_64x1280.npy` · `cache/xh/all_64x448.npy` |
| X5 | `build xc_blob` · `build cache --ds xc` | xa F2 · F3 칼럼을 가로로 이은 한 줄 → `meta/x/xc_items.parquet`, `crops/xc/all.blob`, `cache/xc/all_64x1280.npy` |
| X6 | `search_db` | → `meta/x/xh_db.parquet`(고유 이름 줄 70만), `meta/x/xi_db.parquet`(실제 줄 + 단어 1개 이웃, 20만) |
| X7 | `vocab` (train 환경) | xa · xh · xi의 test 밖 정답 → `meta/x/vocab.json` (단위 138개) |

비교군 학습 데이터(`ft/{ds}/ppocr`, `ft/{ds}/pvl`, `ft/{ds}/ovis`)는 분할표에서 `formrec.baselines.ppocr.ft_data` · `formrec.baselines.vlm.ft_data`가 만듭니다.

## 3. 분할

| 데이터셋 | 묶음 단위 | train / val / tune / test (이미지 수) | 규칙 |
|---|---|---|---|
| xa (AI Hub 서식 필드) | 페이지 | 108,828 / 13,462 / 13,744 / 9,767 (사람 20,443 / 2,527 / 2,580 / 1,751) | 공식 Validation 폴더 → test, Training 페이지 → 0.8 / 0.1 / 0.1 |
| xh (HANA 이름 줄) | 이미지(필기자 ID 없음) | 839,588 / 105,282 / 105,212 / 55,822 | 공식 test 고정, 공식 train → 0.8 / 0.1 / 0.1 |
| xi (IAM 줄) | 필기자(forms.txt) | 5,825 / 976 / 657 / 2,915 | Aachen validation · test 그대로, Aachen train → train / tune(필기자를 해시 순으로 채워 train 줄의 10%) |
| xc (xa 칼럼 한 줄) | xa와 같은 사람 · 같은 세트 | 40,886 / 5,054 / 5,160 / 3,502 | 단계 C2 = F2, C3 = F3와 같은 내용 |

해시 = `md5("x1:" + 묶음 키)` (`formrec.common.splits`). 역할: val = 에폭 선정(loss), tune = 하이퍼파라미터 · 수락 임계값, test = 마지막 1회.
`kie_persons.subset`(train / tune / test)은 이전 단계 분할로, crop 묶음 파일 이름(`crops/kie/{subset}.blob`)과 `kie_db_F3_20k`의 필수 정답 집합에만 남아 있습니다.
