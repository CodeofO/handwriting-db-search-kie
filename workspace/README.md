# workspace/

코드와 분리된 작업 공간입니다. 데이터 · 캐시 · 가중치 · 결과는 여기에 생기며 저장소에는 넣지 않습니다(`.gitignore`).
다른 위치를 쓰려면 `FORMREC_WORKSPACE=/path/to/workspace` 를 지정합니다.

```
workspace/
  data/      원천 데이터 (AI Hub 053, HANA, IAM — 직접 내려받기, docs/data.md)
  meta/      분할표 · 검색 DB · 질의 (meta/x/{ds}_items.parquet 등)
  crops/     crop PNG 묶음 (blob)
  cache/     학습용 이미지 배열 (npy)
  ft/        비교군 프레임워크 학습 데이터
  runs/      학습 산출물 (체크포인트)
  results/   평가 결과 (점수 · 요약)
  envs/      Python 환경 (scripts/setup_envs.sh)
  third_party/ PaddleOCR · PaddleFormers (scripts/setup_third_party.sh)
  logs/      실행 기록
```
