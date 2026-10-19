#!/usr/bin/env bash
# 스모크 산출물 정리 (이름에 smoke가 들어간 실행 · 결과 · 데이터 조각). 실제 실험 산출물은 건드리지 않는다. 스모크 계획의 마지막 작업(S_cleanup).
# FORMREC_SMOKE_DS="xc"처럼 주면 그 데이터셋만(기본: x 사전학습 + xa · xh · xi).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${FORMREC_WORKSPACE:-$REPO/workspace}" || exit 1
DS="${FORMREC_SMOKE_DS:-}"
if [[ -z "$DS" ]]; then
  rm -rf runs/x/*_smoke runs/x*/*_smoke runs/x[ahi]/*_smoke runs/xa/ppocrv5_ft_smoke runs/xh/ppocrv5_ft_smoke runs/xi/ppocrv5_ft_smoke
  rm -f results/x[ahi]/scores/*smoke* results/x[ahi]/g1/*smoke* results/x[ahi]/latency/*smoke* ft/x[ahi]/*/*_smoke.jsonl ft/x[ahi]/ppocr/smoke_*_list.txt
  rm -f results/x[ahi]/summary.csv
else
  for d in $DS; do
    rm -rf runs/x/${d}_prop_smoke runs/$d/*_smoke
    rm -f results/$d/scores/*smoke* results/$d/g1/*smoke* results/$d/latency/*smoke* ft/$d/*/*_smoke.jsonl ft/$d/ppocr/smoke_*_list.txt results/$d/summary.csv
  done
fi
echo "DONE smoke_clean"
