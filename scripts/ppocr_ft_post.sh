#!/usr/bin/env bash
# PP-OCRv5 파인튜닝 후처리 (CPU): 공식 export_model로 선정 가중치 → 추론 모델(runs/{ds}/{run}/infer), 재개용 파일 삭제.
# 선정 가중치 = bn_select 결과 bn_best(있으면), 없으면 공식 best_accuracy. 에폭 가중치(iter_epoch_*.pdparams)는 선정 근거로 보관.
# 사용: env HW_DATASET=xa KIE_PPOCR_POOL=prop bash scripts/ppocr_ft_post.sh [run 이름=ppocrv5_ft]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${FORMREC_PY_PADDLE:-$ROOT/envs/paddle/bin/python}"
DS=${HW_DATASET:-xa}
RUN=$ROOT/runs/$DS/${1:-ppocrv5_ft}
BEST=$RUN/best_accuracy; [[ -f $RUN/bn_best.pdparams ]] && BEST=$RUN/bn_best
[[ -f $BEST.pdparams ]] || { echo "체크포인트 없음: $BEST"; exit 1; }
CUDA_VISIBLE_DEVICES="" HW_DATASET=$DS "$PY" "$REPO/src/formrec/baselines/ppocr/train.py" export \
  -c "$ROOT/ft/$DS/ppocr/${PPOCR_CFG:-korean_v5_ft.yml}" -o Global.use_gpu=False Global.pretrained_model="$BEST" \
  Global.save_inference_dir="$RUN/infer"
ls "$RUN/infer"
[[ -f $RUN/infer/inference.pdiparams && -f $RUN/infer/inference.yml ]] || { echo "export 실패"; exit 1; }
{ echo "export: $BEST"; [[ -f $RUN/bn_select.csv ]] && cat "$RUN/bn_select.csv"; grep -h "best metric" "$RUN"/train.log 2>/dev/null | tail -1; } > "$RUN/best.txt" || true
rm -f "$RUN"/latest.* "$RUN"/iter_epoch_*.pdopt   # 용량 확보
echo "DONE ppocr_ft_post $(cat "$RUN/best.txt" 2>/dev/null)"
