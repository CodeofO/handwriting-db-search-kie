#!/usr/bin/env bash
# OvisOCR2 전체 파인튜닝 (ms-swift 4.5.3, GPU 4장). 저자 공식 FT 코드가 없어 ms-swift의 ovis_ocr2 등록을 쓴다.
# 레시피 = ms-swift 전체 FT 기본값 + Qwen3.5 예제: lr 1e-5 cosine, warmup 0.05, wd 0.1, β2 0.95, clip 1.0,
#   전역 배치 128 = 4 × 32, bf16, gradient checkpointing, ZeRO-2, sdpa, 공식 페이지 프롬프트 + enable_thinking false.
# 저장 · val loss 주기와 에폭 수는 호출 인자로 덮어쓴다(ft/{ds}/ovis/swift_args.txt, 예: --val_dataset … --eval_strategy epoch).
# --strict true: 기본값이면 인코딩 실패 샘플을 조용히 다른 샘플로 바꾼다 → 에러로 드러나게.
# 해상도: 추론(vLLM min/max pixels 448²/2880²)과 같게 IMAGE_MIN/MAX_TOKEN_NUM 196/8100.
# 사용: env HW_DATASET=xa bash scripts/ovis_train.sh --val_dataset ft/xa/ovis/val.jsonl [--max_steps 20 (스모크)]
#       학습 파일 = OVIS_TRAIN (기본 ft/{ds}/ovis/train.jsonl), 출력 = OUT (기본 runs/{ds}/ovis_ft)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
PY="${FORMREC_PY_SWIFT:-$ROOT/envs/swift/bin/python}"
DS=${HW_DATASET:-xa}
OUT=${OUT:-$ROOT/runs/$DS/ovis_ft}
for x in "$@"; do [[ "$x" == --max_steps* ]] && OUT=$ROOT/runs/$DS/ovis_ft_smoke; done
mkdir -p "$OUT"
export IMAGE_MIN_TOKEN_NUM=196 IMAGE_MAX_TOKEN_NUM=8100
export HF_DATASETS_CACHE=$ROOT/cache/hfds/$DS MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}   # 전처리 캐시 격리 → 학습 뒤 정리 가능
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=4 MASTER_PORT=${MASTER_PORT:-29531} TOKENIZERS_PARALLELISM=false
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || { echo "학습은 GPU 4장 필수 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})"; exit 2; }
export PATH="$(dirname "$PY"):$PATH"
swift sft \
  --model "$ROOT/models/OvisOCR2" --model_type ovis_ocr2 --template ovis_ocr2 \
  --tuner_type full --freeze_llm false --freeze_vit false --freeze_aligner false \
  --dataset "${OVIS_TRAIN:-$ROOT/ft/$DS/ovis/train.jsonl}" --dataset_num_proc 4 \
  --enable_thinking false --add_non_thinking_prefix true --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 --bf16 true --attn_impl sdpa \
  --num_train_epochs 3 --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --weight_decay 0.1 --adam_beta2 0.95 --max_grad_norm 1.0 \
  --gradient_checkpointing true --deepspeed zero2 \
  --max_length 1024 --truncation_strategy delete \
  --strict true \
  --eval_strategy no --split_dataset_ratio 0 \
  --save_strategy epoch --save_only_model ${SAVE_ONLY_MODEL:-true} \
  --logging_steps 10 --dataloader_num_workers 4 \
  --seed 23 --data_seed 23 --add_version false \
  --output_dir "$OUT" --report_to none "$@"
