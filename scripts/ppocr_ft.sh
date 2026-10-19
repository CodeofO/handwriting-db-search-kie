#!/usr/bin/env bash
# PP-OCRv5 rec 파인튜닝 (공식 PaddleOCR tools/train.py, GPU 4장). 설정 = formrec.baselines.ppocr.ft_data가 만든 ft/{ds}/ppocr/{korean,en}_v5_ft.yml.
# 에폭마다 가중치 저장 → formrec.baselines.ppocr.bn_select가 에폭을 고른다. latest가 있으면 이어서 학습.
# 학습 전 입력 처리 1회 점검(formrec.common.input_check) — 실제 배치 그림, 학습/추론 칸 수 일치, 글자당 칸 수. 실패면 학습하지 않는다.
# 사용: env HW_DATASET=xa KIE_PPOCR_POOL=prop bash scripts/ppocr_ft.sh          (설정 파일 = PPOCR_CFG, 기본 korean_v5_ft.yml)
#       SMOKE=1 [SMOKE_EPOCHS=2] ... bash scripts/ppocr_ft.sh                    (4,096장, runs/{ds}/ppocrv5_ft_smoke)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${FORMREC_PY_PADDLE:-$ROOT/envs/paddle/bin/python}"
DS=${HW_DATASET:-xa}; export HW_DATASET=$DS
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || { echo "학습은 GPU 4장 필수"; exit 2; }
F=$ROOT/ft/$DS/ppocr; CFG="$F/${PPOCR_CFG:-korean_v5_ft.yml}"
RUN=$ROOT/runs/$DS/${PPOCR_RUN:-ppocrv5_ft}; O=()
[[ -n "${PPOCR_RUN:-}" ]] && O+=(Global.save_model_dir=$RUN Global.save_res_path=$RUN/predicts.txt)
if [[ -n "${SMOKE:-}" ]]; then
  RUN=$ROOT/runs/$DS/ppocrv5_ft_smoke
  head -n 4096 "$F/train_list.txt" > "$F/smoke_train_list.txt"; head -n 1024 "$F/tune_list.txt" > "$F/smoke_tune_list.txt"
  O+=(Global.epoch_num=${SMOKE_EPOCHS:-1} Global.save_epoch_step=1 "Global.eval_batch_step=[1000000000,1000000000]" "Train.dataset.label_file_list=[$F/smoke_train_list.txt]"
      "Eval.dataset.label_file_list=[$F/smoke_tune_list.txt]" Global.save_model_dir=$RUN Global.save_res_path=$RUN/predicts.txt
      Optimizer.lr.warmup_epoch=0)
fi
mkdir -p "$RUN"
if [[ ! -f $RUN/input_check.ok ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" -m formrec.common.input_check ppocr -c "$CFG" --name "${DS}_$(basename "$RUN")_pretrain" \
    || { echo "입력 점검 실패 — logs/checks/ 확인"; exit 3; }
  touch "$RUN/input_check.ok"
fi
[[ -f $RUN/latest.pdparams ]] && O+=(Global.checkpoints=$RUN/latest) && echo "resume from $RUN/latest"
exec "$PY" -m paddle.distributed.launch --gpus 0,1,2,3 --host 127.0.0.1 --log_dir "$ROOT/logs/pdist/${KIE_JOB:-ppocr_ft}" \
  "$REPO/src/formrec/baselines/ppocr/train.py" train -c "$CFG" ${O[@]+-o "${O[@]}"}
