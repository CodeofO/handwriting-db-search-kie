#!/usr/bin/env bash
# 가상환경 구축 (uv, Python 3.12, managed python) → {작업 공간}/envs/<이름>. 인자: base | train | paddle | vllm | pf | swift
# 버전은 x86_64 · RTX A6000(sm_86) · 드라이버 CUDA 13.0에서 동작 확인한 조합. PaddleOCR 요구 패키지는 third_party 사본(scripts/setup_third_party.sh)에서.
#   base  : 데이터 처리·평가 (pandas, pyarrow, pillow, rapidfuzz, pyyaml)
#   train : torch cu128 + transformers 4.57.6 (제안 모델 학습 · 평가, VLM 체크포인트 변환)
#   paddle: paddlepaddle-gpu 3.2.0(cu126) + paddleocr 3.7 (PP-OCRv5 공식 추론·학습)
#   vllm  : vLLM 0.22.1 (VLM 파인튜닝 모델 추론)
#   pf    : PaddleFormers 1.2.0 [paddlefleet] (PaddleOCR-VL 공식 파인튜닝)
#   swift : ms-swift 4.5.3 + transformers 5.16.1 + torch 2.10 cu130 (OvisOCR2 파인튜닝; 저자 공식 FT 코드가 없고
#           ms-swift가 ATH-MaaS/OvisOCR2를 model_type/template ovis_ocr2로 등록). fla 0.5.2 + causal-conv1d 1.7.0(GDN 커널)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
E="$ROOT/envs"; mkdir -p "$E"
export UV_PYTHON_PREFERENCE=only-managed UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_DOWNLOADS=4
PY=3.12
mk() { [[ -x "$E/$1/bin/python" ]] || uv venv -q --python $PY "$E/$1"; }
case "$1" in
  base)
    mk base
    uv pip install --python "$E/base/bin/python" numpy pandas pyarrow pillow rapidfuzz tqdm matplotlib pyyaml
    ;;
  train)
    mk train
    uv pip install --python "$E/train/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$E/train/bin/python" numpy pandas pyarrow pillow rapidfuzz tqdm nvidia-ml-py \
      "transformers==4.57.6" einops safetensors
    ;;
  paddle)
    mk paddle
    uv pip install --python "$E/paddle/bin/python" paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    uv pip install --python "$E/paddle/bin/python" paddleocr==3.7.0 pandas pyarrow pyyaml opencv-python
    [[ -f "$ROOT/third_party/PaddleOCR/requirements.txt" ]] && \
      uv pip install --python "$E/paddle/bin/python" -r "$ROOT/third_party/PaddleOCR/requirements.txt"
    ;;
  vllm)
    mk vllm
    uv pip install --python "$E/vllm/bin/python" "vllm==0.22.1" pillow pandas pyarrow huggingface_hub
    ;;
  pf)
    mk pf
    uv pip install --python "$E/pf/bin/python" pip "paddleformers[paddlefleet]==1.2.0" \
      --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ \
      --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --index-strategy unsafe-best-match --prerelease=allow
    PV=$("$E/pf/bin/python" -c "import importlib.metadata as m; print(m.version('paddlepaddle-gpu'))")
    echo "paddlepaddle-gpu==$PV" > "$E/pf/constraints.txt"
    uv pip install --python "$E/pf/bin/python" --no-deps "paddlefleet_ops==0.3.0" \
      --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --index-strategy unsafe-best-match
    uv pip install --python "$E/pf/bin/python" "paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5" triton "nvidia-cutlass-dsl[cu13]==4.4.1" \
      filelock "apache-tvm-ffi>=0.1.3,<0.1.12" -c "$E/pf/constraints.txt" \
      --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ \
      --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --index-strategy unsafe-best-match --prerelease=allow
    ;;
  swift)
    mk swift
    P="$E/swift/bin/python"
    uv pip install --python "$P" "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python "$P" "ms-swift==4.5.3" "transformers==5.16.1" "trl==0.29.1" "peft==0.20.0" "datasets>=3.0,<4.8.5" \
      accelerate "deepspeed==0.19.7" "qwen_vl_utils==0.0.14" decord tensorboard pandas pyarrow
    uv pip install --python "$P" "flash-linear-attention==0.5.2" "fla-core==0.5.2"
    uv pip install --python "$P" --no-deps \
      "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    "$P" -c "import torch, transformers, swift; assert torch.__version__.startswith('2.10.0'), torch.__version__; from transformers import Qwen3_5ForConditionalGeneration; print('torch', torch.__version__, 'tf', transformers.__version__, 'swift', swift.__version__)"
    ;;
  *) echo "unknown env $1"; exit 2 ;;
esac
echo "ENV_OK $1"
