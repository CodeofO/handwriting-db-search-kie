#!/usr/bin/env bash
# 공식 PaddleOCR · PaddleFormers 저장소를 실험에 쓴 커밋으로 받는다 → {작업 공간}/third_party/{PaddleOCR,PaddleFormers}.
# 두 사본 모두 로컬 수정 없음(공식 코드 그대로). 바꾸는 것은 실행기(formrec.baselines.ppocr.train · vlm.pf_train)의 런타임 패치뿐.
#   PaddleOCR     : tools/train.py · export_model.py, configs/rec/PP-OCRv5 (PP-OCRv5 rec 학습 · 내보내기)
#   PaddleFormers : examples/best_practices/PaddleOCR-VL-1.5 (PaddleOCR-VL 파인튜닝 기준 설정 · 등록 파일의 출처, 참고용)
#                   학습 자체는 pf 환경의 pip 패키지 paddleformers==1.2.0을 쓴다(scripts/setup_envs.sh pf).
# 사용: bash scripts/setup_third_party.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="${FORMREC_WORKSPACE:-$REPO/workspace}"
TP=$ROOT/third_party; mkdir -p "$TP"
fetch() {   # 이름 · 저장소 · 커밋
  local d=$TP/$1
  if [[ -d $d/.git ]] && [[ "$(git -C "$d" rev-parse HEAD)" == "$3" ]]; then echo "$1: 이미 $3"; return; fi
  rm -rf "$d"; git init -q "$d"; git -C "$d" remote add origin "$2"
  git -C "$d" fetch -q --depth 1 origin "$3"; git -C "$d" checkout -q FETCH_HEAD
  echo "$1: $(git -C "$d" log -1 --format='%H %cd %s')"
}
fetch PaddleOCR https://github.com/PaddlePaddle/PaddleOCR.git dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf          # main, 2026-09-16
fetch PaddleFormers https://github.com/PaddlePaddle/PaddleFormers.git 2ca866a9904e5e7b1bf8351f091f9a33de9035dd  # develop, 2026-09-29
echo "DONE third_party $TP"
