"""공식 PaddleOCR tools/train.py · tools/export_model.py 실행기. 바꾸는 것은 이미지 읽기와 칸 수 보정(patch.apply)뿐:
"blob:{subset}/{row}" 경로를 crops/{ds}/{subset}.blob 에서 읽도록 ppocr.data.simple_dataset의 두 함수를 감싼다.
학습 루프·증강·샘플러·평가·저장은 모두 공식 코드 그대로. PaddleOCR 사본 = {작업 공간}/third_party/PaddleOCR (scripts/setup_third_party.sh).
사용(파일 경로로 실행 — paddle.distributed.launch가 스크립트 경로를 받는다, PYTHONPATH=src):
  env HW_DATASET=xa python -m paddle.distributed.launch --gpus 0,1,2,3 --host 127.0.0.1 src/formrec/baselines/ppocr/train.py train -c ft/xa/ppocr/korean_v5_ft.yml
  python src/formrec/baselines/ppocr/train.py export -c ... -o Global.pretrained_model=... Global.save_inference_dir=...
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]   # 이 폴더의 모듈 이름(train · patch …)이 공식 코드를 가리지 않게

from formrec.common import blob  # noqa: E402
from formrec.common import workspace as cfg  # noqa: E402

PADDLEOCR = str(cfg.THIRD_PARTY / "PaddleOCR")


def main():
    sys.path.insert(0, PADDLEOCR)
    mode = sys.argv.pop(1)
    if mode in ("train", "eval"):
        import ppocr.data.simple_dataset as sd
        _load, _exists = sd._load_image_bytes, sd._img_path_exists
        sd._load_image_bytes = lambda p: blob.read(p) if blob.is_ref(p) else _load(p)
        sd._img_path_exists = lambda p: True if blob.is_ref(p) else _exists(p)
    from formrec.baselines.ppocr import patch
    patch.apply()   # KIE_PPOCR_POOL=fix40|prop (없으면 공식 그대로)
    os.chdir(PADDLEOCR)
    sys.argv[0] = f"tools/{ {'train': 'train', 'eval': 'eval'}.get(mode, 'export_model') }.py"
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
