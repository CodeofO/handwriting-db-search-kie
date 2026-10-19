"""PP-OCRv5 rec 백본(PPLCNetV3)의 학습/추론 시간축 불일치 보정과 가로 늘리기.

공식 코드(ppocr/modeling/backbones/rec_lcnetv3.py): 학습 때 F.adaptive_avg_pool2d(x, [1, 40]) (폭과 무관하게 40칸),
추론 때 F.avg_pool2d(x, [3, 2]) (폭/8칸). 공식 입력 폭 320에서는 둘 다 40칸이라 같지만, 폭 960이면
학습 40칸 ↔ 추론 120칸이 되어 글자가 겹쳐 나오고, 40칸에서는 짧은 필드(생년월일 약 110px → 4.6칸)가 칸이 모자란다.
모드 (환경 변수 KIE_PPOCR_POOL 또는 apply(mode)):
  fix40 : 추론도 학습과 같은 40칸 (이미 40칸으로 학습한 모델용)
  prop  : 학습도 추론과 같은 폭/8칸 (폭 960 → 120칸, 공식 320의 칸 밀도와 같음) — 재학습용. 추론은 공식 그대로.
폭 320 모델에서는 두 모드 모두 결과가 공식과 같다.
"""
import os

XS = float(os.environ.get("KIE_PPOCR_XSTRETCH", "1") or 1)


def xstretch(img):
    """가로 늘리기 (HANA · IAM: 줄 이미지 폭에 비해 글자가 많아 글자당 칸이 1 미만 → 가로 2배).
    KIE_PPOCR_XSTRETCH=2 일 때 학습·평가·선정·추론 모든 경로에서 같은 함수를 쓴다. img: HWC 배열."""
    if XS == 1 or img is None:
        return img
    import cv2
    return cv2.resize(img, (max(1, int(round(img.shape[1] * XS))), img.shape[0]), interpolation=cv2.INTER_LINEAR)


def apply(mode=None):
    if XS != 1:
        import ppocr.data.imaug.operators as O
        if not getattr(O.DecodeImage, "_kie_xs", False):
            _call = O.DecodeImage.__call__

            def call(self, data):
                data = _call(self, data)
                if data is not None and data.get("image") is not None:
                    data["image"] = xstretch(data["image"])
                return data
            O.DecodeImage.__call__ = call; O.DecodeImage._kie_xs = True
            print(f"[ppocr_patch] DecodeImage 가로 ×{XS}", flush=True)
    mode = mode or os.environ.get("KIE_PPOCR_POOL", "")
    if not mode:
        return
    import paddle.nn.functional as F
    import ppocr.modeling.backbones.rec_lcnetv3 as L
    if getattr(L.PPLCNetV3, "_kie_patched", None):
        return
    _fwd = L.PPLCNetV3.forward
    if mode == "fix40":
        def forward(self, x):
            t = self.training
            self.training = True            # 이 모듈의 풀링 분기만 학습과 같게(하위 층의 train/eval 상태는 그대로)
            try:
                return _fwd(self, x)
            finally:
                self.training = t
    elif mode == "prop":
        _aap = F.adaptive_avg_pool2d

        def forward(self, x):
            if not self.training:
                return _fwd(self, x)

            def aap(x, output_size, *a, **k):   # 학습 분기의 [1, 40]만 [1, 폭/2](= 추론 avg_pool2d([3, 2])와 같은 칸 수)로
                if isinstance(output_size, (list, tuple)) and list(output_size) == [1, 40]:
                    output_size = [1, x.shape[3] // 2]
                return _aap(x, output_size, *a, **k)
            L.F.adaptive_avg_pool2d = aap
            try:
                return _fwd(self, x)
            finally:
                L.F.adaptive_avg_pool2d = _aap
    else:
        raise ValueError(mode)
    L.PPLCNetV3.forward = forward
    L.PPLCNetV3._kie_patched = mode
    print(f"[ppocr_patch] PPLCNetV3 pooling mode = {mode}", flush=True)
