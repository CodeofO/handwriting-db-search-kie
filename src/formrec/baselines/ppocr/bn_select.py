"""PP-OCRv5 파인튜닝 에폭 선정 + BN 통계 재추정.

문제: 넓은 입력(폭 960 등) 학습에서 BN 누적 통계(running mean/var)가 실제 배치 통계와 어긋나, 공식 평가(eval 모드)가
train 이미지에서도 0%가 된다(같은 가중치로 BN을 배치 통계로 돌리면 정상 인식) → 공식 best_accuracy 선정이 무의미.
해결(표준 BN 재추정): 에폭 체크포인트마다 train 이미지 K장(학습과 같은 전처리, 증강 없음)으로 BN 통계를 누적 평균으로 다시 구하고,
val 전체로 에폭을 고른다 — 기본 프로토콜은 --select loss(val CTC loss 최소), 정확도는 항상 함께 기록(부록 · 대체 규칙).
출력: {run}/bn_best.pdparams(+ .states 복사), {run}/bn_select.csv. 이후 scripts/ppocr_ft_post.sh가 bn_best를 export.
사용: env HW_DATASET=xa KIE_PPOCR_POOL=prop CUDA_VISIBLE_DEVICES=0 python -m formrec.baselines.ppocr.bn_select \
        --run runs/xa/ppocrv5_ft --cfg ft/xa/ppocr/korean_v5_ft.yml --select loss --last 0
"""
import argparse
import glob
import io
import math
import os
import re
import shutil
import sys
import time

import numpy as np

from formrec.common import blob
from formrec.common import workspace as cfg
from formrec.baselines.ppocr import patch as ppocr_patch


def resize_train_style(img, shape):
    """학습(MultiScaleDataSet.resize_norm_img, padding=True)과 같은 전처리: 비율 유지, 폭 초과는 폭에 맞춰 누름, 오른쪽 0 채움."""
    import cv2
    C, H, W = shape
    rw = min(W, int(math.ceil(H * img.shape[1] / img.shape[0])))
    r = cv2.resize(img, (rw, H)).astype("float32").transpose(2, 0, 1) / 255
    out = np.zeros((C, H, W), np.float32); out[:, :, :rw] = (r - 0.5) / 0.5
    return out, rw / W


def load_imgs(lst, shape):
    """blob 참조 목록 → (N, C, H, W) 배열. 학습과 같은 전처리(공식 평가의 resize_norm_img_chinese는 폭 초과 이미지 때문에
    배치 폭이 늘어 같은 배치의 다른 이미지까지 손해 — 추론기와 차이의 원인)."""
    import cv2
    from PIL import Image
    xs = [resize_train_style(ppocr_patch.xstretch(cv2.cvtColor(np.asarray(Image.open(io.BytesIO(blob.read(p))).convert("RGB")), cv2.COLOR_RGB2BGR)), shape)[0]
          for p in lst]
    W = max(x.shape[2] for x in xs)
    out = np.zeros((len(xs), xs[0].shape[0], xs[0].shape[1], W), np.float32)
    for i, x in enumerate(xs):
        out[i, :, :, :x.shape[2]] = x
    return out


def field_kind(t):
    """정답 문자열의 필드 종류(이름 · 생년월일 · 주소 k토큰) — 종류별 정확도 기록용."""
    return f"addr{len(t.split(' '))}" if " " in t else ("birth" if t.isdigit() else "name")


def epoch_checkpoints(run):
    """{run}/epochs/ep*.pdparams 또는 공식 저장 {run}/iter_epoch_*.pdparams → {에폭: 경로}."""
    ck = {}
    for p in glob.glob(f"{run}/epochs/ep*.pdparams") + glob.glob(f"{run}/iter_epoch_*.pdparams"):
        ck[int(re.findall(r"(\d+)\.pdparams$", p)[0])] = p
    return ck


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--calib", type=int, default=8192, help="BN 재추정에 쓰는 train 이미지 수")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--check_warn_only", action="store_true", help="입력 점검 실패여도 계속(스모크 · 이미 알려진 결함 모델용)")
    ap.add_argument("--pool", default="", help="fix40 | prop (formrec.baselines.ppocr.patch), 비우면 KIE_PPOCR_POOL")
    ap.add_argument("--val_limit", type=int, default=0, help="스모크: val 앞 N개")
    ap.add_argument("--select", default="acc", choices=["acc", "loss"],
                    help="loss = val CTC loss(추론 헤드, BN 재추정 뒤) 최소 에폭(프로토콜). 정확도는 항상 함께 기록")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sys.path.insert(0, str(cfg.THIRD_PARTY / "PaddleOCR"))
    import paddle
    import paddle.nn.functional as F_
    import pandas as pd
    import yaml
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    BN = (paddle.nn.BatchNorm2D, paddle.nn.BatchNorm1D, paddle.nn.BatchNorm)

    ppocr_patch.apply(a.pool or None)
    c = yaml.safe_load(open(a.cfg))
    shape = next(t["RecResizeImg"]["image_shape"] for t in c["Eval"]["dataset"]["transforms"] if "RecResizeImg" in t)
    post = build_post_process(c["PostProcess"], c["Global"])
    ch = len(post.character)
    for h in c["Architecture"]["Head"]["head_list"]:
        if "CTCHead" in h:
            h["CTCHead"]["out_channels"] = ch
        if "NRTRHead" in h:
            h["NRTRHead"]["out_channels"] = ch + 3
    c["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": ch, "NRTRLabelDecode": ch + 3}
    m = build_model(c["Architecture"])
    cidx = {c_: i for i, c_ in enumerate(post.character)}          # 0 = blank
    tr = [l.rstrip("\n").split("\t")[0] for l in open(c["Train"]["dataset"]["label_file_list"][0])]
    va = [l.rstrip("\n").split("\t") for l in open(c["Eval"]["dataset"]["label_file_list"][0])]
    va = va[:a.val_limit] if a.val_limit else va
    rng = np.random.default_rng(0)
    calib = [tr[i] for i in rng.choice(len(tr), min(a.calib, len(tr)), replace=False)]
    t0 = time.time(); Xc = [load_imgs(calib[i:i + a.bs], shape) for i in range(0, len(calib), a.bs)]
    print(f"calib batches {len(Xc)} {time.time() - t0:.0f}s", flush=True)

    t0 = time.time(); Xv = [(load_imgs([p for p, _ in va[i:i + 256]], shape).astype(np.float16), [t for _, t in va[i:i + 256]])
                            for i in range(0, len(va), 256)]           # val은 한 번만 읽는다 (fp16)
    print(f"val batches {len(Xv)} {time.time() - t0:.0f}s", flush=True)
    ck = epoch_checkpoints(a.run)
    eps = sorted(ck)[-a.last:] if a.last else sorted(ck)
    print("epochs", eps, flush=True)
    rows = []
    for e in eps:
        m.set_state_dict(paddle.load(ck[e]))
        bns = [s for s in m.sublayers() if isinstance(s, BN)]
        m.eval()
        for s in bns:                                # 누적 평균으로 재추정: t번째 배치 momentum = t/(t+1)
            s._mean.set_value(paddle.zeros_like(s._mean)); s._variance.set_value(paddle.ones_like(s._variance)); s.training = True
        with paddle.no_grad():
            for t, xb in enumerate(Xc):
                for s in bns:
                    s._momentum = t / (t + 1)
                m(paddle.to_tensor(xb))
        for s in bns:
            s.training = False; s._momentum = 0.9
        ok = n = 0; nd = 0.0; ls = 0.0; nl = 0
        byk = {}                                     # 필드 종류별 정확도
        with paddle.no_grad():
            for xb, tb in Xv:
                out = m(paddle.to_tensor(xb.astype(np.float32)))
                out = out["ctc"] if isinstance(out, dict) else out
                if a.select == "loss":                   # val CTC loss: 추론 헤드 출력(softmax 확률) → log → warpctc(내부 softmax = 원래 확률)
                    lab = [[cidx.get(c_, -1) for c_ in t] for t in tb]
                    keep = [i for i, l_ in enumerate(lab) if l_ and min(l_) >= 1]
                    if keep:
                        L_ = max(len(lab[i]) for i in keep)
                        lt = paddle.to_tensor(np.array([lab[i] + [0] * (L_ - len(lab[i])) for i in keep], np.int32))
                        lp = paddle.log(out.clip(1e-12))[paddle.to_tensor(keep)].transpose([1, 0, 2])
                        T_ = lp.shape[0]
                        nll = F_.ctc_loss(lp, lt, paddle.to_tensor(np.full(len(keep), T_, np.int64)),
                                          paddle.to_tensor(np.array([len(lab[i]) for i in keep], np.int64)), blank=0, reduction="none")
                        ls += float(nll.sum()); nl += len(keep)
                idx = paddle.argmax(out, axis=2).numpy()                 # CTC 탐욕 복호(공식 CTCLabelDecode와 같음)
                pred = ["".join(post.character[k] for j, k in enumerate(r) if k != 0 and (j == 0 or k != r[j - 1])) for r in idx]
                ok += sum(pp == t for pp, t in zip(pred, tb)); n += len(tb)
                nd += sum(1 - Levenshtein.distance(pp, t) / max(len(pp), len(t), 1) for pp, t in zip(pred, tb))
                for pp, t in zip(pred, tb):
                    v = byk.setdefault(field_kind(t), [0, 0]); v[0] += pp == t; v[1] += 1
        rows.append(dict(epoch=e, val_acc=ok / n, val_1mned=nd / n, val_loss=(ls / nl if nl else float("nan")), val_loss_n=nl,
                         **{f"acc_{k}": v[0] / v[1] for k, v in sorted(byk.items())}))
        print(f"epoch {e}: val acc (BN 재추정) {ok / n:.4f} · 1−NED {nd / n:.4f} · " + " ".join(f"{k} {v[0] / v[1]:.3f}" for k, v in sorted(byk.items()))
              + f" {time.time() - t0:.0f}s", flush=True)
        if (rows[-1]["val_acc"] >= max(r["val_acc"] for r in rows)) if a.select == "acc" else (rows[-1]["val_loss"] <= min(r["val_loss"] for r in rows)):
            paddle.save(m.state_dict(), f"{a.run}/bn_best.pdparams")
            src = ck[e].replace(".pdparams", ".states")
            shutil.copy(src if os.path.exists(src) else f"{a.run}/latest.states", f"{a.run}/bn_best.states")
    R = pd.DataFrame(rows); R.to_csv(f"{a.run}/bn_select.csv", index=False)
    b = R.sort_values("val_acc").iloc[-1] if a.select == "acc" else R.sort_values("val_loss", ascending=False).iloc[-1]
    if a.select == "loss":
        ba = R.sort_values("val_acc").iloc[-1]
        print(f"선정(val loss) ep{int(b.epoch)} loss {b.val_loss:.4f} acc {b.val_acc:.4f} · 참고(val 정확도 최고) ep{int(ba.epoch)} acc {ba.val_acc:.4f}", flush=True)
    from formrec.common import input_check                # 평가 시작 전 입력 처리 점검(선정된 가중치로 학습/추론 모드 정확도 차이 포함)
    r = input_check.ppocr(a.cfg, f"{os.path.basename(os.path.dirname(a.run.rstrip('/')))}_{os.path.basename(a.run.rstrip('/'))}_bnbest", ckpt=f"{a.run}/bn_best.pdparams")
    if not r["pass"] and not a.check_warn_only:
        sys.exit("입력 점검 실패 — logs/checks/ 확인")
    print(f"DONE ppocr_bn_select best epoch {int(b.epoch)} val acc {b.val_acc:.4f} (선정 기준 {a.select})")


if __name__ == "__main__":
    main()
