"""PP-OCRv5 rec 추론(공식 paddleocr TextRecognition) — 파인튜닝 모델(--model_dir) 또는 공식 가중치.

- 질의 = cfg.queries(subset) 순서(분할표 meta/x/{ds}_items.parquet의 그 세트 행). 이미지는 blob의 PNG 바이트를 디코딩해 BGR 배열로 넣는다.
- 생성 결과 정리 = formrec.baselines.normalize (정답 라벨을 망가뜨리지 않는지 1회 점검).
- val 세트에서 bn_select.csv가 있으면, 공식 추론기 정확도가 선정 때 잰 val 정확도와 KIE_CHECK_TOL(기본 0.03) 안인지 대조한다.
출력: results/{ds}/g1/{out_tag or tag}_{subset}.parquet  (image_id, order, text, raw, pred, score)
사용: env HW_DATASET=xa KIE_PPOCR_POOL=prop python -m formrec.baselines.ppocr.infer --subset tune --tag ppocrv5ft \
        --model_name korean_PP-OCRv5_mobile_rec --model_dir runs/xa/ppocrv5_ft/infer
"""
import argparse
import io
import os
import time

import numpy as np
import pandas as pd

from formrec.common import blob
from formrec.common import workspace as cfg
from formrec.baselines.normalize import allowed_chars, make_normalizer


def load_images(subset):
    """(질의 표, PNG 바이트 목록) — 분할표의 blob 참조 순서."""
    assert cfg.DATASET in cfg.XDS, f"HW_DATASET={cfg.DATASET}: 분할표 데이터셋만({cfg.XDS})"
    q = cfg.queries(subset)
    return q, [blob.read(f"blob:{s}/{r}") for s, r in zip(q.src, q.row)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(cfg.SUBSETS))
    ap.add_argument("--tag", default="ppocrv5")
    ap.add_argument("--model_dir", default="", help="비우면 --model_name의 공식 가중치(자동 다운로드), 한국어는 models/ 사본")
    ap.add_argument("--model_name", default="korean_PP-OCRv5_mobile_rec", help="예: en_PP-OCRv5_mobile_rec (영어 · 덴마크어 데이터셋)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="스모크: 앞 N개만")
    ap.add_argument("--out_tag", default="", help="출력 이름(기본 = --tag)")
    return ap.parse_args(argv)


def decode_bgr(b):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))[:, :, ::-1].copy()


def main(argv=None):
    a = parse_args(argv)
    from paddleocr import TextRecognition
    from formrec.baselines.ppocr import patch as ppocr_patch
    from formrec.common import input_check
    md = a.model_dir or (str(cfg.MODELS / "korean_PP-OCRv5_mobile_rec/infer") if a.model_name.startswith("korean") else None)
    rec = TextRecognition(model_name=a.model_name, model_dir=md, device="gpu:0")
    q, pngs = load_images(a.subset)
    if a.limit:
        q, pngs = q.iloc[:a.limit].reset_index(drop=True), pngs[:a.limit]
    t0 = time.time(); preds, scores = [], []
    for i in range(0, len(pngs), 4096):
        imgs = [ppocr_patch.xstretch(decode_bgr(b)) for b in pngs[i:i + 4096]]   # KIE_PPOCR_XSTRETCH(학습과 같게), 기본 1이면 그대로
        for r in rec.predict(imgs, batch_size=a.batch):
            preds.append(r["rec_text"]); scores.append(float(r["rec_score"]))
        print(f"{len(preds)}/{len(q)} {time.time() - t0:.0f}s", flush=True)
    out = q[["image_id", "order", "text"]].copy()
    out["raw"] = preds
    norm = make_normalizer(cfg.DATASET, allowed_chars(cfg.DATASET, cfg.queries("train").text))
    if not input_check.check_normalize(out.text, norm, f"{cfg.DATASET}_{a.tag}_normalize")["pass"]:
        raise SystemExit("생성 결과 정리 점검 실패 — logs/checks/ 확인")
    out["pred"] = [norm(r) for r in out.raw]
    sel = os.path.join(os.path.dirname(os.path.abspath(a.model_dir.rstrip("/"))), "bn_select.csv") if a.model_dir else ""
    if a.subset == "val" and not a.limit and sel and os.path.exists(sel):   # 선정 때 잰 세트(val)와 같을 때만 대조
        ref = pd.read_csv(sel).val_acc.max(); got = float((out.raw == out.text).mean())   # 같은 기준(정리 전 원문 일치)
        print(f"[check] val exact(원문): 추론기 {got:.4f} vs 선정 {ref:.4f} · 정리 후 {(out.pred == out.text).mean():.4f}", flush=True)
        tol = float(os.environ.get("KIE_CHECK_TOL", "0.03"))
        if abs(got - ref) > tol:
            raise SystemExit(f"추론기 정확도 {got:.4f}가 선정 때 val {ref:.4f}와 {tol:.2f} 넘게 다름 — 추론 전처리·내보내기 확인")
    out["score"] = scores
    p = cfg.RESULTS / f"g1/{a.out_tag or a.tag}_{a.subset}.parquet"
    cfg.atomic_write(p, lambda t: out.to_parquet(t, index=False))
    el = time.time() - t0
    print(f"DONE {a.tag} {a.subset} exact={(out.pred == out.text).mean():.4f} n={len(out)} {1000 * el / len(out):.2f} ms/img")


if __name__ == "__main__":
    main()
