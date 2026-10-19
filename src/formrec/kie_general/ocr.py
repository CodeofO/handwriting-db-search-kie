"""일반 KIE 데이터(xkg) 1단계: AI Hub 053 서식 페이지 전체(O.Form)의 인쇄 글자 검출 · 인식.

목적: 라벨에는 손글씨 토큰만 있다 → 인쇄된 항목명(키)과 위치를 얻어 "인쇄 항목명 → 옆 칸 손글씨 값" 짝을 만든다(cells · build).
모델: PP-OCRv5_server_det(검출) + korean_PP-OCRv5_mobile_rec(공개 가중치, 인쇄 글자용). 문서 방향 · 왜곡 보정 끔.
출력: cache/xkg/ocr_{shard}.parquet (page_id, img, folder, boxes = JSON [[text, x0, y0, x1, y1, score], …]) — 이어서 실행 가능(한 페이지는 건너뜀)
사용: CUDA_VISIBLE_DEVICES=0 python -m formrec.kie_general.ocr --shard 0 --n 4   (paddle 환경)
"""
import argparse
import glob
import json
import time
from pathlib import Path

import pandas as pd

from formrec.common import workspace as cfg


def pages():
    return sorted(glob.glob(str(cfg.RAW / "*/원천데이터/HW-OCR/4.Validation/*/O.Form/*.png")))


def boxes_json(r):
    """PaddleOCR 결과 1장 → [[text, x0, y0, x1, y1, score], …]"""
    return [[t, *map(int, b), round(float(s), 3)] for t, b, s in zip(r["rec_texts"], r["rec_boxes"], r["rec_scores"])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="gpu:0")
    a = ap.parse_args(argv)
    ps = pages()[a.shard::a.n]
    if a.limit:
        ps = ps[:a.limit]
    out = cfg.ROOT / f"cache/xkg/ocr_{a.shard}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r.img: r for r in pd.read_parquet(out).itertuples()} if out.exists() else {}
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device=a.device)
    rows = [dict(page_id=r.page_id, img=r.img, folder=r.folder, boxes=r.boxes) for r in done.values()]
    t0, n0 = time.time(), len(rows)
    for k, p in enumerate(ps):
        if p in done:
            continue
        try:
            bx = boxes_json(ocr.predict(p)[0])
        except Exception as e:                       # 깨진 페이지는 오류 표시로 기록하고 넘어감
            bx = [["#ERR " + type(e).__name__, 0, 0, 0, 0, 0.0]]
        rows.append(dict(page_id=Path(p).stem, img=p, folder="Validation" if "/2.Validation/" in p else "Training", boxes=json.dumps(bx, ensure_ascii=False)))
        if (len(rows) - n0) % 500 == 0:
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"shard {a.shard}: {k + 1}/{len(ps)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"DONE xkg_ocr shard {a.shard} {len(rows)} pages")


if __name__ == "__main__":
    main()
