"""AI Hub 053 라벨 JSON → bbox 단위 표 (Training · Validation 전 셀: PO · PR · TO · TR).

bbox 좌표는 x:[x1,x1,x2,x2], y:[y1,y2,y1,y2] 형태라 min/max로 정규화한다. order = JSON 안 bbox 순서.
출력: meta/bbox.parquet (folder, cell, image_id, order, id, text, x1, y1, x2, y2, img_w, img_h, writer_age, writer_sex)
IO: 작은 JSON 약 9.6만 개 읽기 → 프로세스 8개(낮은 병렬도), IO 작업은 한 번에 하나.
사용: python -m formrec.data_prep.aihub_bbox
"""
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from formrec.common import workspace as cfg


def parse(args):
    folder, cell, path = args
    d = json.load(open(path, encoding="utf-8"))
    im = d["Images"]
    return [dict(folder=folder, cell=cell, image_id=im["identifier"], order=o, id=b["id"], text=str(b["data"]),
                 x1=min(b["x"]), y1=min(b["y"]), x2=max(b["x"]), y2=max(b["y"]), img_w=im["width"], img_h=im["height"],
                 writer_age=im.get("writer_age"), writer_sex=im.get("writer_sex"))
            for o, b in enumerate(d.get("bbox", []))]


def jobs():
    return [(f, c, str(p)) for f in cfg.FOLDERS for c in cfg.CELLS for p in sorted(cfg.label_dir(f, c).glob("*.json"))]


def main(workers=8):
    J = jobs()
    print("json files:", len(J), flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, rs in enumerate(ex.map(parse, J, chunksize=256)):
            rows += rs
            if k % 20000 == 0:
                print(f"{k}/{len(J)}", flush=True)
    df = pd.DataFrame(rows)
    cfg.atomic_write(cfg.BBOX, lambda p: df.to_parquet(p, index=False))
    print(df.groupby(["folder", "cell"]).agg(images=("image_id", "nunique"), bboxes=("text", "size")))
    print("DONE bbox", len(df))


if __name__ == "__main__":
    main()
