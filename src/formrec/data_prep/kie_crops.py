"""AI Hub 서식 2단계: 사람 블록(meta/kie_persons.parquet)의 필드 이미지를 잘라 세트별 blob 하나로 저장한다.

item = 필드 이미지 1장. 사람마다:
  name  : 이름 bbox crop (사방 4px 여백)
  birth : 주민번호 앞 6자리 bbox crop (= 생년월일 필드)
  addr k: 주소 토큰 1..k의 crop을 원래 순서대로 가로로 이어 붙인 한 줄 (k = 2 … n_addr)
          토큰 사이 흰 여백 = 그 사람 주소의 실제 토큰 간격 중앙값(8~80px로 자름). 인쇄 칸 제목 · 괘선은 넣지 않는다.
라벨: name/birth = 그대로, addr = 토큰을 공백 하나로 이은 문자열.
출력(세트 = kie_persons.subset: train / tune / test):
  crops/kie/{subset}.blob + .blobidx.npy (PNG 이어 붙임, 행 = index 순서)  — 작은 파일을 만들지 않는다
  cache/kie/{subset}_index.parquet (image_id = item_id, order = 0, text, pid, ftype, k, subset, kind, official_test)
IO: 페이지 PNG를 한 번씩만 연다. 프로세스 6개.
사용: python -m formrec.data_prep.kie_crops [--limit 50]
"""
import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from formrec.common import workspace as cfg

PAD = 4
SUBSETS = ("train", "tune", "test")
CROPS = cfg.ROOT / "crops/kie"
CACHE = cfg.ROOT / "cache/kie"


def label_json(img):
    return img.replace("원천데이터", "라벨링데이터/라벨").replace(".png", ".json")


def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def person_items(pg, B, p):
    """페이지 그림 · bbox · 사람 1명 → [(pid, ftype, k, text, png)]."""
    W, H = pg.size
    crop = lambda bb: pg.crop((max(0, min(bb["x"]) - PAD), max(0, min(bb["y"]) - PAD), min(W, max(bb["x"]) + PAD), min(H, max(bb["y"]) + PAD)))
    out = [(p["pid"], "name", 0, p["name"], png(crop(B[p["i_name"]]))),
           (p["pid"], "birth", 0, p["birth"], png(crop(B[p["i_birth"]])))]
    toks = [crop(B[p[f"i_a{j}"]]) for j in range(1, p["n_addr"] + 1)]
    gap = int(np.clip(p["gap"], 8, 80))
    for k in range(2, p["n_addr"] + 1):
        cs = toks[:k]
        h = max(c.height for c in cs)
        line = Image.new("L", (sum(c.width for c in cs) + gap * (k - 1), h), 255); x = 0
        for c in cs:
            line.paste(c, (x, (h - c.height) // 2)); x += c.width + gap
        out.append((p["pid"], "addr", k, " ".join(p[f"a{j}"] for j in range(1, k + 1)), png(line)))
    return out


def work(args):
    img, persons = args
    try:
        B = json.load(open(label_json(img)))["bbox"]
        pg = Image.open(img).convert("L")
    except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
        return [], (img, str(e)[:120])
    out = []
    for p in persons:
        out += person_items(pg, B, p)
    return out, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="시험용: 세트마다 앞 N명 → {subset}_limit 이름")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    CROPS.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(cfg.META / "kie_persons.parquet")
    if a.limit:
        P = P.groupby("subset").head(a.limit)
    info = P.set_index("pid")[["subset", "kind", "official_test"]]
    bad = []
    for s in SUBSETS:
        g = P[P.subset == s]
        jobs = [(img, d.to_dict("records")) for img, d in g.groupby("img", sort=True)]
        rows = []
        bpath = CROPS / f"{s}{'_limit' if a.limit else ''}.blob"
        tmp = bpath.with_name(bpath.name + ".tmp")
        off = [0]
        with open(tmp, "wb") as f, ProcessPoolExecutor(a.workers) as ex:
            for n, (items, err) in enumerate(ex.map(work, jobs, chunksize=4)):
                if err:
                    bad.append(err); continue
                for pid, ft, k, text, b in items:
                    f.write(b); off.append(off[-1] + len(b))
                    rows.append(dict(image_id=f"{pid}_{ft}{k or ''}", order=0, text=text, pid=pid, ftype=ft, k=k))
                if (n + 1) % 1000 == 0:
                    f.flush(); os.fsync(f.fileno()); print(f"{s}: {n + 1}/{len(jobs)} pages, {len(rows)} items", flush=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, bpath)
        cfg.atomic_write(bpath.with_suffix(".blobidx.npy"), lambda t: np.save(open(t, "wb"), np.array(off, np.int64)))
        R = pd.DataFrame(rows).join(info, on="pid")
        idx = CACHE / f"{s}_index{'_limit' if a.limit else ''}.parquet"
        cfg.atomic_write(idx, lambda t: R.to_parquet(t, index=False))
        print(s, len(R), "items", R.groupby(["ftype", "k"]).size().to_dict(), f"{off[-1] / 2**20:.0f} MB", flush=True)
    if bad:
        print("skipped pages", len(bad), bad[:3])
    print("DONE kie_crops")


if __name__ == "__main__":
    main()
