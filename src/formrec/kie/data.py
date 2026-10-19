"""손글씨 KIE 질의 세트: AI Hub 053 서식에서 이미지 1장 + 키 목록 → 키 순서대로 값.

두 가지 입력(값 = 기존 파싱 라벨 meta/kie_persons.parquet, 분할 = xa와 같은 사람 · 같은 세트):
  xk  (build) 사람 블록(이름 · 주민등록번호 · 주소 칸)의 세로 범위 ± 여백, 페이지 가로 전체 — 인쇄 항목명 · 다른 칸 손글씨 포함.
      같은 페이지의 다른 사람과 행 범위가 겹치면(부 · 모처럼 엇갈림) 누구의 값인지 모호 → clean=False(질의 제외).
  xkp (page)  한 페이지에 한 사람만 파싱된 서식 페이지 전체(라벨 위치 미사용, 흰 여백만 자동 제거) — 검출 없는 KIE.
      질의 대상(clean) = 손글씨 주민등록번호 형태(6자리 + 6~7자리 연속 토큰)가 1개 · 시도 토큰이 1개 · 주소 칸 전체 = 파싱 주소
      (파싱 주소 마지막 토큰 오른쪽 같은 줄에 이어지는 토큰이 없음) — 페이지에 다른 사람 값이 있거나 라벨 주소가 칸 일부만이면 정답이 모호.
키 풀: 이름 = name, 생년월일 = 주민등록번호 앞 6자리, 주소 = 첫 시 · 도부터 주소 토큰 전체(최대 6), 시도 · 시군구 = 주소 1 · 2번째 토큰.
질의마다 키 1~4개를 무작위로 골라 순서를 섞는다(시드 = md5(item_id, k)). 사람마다 xk = train 3 · 그 외 1개, xkp = train 6 · 그 외 3개.
출력: crops/xk/{src}.blob(PNG) · meta/x/{xk,xkp}_items.parquet · meta/x/{xk,xkp}_queries_{subset}.parquet(qid, item_id, row, keys, target)
이미지 변형: --src all --width 1240 (v1) · all3 --width 1860 · all2 --width 0(원본 해상도). src가 all이 아니면 이미지만 다시 만든다(분할표 · 질의 그대로).
사용: python -m formrec.kie.data build [--src all --width 0]   |   python -m formrec.kie.data page [--width 1240]   |   ... requery --qset xkp
"""
import argparse
import hashlib
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.kie.fields import KEYS

OUT = cfg.META / "x"
CROPS = cfg.ROOT / "crops/xk"
MARGIN = 60
SUBSETS = ("train", "val", "tune", "test")


def values(r):
    addr = [r[f"a{j}"] for j in range(1, 7) if r[f"a{j}"]]
    v = {"이름": r["name"], "생년월일": r["birth"], "주소": " ".join(addr), "시도": addr[0] if addr else ""}
    if len(addr) >= 2:
        v["시군구"] = addr[1]
    return v


def label_path(img):
    return img.replace("원천데이터", "라벨링데이터/라벨").replace(".png", ".json")


def _yrange(args):
    """사람 토큰들의 세로 범위 (y0, y1). 파싱 색인 = bbox 목록 순서(이름으로 확인, 어긋나면 id로)."""
    lab, ids, name = args
    bl = json.load(open(lab))["bbox"]
    if not (ids[0] < len(bl) and bl[ids[0]]["data"] == name):
        byid = {b["id"]: b for b in bl}
        bl = [byid.get(t + 1, byid.get(t)) for t in range(max(ids) + 1)]
    ys = [y for t in ids for y in bl[t]["y"]]
    return min(ys), max(ys)


def _crop(args):
    img, y0, y1, width = args
    from PIL import Image
    im = Image.open(img).convert("L")
    W, H = im.size
    c = im.crop((0, max(0, int(y0)), W, min(H, int(y1))))
    if width:
        c = c.resize((width, max(1, round(c.height * width / W))), Image.BILINEAR)
    b = io.BytesIO()
    c.save(b, format="PNG")
    return b.getvalue()


def _page(args):
    """전체 페이지 → 흰 여백 자르기(밝기 < 200인 행 · 열의 범위 ± 20) + 가로 width로 축소. 라벨 위치는 쓰지 않는다."""
    img, width = args
    from PIL import Image
    im = Image.open(img).convert("L")
    a = np.asarray(im)
    rows, cols = np.where((a < 200).sum(1) > 3)[0], np.where((a < 200).sum(0) > 3)[0]
    if len(rows) and len(cols):
        im = im.crop((max(0, cols[0] - 20), max(0, rows[0] - 20), min(a.shape[1], cols[-1] + 20), min(a.shape[0], rows[-1] + 20)))
    im = im.resize((width, max(1, round(im.height * width / im.width))), Image.BILINEAR)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def queries(D, n_per):
    """사람(item)마다 n_per개 질의. 키 1~min(4, 값 있는 키 수)개를 무작위로 · 무작위 순서로."""
    rows = []
    for r in D.itertuples():
        v = json.loads(r.values)
        for k in range(n_per):
            rng = np.random.default_rng(int(hashlib.md5(f"xk:{r.item_id}:{k}".encode()).hexdigest()[:8], 16))
            ks = [x for x in KEYS if x in v and v[x]]
            m = int(rng.integers(1, min(4, len(ks)) + 1))
            sel = list(rng.choice(ks, m, replace=False))
            rows.append(dict(qid=f"{r.item_id}#{k}", item_id=r.item_id, row=r.row, keys=json.dumps(sel, ensure_ascii=False),
                             target="\n".join(f"{x}: {v[x]}" for x in sel)))
    return pd.DataFrame(rows)


def persons():
    """xa 분할표에 있는 사람만, 세트는 xa와 같게."""
    P = pd.read_parquet(cfg.META / "kie_persons.parquet")
    A = pd.read_parquet(OUT / "xa_items.parquet").drop_duplicates("rec_id")
    sub = dict(zip(A.rec_id, A.subset))
    P = P[P.pid.isin(sub)].reset_index(drop=True)
    P["subset"] = P.pid.map(sub)
    return P


def person_regions(P):
    """사람마다 자를 세로 범위와 분할표 행. 겹치지 않는 이웃이 있으면 두 사람 사이 중간선에서 자른다."""
    jobs, rows = [], []
    for _, g in P.groupby("page_id", sort=False):
        for i, r in g.iterrows():
            o = g.drop(i)
            clean = not ((o.y0 <= r.y1) & (o.y1 >= r.y0)).any()
            up = o[o.y1 < r.y0].y1.max() if (o.y1 < r.y0).any() else -1e9
            dn = o[o.y0 > r.y1].y0.min() if (o.y0 > r.y1).any() else 1e9
            c0, c1 = max(r.y0 - MARGIN, (up + r.y0) / 2), min(r.y1 + MARGIN, (r.y1 + dn) / 2)
            jobs.append((r.img, c0, c1))
            rows.append(dict(item_id=r.pid, rec_id=r.pid, src="all", row=len(rows), page_id=r.page_id, kind=r.kind, subset=r.subset, clean=bool(clean),
                             n_page=len(g), values=json.dumps(values(r), ensure_ascii=False)))
    return jobs, pd.DataFrame(rows)


_open = {}


def blob_bytes(src, row):
    """crops/xk/{src}.blob의 row번째 이미지 바이트(HW_DATASET과 무관하게 xk 묶음을 읽는다)."""
    key = (os.getpid(), src)
    if key not in _open:
        _open[key] = (open(CROPS / f"{src}.blob", "rb"), np.load(CROPS / f"{src}.blobidx.npy", mmap_mode="r"))
    f, off = _open[key]
    f.seek(int(off[row]))
    return f.read(int(off[row + 1] - off[row]))


def write_blob(name, fn, jobs, chunksize, every):
    CROPS.mkdir(parents=True, exist_ok=True)
    off = [0]
    tmp = CROPS / f"{name}.blob.tmp"
    with open(tmp, "wb") as f, ProcessPoolExecutor(8) as ex:
        for k, b in enumerate(ex.map(fn, jobs, chunksize=chunksize)):
            f.write(b)
            off.append(off[-1] + len(b))
            if k % every == 0:
                print(f"{name}: {k}/{len(jobs)}", flush=True)
    np.save(CROPS / f"{name}.blobidx.npy", np.array(off, np.int64))
    os.replace(tmp, CROPS / f"{name}.blob")


def build(src="all", width=0):
    P = persons()
    args = []
    for _, r in P.iterrows():
        ids = [r.i_name, r.i_birth] + [r[f"i_a{j}"] for j in range(1, 7) if r[f"i_a{j}"] >= 0]
        args.append((label_path(r.img), ids, r["name"]))
    with ProcessPoolExecutor(8) as ex:
        yr = list(ex.map(_yrange, args, chunksize=64))
    P["y0"], P["y1"] = [a for a, _ in yr], [b for _, b in yr]
    jobs, D = person_regions(P)
    write_blob(src, _crop, [(img, c0, c1, width) for img, c0, c1 in jobs], 64, 5000)
    if src != "all":                                   # 이미지만 다시 만든 경우: 분할표 · 질의는 그대로(같은 행 순서)
        print("DONE xk images", src)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / "xk_items.parquet", index=False)
    write_queries("xk", D)
    print("DONE xk build")


N_PER = {"xk": (3, 1), "xkp": (6, 3)}       # 사람마다 질의 수 (train, 그 외)
ADDR_GAP = 200                             # 주소 칸 이어짐 판정: 같은 줄 오른쪽 토큰까지의 최대 간격(px)


def page_checks(args):
    """한 페이지 라벨 → (주민등록번호 형태 수, 시도 토큰 수, 주소 칸 전체 = 파싱 주소)."""
    lab, ia, sido = args
    bl = json.load(open(lab))["bbox"]
    t = [b["data"] for b in bl]
    n_rrn = sum(1 for i in range(len(t) - 1) if re.fullmatch(r"\d{6}", t[i]) and re.fullmatch(r"\d{6,7}", t[i + 1]))
    n_sido = sum(x in sido for x in t)
    last, nxt = bl[ia[-1]], bl[ia[-1] + 1] if ia[-1] + 1 < len(bl) else None
    complete = True
    if nxt is not None:
        ly, ny = (min(last["y"]), max(last["y"])), (min(nxt["y"]), max(nxt["y"]))
        ov = max(0, min(ly[1], ny[1]) - max(ly[0], ny[0])) / max(1, min(ly[1] - ly[0], ny[1] - ny[0]))
        complete = not (ov >= 0.5 and 0 <= min(nxt["x"]) - max(last["x"]) < ADDR_GAP)
    return n_rrn, n_sido, complete


def page_items(P):
    """한 사람만 파싱된 페이지 → xkp 분할표(여러 사람 페이지는 '이름' 등이 누구인지 정해지지 않아 제외). clean은 page_flags로 채운다."""
    n = P.groupby("page_id").pid.transform("size")
    P = P[n == 1].reset_index(drop=True)
    D = pd.DataFrame([dict(item_id=r.pid, rec_id=r.pid, src="page", row=i, page_id=r.page_id, kind=r.kind, subset=r.subset, clean=True,
                           values=json.dumps(values(r), ensure_ascii=False)) for i, r in P.iterrows()])
    return P, D


def page_flags(P, D, procs=8):
    """xkp 분할표에 n_rrn · n_sido · clean(질의 대상)을 채운다."""
    sido = sorted({a for a in P.a1 if a})
    args = [(label_path(r.img), [int(r[f"i_a{j}"]) for j in range(1, 7) if r[f"i_a{j}"] >= 0], sido) for _, r in P.iterrows()]
    with ProcessPoolExecutor(procs) as ex:
        F = list(ex.map(page_checks, args, chunksize=64))
    D = D.copy()
    D["n_rrn"], D["n_sido"] = [f[0] for f in F], [f[1] for f in F]
    D["clean"] = (D.n_rrn == 1) & (D.n_sido == 1) & np.array([f[2] for f in F])
    return D


def write_queries(qset, D):
    n_train, n_other = N_PER[qset]
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SUBSETS:
        Q = queries(D[(D.subset == s) & D.clean], n_train if s == "train" else n_other)
        Q.to_parquet(OUT / f"{qset}_queries_{s}.parquet", index=False)
        print(qset, s, int((D.subset == s).sum()), "사람 · 질의 대상", int(D[D.subset == s].clean.sum()), "· 질의", len(Q), flush=True)


def build_page(width=1240):
    P, D = page_items(persons())
    D = page_flags(P, D)
    write_blob("page", _page, [(x, width) for x in P.img], 32, 2000)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / "xkp_items.parquet", index=False)
    write_queries("xkp", D)
    print("DONE xkp build", width)


def requery(qset):
    """저장된 분할표({qset}_items)에서 질의만 다시 만든다(이미지 · 분할표 그대로)."""
    write_queries(qset, pd.read_parquet(OUT / f"{qset}_items.parquet"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "page", "requery"])
    ap.add_argument("--qset", default="xkp", help="requery 대상 질의 세트")
    ap.add_argument("--src", default=os.environ.get("XK_SRC", "all"), help="이미지 묶음 이름(build)")
    ap.add_argument("--width", type=int, default=None, help="출력 가로(build 기본 0 = 원본 해상도, page 기본 1240)")
    a = ap.parse_args(argv)
    if a.cmd == "requery":
        requery(a.qset)
    elif a.cmd == "build":
        build(a.src, a.width if a.width is not None else int(os.environ.get("XK_WOUT", 0)))
    else:
        build_page(a.width if a.width is not None else int(os.environ.get("XKP_W", 1240)))


if __name__ == "__main__":
    main()
