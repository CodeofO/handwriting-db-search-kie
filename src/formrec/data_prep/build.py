"""최종 프로토콜의 데이터: 데이터셋 분할표 + 이미지 묶음(blob) + 제안 모델 입력 캐시.

데이터셋 (HW_DATASET)
  xa : AI Hub 053 서식 사람 블록 필드 (이미지 = crops/kie blob 공유, 링크 crops/xa/src_{train,tune,test})
  xh : HANA 이름 줄 (zip 원본에서 높이 96으로 다시 뽑은 blob "all")
  xi : IAM 줄 이미지 (data/iam/raw/lines, 원본 해상도 blob "all")
  xc : xa F2 · F3 레코드의 칼럼 이미지를 가로로 이어 붙인 한 줄 (blob "all")
분할 (최종 train / val / tune / test)
  원본 train/test만 있는 경우: test 고정, 원본 train → train/val/tune = 0.8/0.1/0.1   — xa(공식 Training/Validation), xh(공식 train/test)
  원본 train/val/test가 있는 경우: train → train/tune = 0.9/0.1, val · test 그대로   — xi(RWTH Aachen 분할)
  묶음 단위: xa = 페이지(page_id), xh = 이미지(필기자 ID 없음), xi = 필기자(forms.txt). 해시 = formrec.common.splits(md5("x1:" + 묶음)).
역할: val = 에폭 선정(loss), tune = 하이퍼파라미터 · 수락 임계값, test = 마지막 1회.
출력: meta/x/{ds}_items.parquet — item(이미지 1장) 단위:
  item_id, rec_id(레코드), src(blob 키), row(blob 행), text(이미지에 쓰인 문자열), kind(필드 종류), group(분할 묶음), subset (+ 데이터셋별 열)
제안 모델 입력 캐시: kie_cache.load와 같은 전처리. xa = cache/kie/{train,tune,test}_64x1280.npy 그대로(src 키로),
  xh = cache/xh/all_64x448.npy(높이 64에서 폭 약 418 → 448 안, 읽을 때 1280까지 흰색), xi · xc = cache/{ds}/all_64x1280.npy
사용: python -m formrec.data_prep.build split | links | iam_blob | hana_blob | xc_blob | cache --ds {xh,xi,xc}
      (blob · cache는 IO 작업: 한 번에 하나, 프로세스 8개. hana_blob은 zip 약 47GB 순차 1회)
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.levels import SEP, level_table
from formrec.common.splits import case2, h01

ROOT = cfg.ROOT
OUT = ROOT / "meta/x"
IAM = ROOT / "data/iam/raw"
HANA_ZIP = ROOT / "data/hana/archive.zip"
CACHE_W = {"xh": 448, "xi": 1280, "xc": 1280}


# ── 분할표
def split_xa():
    """AI Hub: 공식 Validation 폴더 → test, Training 폴더 페이지 → train/val/tune 0.8/0.1/0.1. 이미지 = crops/kie blob 행."""
    P = pd.read_parquet(ROOT / "meta/kie_persons.parquet")[["pid", "page_id", "folder"]]
    sub = {r.pid: ("test" if r.folder == "Validation" else case2(r.page_id)) for r in P.itertuples()}
    grp = dict(zip(P.pid, P.page_id))
    rows = []
    for s in ("train", "tune", "test"):
        q = pd.read_parquet(ROOT / f"cache/kie/{s}_index.parquet")
        n = len(np.load(ROOT / f"crops/kie/{s}.blobidx.npy")) - 1
        assert n == len(q), (s, n, len(q))                       # blob 행 = index 행
        rows.append(pd.DataFrame(dict(item_id=q.image_id.values, rec_id=q.pid.values, src=f"src_{s}", row=np.arange(len(q)),
                                      text=q.text.values, kind=np.where(q.ftype == "addr", "addr" + q.k.astype(str), q.ftype),
                                      form=q.kind.values, group=q.pid.map(grp).values, subset=q.pid.map(sub).values)))
    D = pd.concat(rows, ignore_index=True)
    assert D.subset.notna().all()
    return D


def split_xh():
    """HANA: 공식 test 고정, 공식 train → train/val/tune 0.8/0.1/0.1 (이미지 단위). 행 순서 = meta/hana_cols.parquet."""
    C = pd.read_parquet(ROOT / "meta/hana_cols.parquet")
    sub = np.where(C.subset == "test", "test", [case2(i) for i in C.image_id])
    # 이미지에 쓰인 순서 = 성 · 이름 · 중간이름 (예: "Dahl Carl William Heinrich"). 라벨은 원본 그대로 소문자.
    txt = [" ".join(x for x in (a_, b_, c_) if x) for a_, b_, c_ in zip(C["last"], C["first"], C["middle"].fillna(""))]
    D = pd.DataFrame(dict(item_id=C.image_id.values, rec_id=C.image_id.values, src="all", row=np.arange(len(C)),
                          text=txt, kind="name_line", group=C.image_id.values, subset=sub, n_words=C.n_words.values, level=C.level.values))
    D["fields"] = [[a, b, c] for a, b, c in zip(C["last"], C["first"], C["middle"].fillna(""))]   # DB 칼럼 (성 · 이름 · 중간이름)
    D["zip_name"] = "HANA/minipics/HANA/" + C.image_id.values
    D["orig_subset"] = C.subset.values
    return D


def split_xi():
    """IAM: Aachen train/validation/test(페이지 목록). train 페이지 → train/tune 0.9/0.1(필기자 단위). 목록 밖 페이지는 쓰지 않는다."""
    lines = [l.rstrip("\n").split(" ", 8) for l in open(IAM / "lines.txt") if not l.startswith("#")]
    L = pd.DataFrame(lines, columns=["id", "seg", "gray", "ncomp", "x", "y", "w", "h", "text"])
    L["text"] = L.text.str.replace("|", " ", regex=False).str.strip()
    L["form"] = L.id.str.rsplit("-", n=1).str[0]
    F = [l.split() for l in open(IAM / "forms.txt") if not l.startswith("#")]
    writer = {f[0]: f[1] for f in F}
    aach = {}
    for s, name in (("train", "train"), ("validation", "val"), ("test", "test")):
        for f in open(IAM / f"splits/{s}.uttlist").read().split():
            aach[f] = name
    L["aachen"] = L.form.map(aach)
    L = L[L.aachen.notna()].reset_index(drop=True)
    L["writer"] = L.form.map(writer)
    # 필기자마다 줄 수가 달라 md5 < 0.1 로는 tune이 6%에 그친다 → md5 순으로 필기자를 채워 train 줄의 10%가 될 때까지 tune
    tw = L[L.aachen == "train"].groupby("writer").size()
    order = sorted(tw.index, key=lambda w: h01("iam_writer:" + w))
    tune_w, n = set(), 0
    for w in order:
        if n >= 0.1 * tw.sum():
            break
        tune_w.add(w); n += tw[w]
    sub = [a if a != "train" else ("tune" if w in tune_w else "train") for a, w in zip(L.aachen, L.writer)]
    D = pd.DataFrame(dict(item_id=L.id.values, rec_id=L.id.values, src="all", row=np.arange(len(L)), text=L.text.values, kind="line",
                          group=L.writer.values, subset=sub, form=L.form.values, seg=L.seg.values, n_words=L.text.str.split().str.len().values))
    D["path"] = [str(IAM / f"{i.split('-')[0]}/{i.rsplit('-', 1)[0]}/{i}.png") for i in L.id]
    return D


def split_xc():
    """(b) 한 줄 여러 칼럼: xa F2(이름 · 생년월일) · F3(+주소 4토큰) 레코드 → 칼럼 이미지를 가로로 이은 이미지 1장.
    정답 = 칼럼을 이미지 순서대로 공백으로 이은 문자열(HANA와 같은 표기). 분할 = xa와 같은 사람 · 같은 세트.
    단계 C2 = F2와 같은 사람 · 같은 내용, C3 = F3와 같은 사람 · 같은 내용. → (분할표, 행마다 칼럼 item_id 목록)"""
    A = pd.read_parquet(OUT / "xa_items.parquet"); grp = dict(zip(A.rec_id, A.group))
    rows, srcs = [], []
    for sub in ("train", "val", "tune", "test"):
        L = level_table("xa", sub)
        for lv, c in (("F2", "C2"), ("F3", "C3")):
            for r in L[L.level == lv].itertuples():
                fl = r.text.split(SEP)
                rows.append(dict(item_id=f"{r.pid}_{c}", rec_id=f"{r.pid}_{c}", pid=r.pid, src="all", row=len(rows), text=" ".join(fl), fields=fl,
                                 kind="row", level=c, group=grp.get(r.pid, r.pid), subset=sub))
                srcs.append(list(r.items))
    return pd.DataFrame(rows), srcs


def cmd_split():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds, fn in (("xa", split_xa), ("xh", split_xh), ("xi", split_xi)):
        D = fn()
        D.to_parquet(OUT / f"{ds}_items.parquet", index=False)
        print(ds, "이미지", D.groupby("subset").size().to_dict(), "레코드", D.groupby("subset").rec_id.nunique().to_dict())
        g = D.groupby("group").subset.nunique()
        assert (g == 1).all(), f"{ds}: 묶음이 두 세트에 걸침 {int((g > 1).sum())}"
    print("DONE x_data split", OUT)


def cmd_links():
    d = ROOT / "crops/xa"; d.mkdir(parents=True, exist_ok=True)
    for s in ("train", "tune", "test"):
        for suf in (".blob", ".blobidx.npy"):
            p = d / f"src_{s}{suf}"
            if not p.exists():
                p.symlink_to(Path("../kie") / f"{s}{suf}")
    print("DONE links", sorted(os.listdir(d)))


# ── 이미지 묶음
def png_bytes(im):
    b = io.BytesIO(); im.save(b, format="PNG", optimize=False); return b.getvalue()


def write_blob(path_blob, items, load):
    """items 순서대로 PNG 바이트를 이어 붙인다. load(item) → PNG bytes."""
    off = [0]
    tmp = path_blob.with_suffix(".blob.tmp")
    with open(tmp, "wb") as f:
        for k, it in enumerate(items):
            b = load(it); f.write(b); off.append(off[-1] + len(b))
            if k % 50000 == 0:
                print(f"{path_blob.name}: {k}/{len(items)}", flush=True)
    np.save(path_blob.with_name(path_blob.stem + ".blobidx.npy"), np.array(off, np.int64))
    os.replace(tmp, path_blob)


def write_blob_parallel(path_blob, results, total, every):
    """병렬 결과(PNG 바이트, 또는 바이트 목록) 반복자를 순서대로 이어 붙인다."""
    off = [0]; tmp = path_blob.with_name(path_blob.name + ".tmp")
    with open(tmp, "wb") as f:
        for k, res in enumerate(results):
            for b in (res if isinstance(res, list) else [res]):
                f.write(b); off.append(off[-1] + len(b))
            if k % every == 0:
                print(f"{path_blob.parent.name}: {len(off) - 1}/{total}", flush=True)
    np.save(path_blob.with_name(path_blob.stem + ".blobidx.npy"), np.array(off, np.int64)); os.replace(tmp, path_blob)


def iam_png(path):
    from PIL import Image
    return png_bytes(Image.open(path).convert("L"))     # 원본 해상도 그대로


def cmd_iam_blob():
    D = pd.read_parquet(OUT / "xi_items.parquet")
    d = ROOT / "crops/xi"; d.mkdir(parents=True, exist_ok=True)
    write_blob(d / "all.blob", list(D.path), iam_png)
    print("DONE iam_blob", len(D))


def hana_png(z, name, H=96):
    """zip 멤버 → 높이 H(비율 유지, BILINEAR) 그레이스케일 PNG. 원본 약 1045×160."""
    from PIL import Image
    im = Image.open(io.BytesIO(z.read(name))).convert("L")
    w = max(1, round(im.width * H / im.height))
    return png_bytes(im.resize((w, H), Image.BILINEAR))


def _hana_chunk(args):
    import zipfile
    names, H = args
    z = zipfile.ZipFile(HANA_ZIP)
    return [hana_png(z, n, H) for n in names]


def cmd_hana_blob(H=96, procs=8):
    """프로세스 8개(각자 zip 핸들), 순서 유지."""
    from concurrent.futures import ProcessPoolExecutor
    D = pd.read_parquet(OUT / "xh_items.parquet")
    d = ROOT / "crops/xh"; d.mkdir(parents=True, exist_ok=True)
    names = list(D.zip_name); CH = 2000
    chunks = [(names[i:i + CH], H) for i in range(0, len(names), CH)]
    with ProcessPoolExecutor(procs) as ex:
        write_blob_parallel(d / "all.blob", ex.map(_hana_chunk, chunks), len(names), 25)
    print("DONE hana_blob", len(names))


def xc_compose(pngs, HC=64, GAP=32):
    """칼럼 crop PNG 여러 장 → 높이 HC로 맞춰 가로로 이어 붙인 한 줄 PNG(칼럼 사이 흰 여백 GAP px)."""
    from PIL import Image
    ims = []
    for b in pngs:
        im = Image.open(io.BytesIO(b)).convert("L")
        ims.append(im.resize((max(1, round(im.width * HC / im.height)), HC), Image.BILINEAR))
    W_ = sum(i.width for i in ims) + GAP * (len(ims) - 1)
    out = Image.new("L", (W_, HC), 255); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + GAP
    return png_bytes(out)


def _xc_compose(args):
    pngs, HC, GAP = args
    return xc_compose(pngs, HC, GAP)


class XaReader:
    """xa item_id → 칼럼 crop PNG 바이트 (crops/xa/src_* blob)."""
    def __init__(self):
        A = pd.read_parquet(OUT / "xa_items.parquet"); self.loc = dict(zip(A.item_id, zip(A.src, A.row)))
        self.bl = {s: (np.load(ROOT / f"crops/xa/{s}.blobidx.npy"), open(ROOT / f"crops/xa/{s}.blob", "rb")) for s in set(A.src)}

    def __call__(self, iid):
        s, r = self.loc[iid]; off, f = self.bl[s]; f.seek(int(off[r])); return f.read(int(off[r + 1] - off[r]))


def cmd_xc_blob(procs=8):
    from concurrent.futures import ProcessPoolExecutor
    D, srcs = split_xc()
    rd = XaReader()
    jobs = (([rd(i) for i in its], 64, 32) for its in srcs)
    d = ROOT / "crops/xc"; d.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(procs) as ex:
        write_blob_parallel(d / "all.blob", ex.map(_xc_compose, jobs, chunksize=256), len(D), 20000)
    D.to_parquet(OUT / "xc_items.parquet", index=False)
    print("DONE xc_blob", D.groupby(["subset", "level"]).size().to_dict())


# ── 제안 모델 입력 캐시
def cmd_cache(ds, procs=8):
    from formrec.data_prep.kie_cache import write_cache
    H, W = 64, CACHE_W[ds]
    out = ROOT / f"cache/{ds}/all_{H}x{W}.npy"
    n = write_cache(ROOT / f"crops/{ds}/all.blob", ROOT / f"crops/{ds}/all.blobidx.npy", out, H, W, procs, chunksize=256, flush_every=100000)
    print("DONE cache", ds, n, out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "links", "iam_blob", "hana_blob", "cache", "xc_blob"])
    ap.add_argument("--ds", default="")
    a = ap.parse_args(argv)
    if a.cmd == "cache":
        cmd_cache(a.ds)
    else:
        {"split": cmd_split, "links": cmd_links, "iam_blob": cmd_iam_blob, "hana_blob": cmd_hana_blob, "xc_blob": cmd_xc_blob}[a.cmd]()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
