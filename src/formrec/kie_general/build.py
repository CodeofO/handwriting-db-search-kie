"""일반 KIE 데이터(xkg) 2단계: 인쇄 항목명(키) ↔ 손글씨 값 짝(cells.pairs_cell) → 분할 · 질의 · 페이지 이미지.

입력: cache/xkg/ocr_*.parquet(인쇄 글자, ocr), AI Hub 라벨 json(손글씨 토큰 · bbox — 데이터 구축에만, 모델 입력은 페이지 이미지뿐).
정리: 드문 키(20페이지 미만 — 라벨에서 빠진 손글씨가 인쇄 글자로 잡힌 것 등) 제거.
서식 묶기: 서식 번호([서식 NN])를 못 찾은 페이지는 인쇄 항목명 집합이 가장 비슷한(자카드 ≥ 0.5) 서식으로, 없으면 '기타:…'.
분할: 공식 Training 폴더 → train/val/tune(xa와 같은 페이지 해시), Validation 폴더 → test.
  서식 분리 평가: 서식 번호 일부(15%, 시드 7)를 통째로 test_tpl로 — 그 서식의 페이지는 학습 · 선정 · 보정에 쓰지 않는다(처음 보는 서식).
질의: 페이지당 1개, 키 1~5개 무작위 + 질의의 25%에 네거티브 키 1개('[없음]', A = 페이지에 없는 키, B = 빈 칸).
출력: meta/x/xkg_items.parquet (page_id, img, folder, tpl, kv, n_kv, empty, present, subset, row), meta/x/xkg_vocab.json,
      crops/xk/gpage.blob(JPEG, 가로 1240, 흰 여백 자동 제거), meta/x/xkg_queries_{subset}.parquet
사용: python -m formrec.kie_general.build [--limit N] [--no_images]   (paddle 환경 — cv2 필요)
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.splits import case2
from formrec.kie_general.cells import pairs_cell

OUT = cfg.META / "x"
PW = 1240
NONE = "[없음]"
MIN_PAGES = 20                   # 키가 나와야 하는 최소 페이지 수
SUBSETS = ("train", "val", "tune", "test", "test_tpl")


def core(k):
    """키 핵심부(구역 접두 · 괄호 · 공백 제거): 네거티브 A가 페이지에 실제로 있는 항목을 '없음'으로 만들지 않게 비교용."""
    return re.sub(r"\([^()]*\)|\s", "", k.split(" ")[-1])


def _jpg(img):
    from PIL import Image
    im = Image.open(img).convert("L")
    a = np.asarray(im)
    rows, cols = np.where((a < 200).sum(1) > 3)[0], np.where((a < 200).sum(0) > 3)[0]
    if len(rows) and len(cols):
        im = im.crop((max(0, cols[0] - 20), max(0, rows[0] - 20), min(a.shape[1], cols[-1] + 20), min(a.shape[0], rows[-1] + 20)))
    im = im.resize((PW, max(1, round(im.height * PW / im.width))), Image.BILINEAR)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=90)
    return b.getvalue()


def queries(D, n_per, vocab, p_neg=0.25):
    """키 1~5개 무작위 + 질의의 p_neg에 네거티브 키 1개를 무작위 위치에: A = 이 페이지 인쇄 글자에 없는 키(다른 서식의 흔한 항목명),
    B = 항목명은 있는데 손글씨가 없는 칸. 네거티브의 정답 = '[없음]'."""
    rows = []
    for r in D.itertuples():
        kv = json.loads(r.kv)
        ks, pres, emp = sorted(kv), set(json.loads(r.present)), json.loads(r.empty)
        txt = re.sub(r"\s", "", "".join(pres) + "".join(kv) + "".join(emp))
        absent = [k for k in vocab if " " not in k and core(k) and core(k) not in txt]   # 구역 접두 키 제외 · 핵심부가 페이지에 없을 때만
        for k in range(n_per):
            rng = np.random.default_rng(int(hashlib.md5(f"xkg:{r.page_id}:{k}".encode()).hexdigest()[:8], 16))
            sel = list(rng.choice(ks, int(rng.integers(1, min(5, len(ks)) + 1)), replace=False))
            tg, neg = {x: kv[x] for x in sel}, ""
            if rng.random() < p_neg:
                kind = "B" if emp and rng.random() < 0.5 else "A"
                pool = emp if kind == "B" else absent
                if pool:
                    nk = pool[rng.integers(len(pool))]
                    sel.insert(int(rng.integers(len(sel) + 1)), nk)
                    tg[nk], neg = NONE, kind
            rows.append(dict(qid=f"{r.page_id}#{k}", item_id=r.page_id, row=r.row, keys=json.dumps(sel, ensure_ascii=False),
                             target="\n".join(f"{x}: {tg[x]}" for x in sel), neg=neg))
    return pd.DataFrame(rows)


def drop_rare(R, min_pages=MIN_PAGES):
    kc = pd.Series([k for r in R for k in r[1]]).value_counts()
    keep = set(kc[kc >= min_pages].index)
    return [(t, {k: v for k, v in kv.items() if k in keep}, [e for e in em if e in keep], pr) for t, kv, em, pr in R]


def group_templates(R):
    """서식 번호를 못 찾은 페이지 → 인쇄 항목명 집합(그 서식 페이지 절반 이상에 나온 항목명)과 자카드가 가장 큰 서식(≥ 0.5)."""
    sig = {}
    for t, kv, em, pr in R:
        if t and t.startswith("서식"):
            sig.setdefault(t, {}).update({k: sig.get(t, {}).get(k, 0) + 1 for k in pr})
    tsig = {t: {k for k, n in d.items() if n >= 0.5 * max(d.values())} for t, d in sig.items()}
    out = []
    for t, kv, em, pr in R:
        if not (t and t.startswith("서식")) and tsig:
            ps = set(pr)
            best = max(tsig, key=lambda u: len(ps & tsig[u]) / max(1, len(ps | tsig[u])))
            j = len(ps & tsig[best]) / max(1, len(ps | tsig[best]))
            t = best if j >= 0.5 else "기타:" + (t or "?")
        out.append((t, kv, em, pr))
    return out


def assemble(O, R):
    """OCR 표 + 짝 결과 → 분할표(짝 1개 이상 페이지). -> (분할표, 분리 서식 집합)"""
    O = O.copy()
    O["tpl"] = [r[0] for r in R]
    O["kv"] = [json.dumps(r[1], ensure_ascii=False) for r in R]
    O["n_kv"] = [len(r[1]) for r in R]
    O["empty"] = [json.dumps(r[2], ensure_ascii=False) for r in R]
    O["present"] = [json.dumps(r[3], ensure_ascii=False) for r in R]
    O = O[O.n_kv > 0].reset_index(drop=True)
    O["subset"] = ["test" if f == "Validation" else case2(p) for f, p in zip(O.folder, O.page_id)]
    tpls = sorted(t for t in set(O.tpl) if t.startswith("서식"))
    hold = set(np.random.default_rng(7).choice(tpls, max(1, round(0.15 * len(tpls))), replace=False))
    O.loc[O.tpl.isin(hold), "subset"] = "test_tpl"
    O["row"] = np.arange(len(O))
    return O, hold, tpls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--procs", type=int, default=16, help="짝 만들기 프로세스 수")
    a = ap.parse_args(argv)
    O = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(cfg.ROOT / "cache/xkg/ocr_*.parquet")))], ignore_index=True)
    if a.limit:
        O = O.head(a.limit)
    with ProcessPoolExecutor(a.procs) as ex:
        R = list(ex.map(pairs_cell, zip(O.img, O.boxes), chunksize=32))
    R = group_templates(drop_rare(R))
    O, hold, tpls = assemble(O, R)
    print("페이지", len(O), "세트", O.subset.value_counts().to_dict(), "서식", len(tpls), "분리 서식", sorted(hold), flush=True)
    kc = pd.Series([k for v in O.kv for k in json.loads(v)]).value_counts()
    vocab = sorted(kc[kc >= MIN_PAGES].index)
    print("키 종류", len(kc), "상위", kc.head(25).to_dict(), flush=True)
    if not a.no_images:
        d = cfg.ROOT / "crops/xk"
        d.mkdir(parents=True, exist_ok=True)
        off, tmp = [0], d / "gpage.blob.tmp"
        with open(tmp, "wb") as f, ProcessPoolExecutor(8) as ex:
            for k, b in enumerate(ex.map(_jpg, O.img, chunksize=32)):
                f.write(b)
                off.append(off[-1] + len(b))
                if k % 5000 == 0:
                    print(f"gpage: {k}/{len(O)}", flush=True)
        np.save(d / "gpage.blobidx.npy", np.array(off, np.int64))
        os.replace(tmp, d / "gpage.blob")
    OUT.mkdir(parents=True, exist_ok=True)
    O.drop(columns=["boxes"]).to_parquet(OUT / "xkg_items.parquet", index=False)
    json.dump(vocab, open(OUT / "xkg_vocab.json", "w"), ensure_ascii=False)
    for s in SUBSETS:
        Q = queries(O[O.subset == s], 1, vocab)              # 학습 예산 때문에 페이지당 질의 1개(페이지가 많아 다양성은 충분)
        Q.to_parquet(OUT / f"xkg_queries_{s}.parquet", index=False)
        print(s, len(Q), flush=True)
    print("DONE xkg_build")


if __name__ == "__main__":
    main()
