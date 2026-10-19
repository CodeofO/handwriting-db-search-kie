"""데이터 준비 원본 대조(LEGACY_WORK · FORMREC_WORKSPACE = 원본 산출물이 있는 작업 공간, 읽기 전용, 함수 단위).

분할표 · DB · 어휘는 전체, 원천 파싱(bbox · 사람 블록)은 페이지 표본, 이미지 묶음 · 캐시는 행 표본(20행)으로 비교한다.
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[2] / "src")


def run(code, ws, py=sys.executable):
    e = dict(os.environ, PYTHONPATH=SRC, FORMREC_WORKSPACE=ws, CUDA_VISIBLE_DEVICES="", HW_DATASET="xa")
    r = subprocess.run([py, "-c", code], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def rt(D):
    import pandas as pd
    b = io.BytesIO(); D.to_parquet(b, index=False); b.seek(0); return pd.read_parquet(b)


@pytest.fixture(scope="module")
def ws(legacy_work, workspace_root):
    os.environ.setdefault("HW_DATASET", "xa")
    return Path(workspace_root)


def _eq(a, b, root=None):
    import numpy as np
    import pandas as pd
    a, b = rt(a), b.reset_index(drop=True)
    for c in a.columns:
        if len(a) and isinstance(a[c].iloc[0], np.ndarray):
            a[c], b[c] = a[c].map(list), b[c].map(list)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_split_tables(ws):
    import pandas as pd
    from formrec.data_prep import build
    for ds, fn in (("xa", build.split_xa), ("xh", build.split_xh), ("xi", build.split_xi)):
        _eq(fn(), pd.read_parquet(ws / f"meta/x/{ds}_items.parquet"))
    _eq(build.split_xc()[0], pd.read_parquet(ws / "meta/x/xc_items.parquet"))


def test_names_hana_search_db(ws):
    import pandas as pd
    from formrec.data_prep import hana, names_v1, search_db
    _eq(names_v1.build(pd.read_parquet(ws / "meta/bbox.parquet")), pd.read_parquet(ws / "meta/names_v1.parquet"))
    if (ws / "data/hana/archive.zip").exists():
        C, DB = hana.columns(hana.labels(ws / "data/hana/archive.zip"))
        _eq(C, pd.read_parquet(ws / "meta/hana_cols.parquet")); _eq(DB, pd.read_parquet(ws / "meta/hana_db.parquet"))
    _eq(search_db.xh_db(pd.read_parquet(ws / "meta/x/xh_items.parquet")), pd.read_parquet(ws / "meta/x/xh_db.parquet"))
    _eq(search_db.xi_db(search_db.iam_real_lines(ws / "data/iam/raw/lines.txt")), pd.read_parquet(ws / "meta/x/xi_db.parquet"))


def test_kie_db_first_level(ws):
    """난수 순서상 첫 단계(L1)만 따로 만들어도 전체 실행과 같다."""
    import pandas as pd
    from formrec.data_prep import kie_db
    dbs, L = kie_db.build(kie_db.load_persons(), pd.read_parquet(ws / "meta/names_v1.parquet").text.unique(), levels="L1")
    _eq(dbs["L1"], pd.read_parquet(ws / "meta/kie_db_L1.parquet"))
    ref = pd.read_parquet(ws / "meta/kie_levels.parquet"); ref = ref[ref.level == "L1"]
    _eq(L, ref)


def test_kie_persons_and_bbox_sample(ws):
    import json
    import random
    import pandas as pd
    from formrec.data_prep import aihub_bbox, kie_persons
    P = kie_persons.pages()
    P = P.groupby(["split", "kind"]).head(150)
    rows = [r for a in zip(P.split, P.kind, P.page_id, P.img, P.json) for r in kie_persons.parse(a)[0]]
    new = kie_persons.finalize(pd.DataFrame(rows))
    ref = pd.read_parquet(ws / "meta/kie_persons.parquet"); ref = ref[ref.page_id.isin(set(P.page_id))]
    key = ["page_id", "i_name", "i_birth"]
    _eq(new.sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True))
    J = aihub_bbox.jobs(); random.Random(1).shuffle(J); J = J[:100]
    nb = pd.DataFrame([r for j in J for r in aihub_bbox.parse(j)])
    B = pd.read_parquet(ws / "meta/bbox.parquet"); ids = set(zip(nb.folder, nb.image_id))
    B = B[[(f, i) in ids for f, i in zip(B.folder, B.image_id)]]
    k = ["folder", "cell", "image_id", "order"]
    _eq(nb.sort_values(k)[list(B.columns)].reset_index(drop=True), B.sort_values(k).reset_index(drop=True))


def test_blob_and_cache_sample(ws):
    import zipfile
    import numpy as np
    import pandas as pd
    from formrec.data_prep import build, kie_cache

    def blob_row(p, r):
        off = np.load(p.with_name(p.stem + ".blobidx.npy"))
        with open(p, "rb") as f:
            f.seek(int(off[r])); return f.read(int(off[r + 1] - off[r]))
    rs = range(0, 10000, 500)
    xi = pd.read_parquet(ws / "meta/x/xi_items.parquet")
    assert all(build.iam_png(xi.path[r]) == blob_row(ws / "crops/xi/all.blob", r) for r in rs)
    if (ws / "data/hana/archive.zip").exists():
        xh = pd.read_parquet(ws / "meta/x/xh_items.parquet", columns=["zip_name"]); z = zipfile.ZipFile(ws / "data/hana/archive.zip")
        assert all(build.hana_png(z, xh.zip_name[r]) == blob_row(ws / "crops/xh/all.blob", r) for r in rs)
    D, srcs = build.split_xc(); rd = build.XaReader()
    assert all(build.xc_compose([rd(i) for i in srcs[r]]) == blob_row(ws / "crops/xc/all.blob", r) for r in rs)
    for src, npy, W in (("crops/xi/all.blob", "cache/xi/all_64x1280.npy", 1280), ("crops/xh/all.blob", "cache/xh/all_64x448.npy", 448),
                        ("crops/kie/train.blob", "cache/kie/train_64x1280.npy", 1280)):
        A = np.load(ws / npy, mmap_mode="r")
        assert all(np.array_equal(kie_cache.load((blob_row(ws / src, r), 64, W)), A[r]) for r in rs)


def test_vocab(ws):
    py = os.environ.get("FORMREC_PY_TRAIN", str(ws / "envs/train/bin/python"))
    if not os.path.exists(py):
        pytest.skip("train 환경(torch) 없음")
    out = run("import json\nfrom formrec.data_prep import vocab\nfrom formrec.retrieval.tokenizer import UnitTok3\n"
              "print('SAME', json.dumps(UnitTok3(vocab.vocab_texts()).vocab, ensure_ascii=False) == open(vocab.PATH).read())", str(ws), py)
    assert "SAME True" in out
