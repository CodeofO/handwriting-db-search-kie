"""데이터 준비 단위 테스트 (합성 입력, 작업 공간 불필요)."""
import io

import numpy as np
import pandas as pd
from PIL import Image

from formrec.common.levels import SEP
from formrec.data_prep import build, hana, kie_cache, kie_db, kie_persons, kie_sub_db, names_v1, search_db


def _png(w, h, v=0):
    b = io.BytesIO(); Image.new("L", (w, h), v).save(b, "PNG"); return b.getvalue()


def _box(t, x0, y0=100, w=50, h=30):
    return {"data": t, "x": [x0, x0, x0 + w, x0 + w], "y": [y0, y0 + h, y0, y0 + h]}


def test_kie_cache_load_geometry():
    a = kie_cache.load((_png(100, 32), 64, 1280))                 # 높이 기준 확대 → 200 × 64, 왼쪽 정렬
    assert a.shape == (64, 1280) and (a[:, :200] == 0).all() and (a[:, 200:] == 255).all()
    a = kie_cache.load((_png(4000, 10), 64, 1280))                # 폭 기준 축소 → 1280 × 3, 세로 가운데
    assert (a[:, :] == 255).sum() == 64 * 1280 - 1280 * 3 and (a[30:33] == 0).all()


def test_xc_compose_and_png():
    out = Image.open(io.BytesIO(build.xc_compose([_png(50, 32), _png(30, 64)])))
    assert out.size == (100 + 32 + 30, 64) and out.getpixel((110, 10)) == 255 and out.getpixel((140, 10)) == 0


def test_write_blob_roundtrip(tmp_path):
    items = [_png(5, 5), _png(7, 3, 128)]
    build.write_blob(tmp_path / "all.blob", items, lambda b: b)
    off = np.load(tmp_path / "all.blobidx.npy"); data = (tmp_path / "all.blob").read_bytes()
    assert [data[off[k]:off[k + 1]] for k in range(2)] == items
    build.write_blob_parallel(tmp_path / "p.blob", iter([[items[0]], [items[1], items[0]]]), 3, 1)
    off = np.load(tmp_path / "p.blobidx.npy")
    assert len(off) == 4 and (tmp_path / "p.blob").read_bytes() == items[0] + items[1] + items[0]


def test_split_xi_writer_groups(tmp_path, monkeypatch):
    (tmp_path / "splits").mkdir()
    lines, forms, train = [], [], []
    for w in range(30):                                          # 필기자 30명 × 서식 1장 × 줄 1~4개
        f = f"a{w:02d}-000"; forms.append(f"{f} w{w:03d} 1 1 1 1 1 1"); train.append(f)
        for j in range(1 + w % 4):
            lines.append(f"{f}-{j:02d} ok 180 1 0 0 10 10 A|line{j}")
    (tmp_path / "lines.txt").write_text("#\n" + "\n".join(lines) + "\n")
    (tmp_path / "forms.txt").write_text("\n".join(forms) + "\n")
    (tmp_path / "splits/train.uttlist").write_text("\n".join(train[:25]))
    (tmp_path / "splits/validation.uttlist").write_text("\n".join(train[25:27]))
    (tmp_path / "splits/test.uttlist").write_text("\n".join(train[27:]))
    monkeypatch.setattr(build, "IAM", tmp_path)
    D = build.split_xi()
    assert set(D.subset) == {"train", "tune", "val", "test"} and (D.groupby("group").subset.nunique() == 1).all()
    tr = D[D.subset.isin(["train", "tune"])]
    assert (D.subset == "tune").sum() >= 0.1 * len(tr) and D.text.iloc[0] == "A line0" and D.path.iloc[0].endswith("a00/a00-000/a00-000-00.png")


def test_kie_persons_parse():
    b = [_box("성명", 0), _box("김민준", 60), _box("900101", 120), _box("1234567", 180), _box("주소", 240),
         _box("서울특별시", 300), _box("강남구", 360), _box("역삼로", 420), _box("12", 480), _box("행복아파트", 540), _box("101", 600)]
    P = kie_persons.persons_in(b, "train", "TO", "pg1", "x.png")
    assert len(P) == 1 and P[0]["name"] == "김민준" and P[0]["birth"] == "900101" and P[0]["n_addr"] == 6 and P[0]["a6"] == "101"
    assert kie_persons.addr_at(["서울특별시", "강남구", "역삼동", "삼성동"], 0, "PO") == 4
    D = kie_persons.finalize(pd.DataFrame(P))
    assert D.pid[0] == "pg1_1_2" and D.folder[0] == "Training" and D.subset[0] in ("train", "tune", "test")
    assert kie_persons.kie_subset("Validation", "pg1") == "test"


def test_kie_db_deterministic():
    P = pd.DataFrame(dict(pid=["p1", "p2", "p3"], name=["김민준", "이서연", "박지후"], birth=["900101", "851212", "000303"], kind=["TO", "TO", "PO"],
                          subset=["train", "tune", "test"], official_test=[False, False, True], n_addr=[4, 4, 4]))
    for j, col in enumerate([["서울특별시", "부산광역시", "경기도"], ["강남구", "해운대구", "수원시"], ["역삼로", "중앙로", "매탄동"], ["12", "7", "원천동"]]):
        P[f"a{j + 1}"] = col
    for j in (5, 6):
        P[f"a{j}"] = ""
    for k in range(2, 7):
        P[f"addr{k}"] = [" ".join(r[f"a{j}"] for j in range(1, k + 1)) if r.n_addr >= k else "" for _, r in P.iterrows()]
    v1 = np.array(["최하늘", "정다은", "한결"])
    d1, L1 = kie_db.build(P, v1, N=50, levels="L1,F2")
    d2, L2 = kie_db.build(P, v1, N=50, levels="L1,F2")
    assert set(d1) == {"L1", "F2"} and d1["F2"].equals(d2["F2"]) and L1.equals(L2)
    assert len(d1["F2"]) == 50 and d1["F2"].real.sum() == 3 and all(SEP in t for t in d1["F2"].text)
    assert L1[L1.level == "F2"]["items"].iloc[0] == ["p1_name", "p1_birth"]
    sub, nk = kie_sub_db.build(d1["F2"], L1.assign(level="F3"), n=10)
    assert len(sub) == 10 and nk == 2 and sub.text.is_monotonic_increasing


def test_names_section_and_build():
    assert names_v1.pr_name_section(["1", "漢字", "김민준", "이서연"]) == [2, 3]
    assert names_v1.pr_name_section(["가", "나"]) == []
    pr = ["1", "서울", "한자漢"] + [f"김{chr(0xAC00 + i)}" for i in range(12)]
    df = pd.DataFrame(dict(folder="Training", cell=["PR"] * len(pr) + ["TR"], image_id=["i1"] * len(pr) + ["i2"],
                           order=list(range(len(pr))) + [0], text=pr + ["김가"]))
    n = names_v1.build(df)
    assert len(n) == 11 and "김가" not in set(n.text) and set(n.subset) <= {"train", "tune"}   # TR 어휘와 겹치는 값 제외


def test_hana_columns():
    assert hana.split_name("carl william heinrich dahl") == ("dahl", "carl", "william heinrich")
    d = pd.DataFrame(dict(image_id=["b.jpg", "a.jpg", "c.jpg"], name=["anna hansen", "carl w dahl", "x"], subset=["train", "train", "test"]))
    C, DB = hana.columns(d.sort_values(["subset", "image_id"]).reset_index(drop=True))
    assert C.subset.tolist() == ["train", "train", "test"] and C.row.tolist() == [0, 1, 0]
    assert C.level.tolist() == ["H3", "H2", "H1"] and C["middle"].tolist() == ["w", "", ""] and len(DB) == 3


def test_search_db_xi_neighbors():
    real = ["a b c", "d e"]
    D = search_db.xi_db(real, k=5)
    assert set(real) <= set(D.text) and D.real.sum() == 2 and D.text.is_monotonic_increasing
    for t in D.text[~D.real]:
        assert any(len(t.split()) == len(r.split()) and sum(x != y for x, y in zip(t.split(), r.split())) == 1 for r in real)
    assert search_db.xh_db(pd.DataFrame(dict(text=["b", "a", "b"]))).text.tolist() == ["a", "b"]
