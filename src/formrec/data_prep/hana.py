"""HANA 덴마크 이름 줄: 라벨(zip 안 CSV) → 칼럼 표 meta/hana_cols.parquet + 고유 이름 표 meta/hana_db.parquet.

라벨 = 소문자 "이름 (중간이름…) 성". 성 = 마지막 단어, 이름 = 첫 단어, 중간이름 = 나머지(없으면 "").
이전 분할(subset): 공식 train의 5%(image_id md5) → tune, 공식 test → test — 최종 분할(xh)은 build.split_xh가 다시 정하고
여기서는 공식 test 여부만 쓴다. 행 순서 = 세트(train · tune · test) 안 image_id 순 — xh 분할표 · blob 행 순서가 이 순서를 따른다.
단계: H1 · H2 · H3 · H4 · H5+ (단어 수). zip은 CSV 두 개만 읽는다(압축 해제 파일 없음).
사용: python -m formrec.data_prep.hana
"""
import hashlib
import io
import zipfile

import pandas as pd

from formrec.common import workspace as cfg

ZIP = cfg.ROOT / "data/hana/archive.zip"
SUBSETS = ("train", "tune", "test")


def labels(zip_path=ZIP):
    """(image_id, name, subset) — 이전 분할, 세트 · image_id 정렬."""
    z = zipfile.ZipFile(zip_path)
    tr = pd.read_csv(io.BytesIO(z.read("HANA/labels/train/HANA.csv"))); te = pd.read_csv(io.BytesIO(z.read("HANA/labels/test/HANA.csv")))
    tr["subset"] = ["tune" if int(hashlib.md5(i.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.05 else "train" for i in tr.image_id]
    te["subset"] = "test"
    d = pd.concat([tr, te])
    d["name"] = d.name.astype(str).str.strip().str.lower()
    return d.sort_values(["subset", "image_id"]).reset_index(drop=True)


def split_name(n):
    w = n.split()
    return w[-1], w[0], " ".join(w[1:-1])


def columns(d):
    rows = []
    for s in SUBSETS:
        q = d[d.subset == s][["image_id", "name"]].reset_index(drop=True)
        q["subset"] = s; q["row"] = q.index
        rows.append(q)
    D = pd.concat(rows, ignore_index=True)
    D[["last", "first", "middle"]] = pd.DataFrame([split_name(n) for n in D.name], index=D.index)
    D["n_words"] = D.name.str.split().str.len()
    D["level"] = D.n_words.clip(upper=5).map({1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "H5+"})
    C = D[["subset", "row", "image_id", "name", "last", "first", "middle", "n_words", "level"]]
    DB = D.drop_duplicates("name")[["name", "last", "first", "middle"]].sort_values("name").reset_index(drop=True)
    return C, DB


def main():
    C, DB = columns(labels())
    cfg.atomic_write(cfg.META / "hana_cols.parquet", lambda t: C.to_parquet(t, index=False))
    cfg.atomic_write(cfg.META / "hana_db.parquet", lambda t: DB.to_parquet(t, index=False))
    print(C.groupby(["level", "subset"]).size().unstack().to_string())
    print("DB", len(DB), "| 칼럼 고유값 성", DB["last"].nunique(), "이름", DB["first"].nunique(), "중간", DB.middle.nunique())
    print("DONE hana")


if __name__ == "__main__":
    main()
