"""검색 DB(xh · xi): 제안 모델 평가가 레코드를 찾는 후보 문자열 집합.

  xh : HANA 전체 이름 줄의 고유 문자열(이미지 순서 성 · 이름 · 중간이름, 약 70만, 실제 값만) — meta/x/xh_db.parquet (text)
  xi : IAM 전체 줄의 고유 문자열 + 줄마다 단어 1개만 바꾼 이웃 15개(시드 7) → 약 20만 — meta/x/xi_db.parquet (text, real)
  xa · xc는 AI Hub 단계 DB(meta/kie_db_*, formrec.data_prep.kie_db)를 쓴다.
이미 있으면 다시 만들지 않는다(평가 시작 때 불러도 된다).
사용: python -m formrec.data_prep.search_db
"""
import numpy as np
import pandas as pd

from formrec.common import workspace as cfg

OUT = cfg.META / "x"
N_NEIGHBORS = 15


def xh_db(xh_items):
    return pd.DataFrame(dict(text=np.unique(xh_items.text.values)))


def iam_real_lines(lines_txt):
    lines = [l.rstrip("\n").split(" ", 8) for l in open(lines_txt) if not l.startswith("#")]
    return sorted({l[8].replace("|", " ").strip() for l in lines})


def xi_db(real, seed=7, k=N_NEIGHBORS):
    """실제 줄 + 줄마다 단어 1개만 바꾼 이웃 k개(한국어 DB의 토큰 1개 이웃과 같은 종류)."""
    words = sorted({w for t in real for w in t.split(" ")})
    rng = np.random.default_rng(seed); db = set(real)
    for t in real:
        w = t.split(" ")
        for _ in range(k):
            i = rng.integers(len(w)); u = words[rng.integers(len(words))]
            if u != w[i]:
                db.add(" ".join(w[:i] + [u] + w[i + 1:]))
    t = np.array(sorted(db))
    return pd.DataFrame(dict(text=t, real=np.isin(t, real)))


def build_dbs():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "xh_db.parquet").exists():
        D = xh_db(pd.read_parquet(OUT / "xh_items.parquet"))
        D.to_parquet(OUT / "xh_db.parquet", index=False); print("xh DB", len(D))
    if not (OUT / "xi_db.parquet").exists():
        real = iam_real_lines(cfg.ROOT / "data/iam/raw/lines.txt")
        D = xi_db(real)
        D.to_parquet(OUT / "xi_db.parquet", index=False); print("xi DB", len(D), "실제", len(real))


if __name__ == "__main__":
    build_dbs()
