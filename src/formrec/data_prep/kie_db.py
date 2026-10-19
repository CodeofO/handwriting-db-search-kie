"""AI Hub 서식 3단계: 단계별 검색 DB(크기 N 고정)와 단계 질의 표(meta/kie_levels.parquet)를 만든다.

단계: L1 이름 · L2~L6 주소 k토큰 · F2 이름␞생년월일 · F3 이름␞생년월일␞주소 4토큰 (␞ = 필드 구분 기호)
DB = 실제 값(모든 세트의 고유값) ∪ 합성 방해 항목, 전체 N(기본 200,000). 방해 항목의 절반 = 실제 값에서 토큰 1개만 바꾼 이웃,
나머지 = 같은 값 목록에서 무작위 재조합. 조합 공간이 작은 단계(주소 2토큰 등)는 N = 공간의 60%로 줄인다.
  이름   : 무작위 = 실제 한국어 이름 목록(meta/names_v1.parquet)에서, 이웃 = 음절 1개를 음절 풀의 다른 음절로
  생년월일: 무작위 = 올바른 YYMMDD, 이웃 = 숫자 1개 변경
  주소   : 서식(TO/PO)별 · 자리별 토큰 목록. 무작위 = 자리마다 독립 추출, 이웃 = 한 자리만 다른 토큰으로
  F2 · F3: 이웃 = 필드 하나만 그 필드의 이웃 값으로, 무작위 = 필드마다 무작위
난수 순서가 결과를 정한다(시드 0) — 호출 순서를 바꾸지 말 것.
출력: meta/kie_db_{단계}{suffix}.parquet (text, real), meta/kie_levels.parquet (level, subset, pid, items, text, kind, official_test)
사용: python -m formrec.data_prep.kie_db                                                  (기본 8단계 + 단계 표)
      python -m formrec.data_prep.kie_db --levels F3 --N 2000000 --suffix _2M --no_levels_file   (서브 실험 DB 규모 200만)
"""
import argparse

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.levels import SEP

LEVELS = {"L1": ("name",), "L2": ("addr2",), "L3": ("addr3",), "L4": ("addr4",), "L5": ("addr5",), "L6": ("addr6",),
          "F2": ("name", "birth"), "F3": ("name", "birth", "addr4")}
SUBSETS = ("train", "tune", "test")


def load_persons():
    """crop이 있는 사람(원본 PNG 손상 페이지 제외) + addr2..addr6 문자열."""
    P = pd.read_parquet(cfg.META / "kie_persons.parquet")
    have = set(pd.concat([pd.read_parquet(cfg.ROOT / f"cache/kie/{x}_index.parquet", columns=["pid"]) for x in SUBSETS]).pid)
    P = P[P.pid.isin(have)].reset_index(drop=True)
    for k in range(2, 7):
        P[f"addr{k}"] = [" ".join(r[f"a{j}"] for j in range(1, k + 1)) if r.n_addr >= k else "" for _, r in P.iterrows()]
    return P


def build(P, v1, N=200_000, seed=0, levels=""):
    """→ ({단계: DB 표}, 단계 질의 표). v1 = 실제 이름 목록(names_v1.text.unique(), 파일 순서)."""
    rng = np.random.default_rng(seed)
    syl = np.array(sorted({c for s in list(P.name) + list(v1[:50000]) for c in s}))
    vocab = {kind: [sorted(set(P[P.kind == kind][f"a{j}"]) - {""}) for j in range(1, 7)] for kind in ("TO", "PO")}

    def rnd_name():
        return v1[rng.integers(len(v1))]

    def nb_name(s):
        i = rng.integers(len(s)); c = syl[rng.integers(len(syl))]
        return s[:i] + c + s[i + 1:] if c != s[i] else None

    def rnd_birth():
        return f"{rng.integers(100):02d}{rng.integers(1, 13):02d}{rng.integers(1, 29):02d}"

    def nb_birth(s):
        i = rng.integers(6); d = str(rng.integers(10))
        return s[:i] + d + s[i + 1:] if d != s[i] else None

    def rnd_addr(k, kind):
        return " ".join(vocab[kind][j][rng.integers(len(vocab[kind][j]))] for j in range(k))

    def nb_addr(s, kind):
        t = s.split(" "); i = rng.integers(len(t)); v = vocab[kind][i]; u = v[rng.integers(len(v))]
        return " ".join(t[:i] + [u] + t[i + 1:]) if u != t[i] else None

    def field_rnd(f, kind):
        return rnd_name() if f == "name" else rnd_birth() if f == "birth" else rnd_addr(int(f[4:]), kind)

    def field_nb(f, v, kind):
        return nb_name(v) if f == "name" else nb_birth(v) if f == "birth" else nb_addr(v, kind)

    dbs, lv_rows = {}, []
    for lv, fields in LEVELS.items():
        if levels and lv not in levels.split(","):
            continue
        Q = P[(P[list(fields)] != "").all(axis=1)] if any(f.startswith("addr") for f in fields) else P
        Q = Q[(Q[list(fields)] != "").all(axis=1)]
        text = Q[list(fields)].agg(SEP.join, axis=1)
        real_set = set(text.values)
        kinds = Q.kind.values; tuples = Q[list(fields)].values
        if len(fields) == 1 and fields[0].startswith("addr"):    # 주소만 있는 단계: 가능한 조합 수를 넘지 않게
            k = int(fields[0][4:])
            space = sum(int(np.prod([len(vocab[kd][j]) for j in range(k)])) for kd in set(kinds) if k <= (6 if kd == "TO" else 4))
            n_target = min(N, int(space * 0.6))
        else:
            space, n_target = None, N
        need = max(0, n_target - len(real_set))
        extra = set()
        n_nb = need // 2
        tries = 0
        while len(extra) < n_nb and tries < 50 * max(n_nb, 1):
            tries += 1                        # 이웃: 실제 레코드 하나를 골라 필드 하나만 이웃 값으로
            i = rng.integers(len(tuples)); t = list(tuples[i]); j = rng.integers(len(fields))
            v = field_nb(fields[j], t[j], kinds[i])
            if v is None:
                continue
            t[j] = v; s = SEP.join(t)
            if s not in real_set:
                extra.add(s)
        kr = pd.Series(kinds).value_counts(normalize=True)
        tries = 0
        while len(extra) < need and tries < 50 * max(need, 1):   # 무작위 재조합 (서식 비율 유지)
            tries += 1
            kind = rng.choice(kr.index.values, p=kr.values)
            s = SEP.join(field_rnd(f, kind) for f in fields)
            if s not in real_set:
                extra.add(s)
        dbs[lv] = pd.DataFrame({"text": sorted(real_set) + sorted(extra), "real": [True] * len(real_set) + [False] * len(extra)})
        print(f"{lv}: 질의 {len(Q):,}명 · DB {len(dbs[lv]):,} (실제 {len(real_set):,}, 합성 {len(extra):,}, 조합 공간 {space if space else '충분'})", flush=True)
        items = [[f"{pid}_{f}" for f in fields] for pid in Q.pid]
        lv_rows.append(pd.DataFrame(dict(level=lv, subset=Q.subset.values, pid=Q.pid.values, items=items, text=text.values,
                                         kind=Q.kind.values, official_test=Q.official_test.values)))
    return dbs, pd.concat(lv_rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", default="", help="일부 단계만 (예: --levels F3 --N 2000000 --suffix _2M --no_levels_file)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no_levels_file", action="store_true", help="meta/kie_levels.parquet를 다시 쓰지 않음")
    a = ap.parse_args(argv)
    v1 = pd.read_parquet(cfg.META / "names_v1.parquet").text.unique()
    dbs, L = build(load_persons(), v1, a.N, a.seed, a.levels)
    for lv, DB in dbs.items():
        cfg.atomic_write(cfg.META / f"kie_db_{lv}{a.suffix}.parquet", lambda t_, DB=DB: DB.to_parquet(t_, index=False))
    if a.no_levels_file:
        print("DONE kie_db (단계 파일 유지)", a.suffix); return
    cfg.atomic_write(cfg.META / "kie_levels.parquet", lambda t_: L.to_parquet(t_, index=False))
    print(L.groupby(["level", "subset"]).size().unstack().to_string())
    print("DONE kie_db")


if __name__ == "__main__":
    main()
