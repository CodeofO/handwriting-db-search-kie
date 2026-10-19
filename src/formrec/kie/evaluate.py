"""손글씨 KIE 채점: 질의(키 목록) → 키별 값. 방법별 예측을 같은 질의 · 같은 채점으로.

방법:
  ocr:{tag}  대조군 crop+OCR(검출이 끝났다고 가정) — 키에 해당하는 칼럼 crop(xa 필드 이미지)을 일반 OCR로 읽은 결과
             results/xa/g1/{tag}_{subset}.parquet: 이름 = {pid}_name, 생년월일 = {pid}_birth, 주소 = {pid}_addr{주소 토큰 수},
             시도 · 시군구 = {pid}_addr2 결과의 1 · 2번째 단어.
  vlm:{name} VLM KIE — results/xk/{name}_{subset}.parquet (qid, pred = {키: 값} JSON).
DB 조회(--db): 키별(키마다 DB 값 중 편집 거리 최소) · 레코드(요청 키를 한 레코드로 묶어 편집 거리 합 최소 — 칼럼 간 상호 보정).
  레코드 DB = 실제 사람 전체(닫힌 세계) + 사람마다 키 1개를 한 글자 · 한 토큰만 바꾼 어려운 이웃 6개(시드 0).
자동화(--auto): 확신도 종류 · 임계값을 tune에서 정해(정밀도 99%) test에 적용.
채점: 공백 정규화 완전 일치(생년월일은 숫자만). 지표 = 키별 정확도, 질의 완전 일치, 키 수별 질의 완전 일치.
출력: results/xk/summary_{qset}_{subset}.csv
사용: python -m formrec.kie.evaluate --qset xkp --subset test --methods ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft,vlm:pvlkiep --db --auto
"""
import argparse
import json

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.levels import SEP
from formrec.common.selective import apply_threshold, coverage_at_precision
from formrec.kie.fields import norm, query_ok, target_dict

ROOT = cfg.ROOT
RESULTS = ROOT / "results/xk"
COLS = ("이름", "생년월일", "주소", "시도", "시군구")


def load_queries(qset, sub):
    return pd.read_parquet(ROOT / f"meta/x/{qset}_queries_{sub}.parquet")


def ocr_preds(tag, sub, Q, qset):
    G = pd.read_parquet(ROOT / f"results/xa/g1/{tag}_{sub}.parquet")
    p = dict(zip(G.image_id, G.pred.fillna("")))
    I = pd.read_parquet(ROOT / f"meta/x/{qset}_items.parquet").set_index("item_id")
    out = []
    for r in Q.itertuples():
        v = json.loads(I.loc[r.item_id, "values"])
        n, pid = len(v["주소"].split()), r.item_id
        a2 = p.get(f"{pid}_addr2", "").split()
        m = {"이름": p.get(f"{pid}_name"), "생년월일": p.get(f"{pid}_birth"), "주소": p.get(f"{pid}_addr{n}"),
             "시도": a2[0] if a2 else "", "시군구": a2[1] if len(a2) > 1 else ""}
        out.append({k: m[k] for k in json.loads(r.keys)})
    return out


def method_preds(kind, name, sub, qset):
    """(질의, 예측 목록, VLM 확신도 후보). VLM은 결과가 있는 질의만(원래 순서 유지)."""
    Q = load_queries(qset, sub)
    if kind == "ocr":
        return Q, ocr_preds(name, sub, Q, qset), {}
    V = pd.read_parquet(RESULTS / f"{name}_{sub}.parquet").set_index("qid")
    Q = Q[Q.qid.isin(V.index)].reset_index(drop=True)
    extra = {"VLM 토큰 최저 확률": V.loc[Q.qid, "lp_min"].fillna(-1e9).values, "VLM 토큰 평균 확률": V.loc[Q.qid, "lp_mean"].fillna(-1e9).values}
    return Q, [json.loads(x) for x in V.loc[Q.qid, "pred"]], extra


# ── 키별 DB 조회
_KEY_DB = {}


def key_db_values():
    """키별 DB 값 목록(검색 실험과 같은 AI Hub DB): 이름 = L1, 생년월일 = F2의 2번째 칸, 주소 = 토큰 수 k별 L{k}, 시도 · 시군구 = L2의 1 · 2번째 토큰."""
    if not _KEY_DB:
        db = lambda lv: pd.read_parquet(ROOT / f"meta/kie_db_{lv}.parquet").text   # noqa: E731
        _KEY_DB["이름"] = sorted(set(db("L1")))
        _KEY_DB["생년월일"] = sorted({t.split(SEP)[1] for t in db("F2")})
        L2 = db("L2")
        _KEY_DB["시도"] = sorted({t.split()[0] for t in L2})
        _KEY_DB["시군구"] = sorted({t.split()[1] for t in L2 if len(t.split()) > 1})
        for k in range(2, 7):
            _KEY_DB[f"주소{k}"] = sorted(set(db(f"L{k}")))
    return _KEY_DB


def db_fix_keys(preds):
    """예측값을 그 키의 DB 값 중 편집 거리 최소값으로 바꾼다. 주소는 예측 토큰 수(2~6)의 DB에서."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    V = key_db_values()
    out = []
    for pr in preds:
        q = {}
        for k, v in pr.items():
            v = " ".join(str(v or "").split())
            cand = V[f"주소{min(6, max(2, len(v.split())))}"] if k == "주소" else V[k]
            m = process.extractOne(v, cand, scorer=Levenshtein.distance) if v else None
            q[k] = m[0] if m else v
        out.append(q)
    return out


# ── 레코드 DB 조회
_RECORD_DB = {}


def record_db(n_nb=6, seed=0, hard=True):
    """레코드 DB: 실제 사람 전체(AI Hub 파싱 레코드, 모든 세트 — 닫힌 세계) + 사람마다 이웃 n_nb개. src = 그 레코드를 만든 사람(DB 밖 실험용).
    hard: 이웃 = 키 1개를 한 글자만 바꾼 값(이름 = 음절 1개, 생년월일 = 숫자 1개, 주소 = 토큰 1개를 같은 자리의 다른 실제 토큰으로) —
    검색 실험 DB의 '토큰 1개만 다른 이웃'과 같은 난이도. hard=False: 다른 사람의 값으로 칼럼 1개 교체."""
    key = (n_nb, seed, hard)
    if key in _RECORD_DB:
        return _RECORD_DB[key]
    I = pd.read_parquet(ROOT / "meta/x/xk_items.parquet")
    real = pd.DataFrame([json.loads(v) for v in I["values"]]).fillna("")
    real["src"] = I.item_id.values
    rng = np.random.default_rng(seed)
    rows = [real]
    syl = sorted({c for t in real["이름"] for c in t})
    toks = {}
    for t in real["주소"]:
        for j, w in enumerate(t.split()):
            toks.setdefault(j, set()).add(w)
    toks = {j: sorted(v) for j, v in toks.items()}

    def perturb(c, v):
        if c == "이름" and v:
            i = rng.integers(len(v))
            return v[:i] + syl[rng.integers(len(syl))] + v[i + 1:]
        if c == "생년월일" and v:
            i = rng.integers(len(v))
            return v[:i] + str(rng.integers(10)) + v[i + 1:]
        w = v.split()
        if not w:
            return v
        i = int(rng.integers(len(w)))
        w[i] = toks[i][rng.integers(len(toks[i]))]
        return " ".join(w)

    for _ in range(n_nb):
        nb = real.copy()
        col = rng.choice(["이름", "생년월일", "주소"], len(nb))
        for c in ("이름", "생년월일", "주소"):
            m = np.where(col == c)[0]
            if hard:
                nb.loc[m, c] = [perturb(c, real.at[i, c]) for i in m]
            else:
                nb.loc[m, c] = real[c].values[rng.integers(len(real), size=len(m))]
        a = nb["주소"].str.split()
        nb["시도"] = a.str[0].fillna("")
        nb["시군구"] = a.str[1].fillna("")
        rows.append(nb)
    D = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["이름", "생년월일", "주소"]).reset_index(drop=True)
    print(f"레코드 DB {len(D):,} (실제 {len(real):,}, 이웃 {'한 글자 · 한 토큰' if hard else '칼럼 교체'})", flush=True)
    _RECORD_DB[key] = D
    return D


def db_columns(D):
    return {c: D[c].astype(str).tolist() for c in COLS}


def distance_sum(texts_by_key, cols):
    """키별 질의 문자열 목록 → DB 레코드마다 편집 거리 합 (질의 수 × 레코드 수, int32)."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    tot = None
    for k, q in texts_by_key:
        d = cdist(q, cols[k], scorer=Levenshtein.distance, dtype=np.int32, workers=8)
        tot = d if tot is None else tot + d
    return tot


def db_fix_record(preds, db=None, chunk=64):
    """예측의 키들을 한 레코드로 묶어 레코드 DB에서 편집 거리 합이 최소인 레코드의 값으로 바꾼다.
    -> (바꾼 예측 목록, conf = (n, 2) 배열 [1등 거리, 2등 거리])."""
    cols = db_columns(record_db() if db is None else db)
    out, conf = [None] * len(preds), np.zeros((len(preds), 2), np.int64)
    groups = {}
    for i, pr in enumerate(preds):
        groups.setdefault(tuple(pr.keys()), []).append(i)
    for keys, idx in groups.items():
        for j in range(0, len(idx), chunk):
            ii = idx[j:j + chunk]
            tot = distance_sum([(k, [" ".join(str(preds[i].get(k) or "").split()) for i in ii]) for k in keys], cols)
            best = tot.argmin(1)
            two = np.partition(tot, 1, axis=1)[:, :2]
            for i, b, (d1, d2) in zip(ii, best, two):
                out[i] = {k: cols[k][b] for k in keys}
                conf[i] = (int(d1), int(d2))
    return out, conf


# ── 채점
def score(Q, preds):
    rows = []
    for r, pr in zip(Q.itertuples(), preds):
        keys, tg = json.loads(r.keys), target_dict(r.target)
        ok = {k: pr.get(k) is not None and norm(k, pr.get(k, "")) == norm(k, tg[k]) for k in keys}
        rows.append(dict(qid=r.qid, n_keys=len(keys), ok=ok, all_ok=all(ok.values())))
    R = pd.DataFrame(rows)
    per = {}
    for o in R.ok:
        for k, v in o.items():
            per.setdefault(k, []).append(v)
    return dict(n=len(R), query_em=float(R.all_ok.mean()), **{f"key_{k}": float(np.mean(v)) for k, v in per.items()},
                **{f"em_{n}keys": float(R[R.n_keys == n].all_ok.mean()) for n in sorted(R.n_keys.unique())})


def queries_ok(Q, preds):
    return np.array([query_ok(json.loads(r.keys), r.target, pr) for r, pr in zip(Q.itertuples(), preds)])


def automation(kind, name, qset, target=0.99):
    """레코드 DB 조회 결과의 자동화율. 확신도 후보 = DB 1 · 2등 거리 차, −1등 거리, (VLM) 토큰 확률 — tune에서 고른다."""
    R = {}
    for sub in ("tune", "test"):
        Q, P, extra = method_preds(kind, name, sub, qset)
        F, d = db_fix_record(P)
        cands = {"DB 1·2등 거리 차": (d[:, 1] - d[:, 0]).astype(float), "−DB 1등 거리": -d[:, 0].astype(float), **extra}
        R[sub] = (queries_ok(Q, F), cands)
    ok_t, C_t = R["tune"]
    best = None
    for c, v in C_t.items():
        cov, thr = coverage_at_precision(ok_t, v, (target,))[target]
        if best is None or cov > best[1]:
            best = (c, cov, thr)
    ok_s, C_s = R["test"]
    r = apply_threshold(ok_s, C_s[best[0]], best[2])
    return dict(conf=best[0], tune_cov=best[1], test_cov=r["cov"], test_prec=r["prec"], test_em=float(ok_s.mean()))


def label(kind, name):
    return f"{'crop+OCR' if kind == 'ocr' else 'VLM KIE'} · {name}"


def run(qset, subset, methods, db=False, auto=False):
    res = []
    for m in methods:
        kind, name = m.split(":")
        q, P, _ = method_preds(kind, name, subset, qset)
        lab = label(kind, name)
        res.append(dict(method=lab, **score(q, P)))
        if db:
            res.append(dict(method=lab + " + DB 조회(키별)", **score(q, db_fix_keys(P))))
            res.append(dict(method=lab + " + DB 조회(레코드)", **score(q, db_fix_record(P)[0])))
    if auto:
        for m in methods:
            kind, name = m.split(":")
            if kind == "vlm" and not (RESULTS / f"{name}_tune.parquet").exists():   # tune 생성이 없는 방법(데이터량 곡선)은 생략
                continue
            res.append(dict(method=label(kind, name) + " + DB 조회(레코드) · 자동화", **automation(kind, name, qset)))
    return pd.DataFrame(res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="test")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--qset", default="xk")
    a = ap.parse_args(argv)
    R = run(a.qset, a.subset, a.methods.split(","), a.db, a.auto)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(R.round(4).to_string(index=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / f"summary_{a.qset}_{a.subset}.csv", index=False)


if __name__ == "__main__":
    main()
