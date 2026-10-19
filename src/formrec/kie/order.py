"""키 순서 일관성 · 결합 오류 (KIE VLM).

make : test 질의 중 키 2개 이상인 것의 키 순서를 바꾼 질의 세트 meta/x/{qset}perm_queries_test.parquet (qid 뒤에 '~p')
score: 원래 순서 결과(results/xk/{out}_test) vs 순서를 바꾼 결과(results/xk/{out}perm_test)
  순서 일관성 = 같은 이미지 · 같은 키 집합에서 키별 값이 모두 같은 비율
  결합 오류 = 키 k의 출력값이 그 질의의 다른 키 j의 정답과 같은(값이 다른 키에 붙은) 키 비율
  순서 바꿈 전 · 후 질의 완전 일치
사용: python -m formrec.kie.order make --qset xkp
      python -m formrec.kie.order score --qset xkp --out pvlkiep
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.kie.fields import norm, target_dict

ROOT = cfg.ROOT
RESULTS = ROOT / "results/xk"


def permuted_queries(Q):
    """키 순서를 바꾼 질의(원래 순서와 반드시 다르게). 시드 = md5(qid) — 실행마다 같은 세트."""
    rows = []
    for r in Q.itertuples():
        keys = json.loads(r.keys)
        if len(keys) < 2:
            continue
        rng = np.random.default_rng(int(hashlib.md5(f"perm:{r.qid}".encode()).hexdigest()[:8], 16))
        p = keys[:]
        while p == keys:
            p = list(rng.permutation(keys))
        tg = target_dict(r.target)
        rows.append(dict(qid=r.qid + "~p", item_id=r.item_id, row=r.row, keys=json.dumps(p, ensure_ascii=False), target="\n".join(f"{k}: {tg[k]}" for k in p)))
    return pd.DataFrame(rows)


def make(a):
    P = permuted_queries(pd.read_parquet(ROOT / f"meta/x/{a.qset}_queries_test.parquet"))
    P.to_parquet(ROOT / f"meta/x/{a.qset}perm_queries_test.parquet", index=False)
    print("DONE xk_order make", len(P))


def order_metrics(A, B):
    """A = 원래 순서 결과(qid 색인), B = 순서 바꾼 결과."""
    same, bind, nb, em_a, em_b = [], 0, 0, [], []
    for r in B.itertuples():
        q = r.qid[:-2]
        if q not in A.index:
            continue
        keys, tg = json.loads(r.keys), target_dict(r.target)
        pa, pb = json.loads(A.at[q, "pred"]), json.loads(r.pred)
        same.append(all(norm(k, pa.get(k)) == norm(k, pb.get(k)) for k in keys))
        em_a.append(all(norm(k, pa.get(k)) == norm(k, tg[k]) for k in keys))
        em_b.append(all(norm(k, pb.get(k)) == norm(k, tg[k]) for k in keys))
        for pr in (pa, pb):
            for k in keys:
                nb += 1
                v = norm(k, pr.get(k))
                if v != norm(k, tg[k]) and any(v == norm(k, tg[j]) and norm(k, tg[j]) != norm(k, tg[k]) for j in keys if j != k):
                    bind += 1
    return dict(n=len(same), order_consistency=float(np.mean(same)), binding_error=bind / max(1, nb),
                em_original=float(np.mean(em_a)), em_permuted=float(np.mean(em_b)))


def score(a):
    A = pd.read_parquet(RESULTS / f"{a.out}_test.parquet").set_index("qid")
    B = pd.read_parquet(RESULTS / f"{a.out}perm_test.parquet")
    out = order_metrics(A, B)
    print(json.dumps(out, ensure_ascii=False))
    json.dump(out, open(RESULTS / f"order_{a.out}.json", "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--qset", default="xkp")
    ap.add_argument("--out", default="pvlkiep")
    a = ap.parse_args(argv)
    {"make": make, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
