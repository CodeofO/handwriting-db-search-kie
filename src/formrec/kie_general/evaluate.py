"""일반 KIE(xkg) 채점: 질의 완전 일치 · 키 정확도(양성 키) · 네거티브 정답률('[없음]'을 맞힘, A = 서식에 없는 키 / B = 빈 칸) ·
놓침률(값이 있는데 '[없음]') — test(같은 서식) · test_tpl(처음 보는 서식) 따로.
입력: results/xk/{out}_{test,test_tpl}.parquet (formrec.kie.vlm infer --qset xkg 결과). 출력: results/xk/xkg_eval_{out}.csv
사용: python -m formrec.kie_general.evaluate --out ovkieg
"""
import argparse
import json

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.kie.fields import target_dict
from formrec.kie_general.build import NONE

ROOT = cfg.ROOT


def norm(v):
    return " ".join(str(v or "").split())


def subset_metrics(sub, V, Q):
    qem, pos, negA, negB, miss = [], [], [], [], []
    for r in V.itertuples():
        keys, tg, pr = json.loads(r.keys), target_dict(r.target), json.loads(r.pred)
        neg = Q.at[r.qid, "neg"] if r.qid in Q.index else ""
        oks = []
        for k in keys:
            ok = norm(pr.get(k)) == norm(tg[k])
            oks.append(ok)
            if tg[k] == NONE:
                (negB if neg == "B" else negA).append(ok)
            else:
                pos.append(ok)
                miss.append(norm(pr.get(k)) == NONE)
        qem.append(all(oks))
    return dict(subset=sub, n=len(V), query_em=np.mean(qem), key_acc_pos=np.mean(pos), neg_A_acc=np.mean(negA) if negA else np.nan, n_negA=len(negA),
                neg_B_acc=np.mean(negB) if negB else np.nan, n_negB=len(negB), miss_rate=np.mean(miss))


def run(out):
    rows = []
    for sub in ("test", "test_tpl"):
        p = ROOT / f"results/xk/{out}_{sub}.parquet"
        if p.exists():
            rows.append(subset_metrics(sub, pd.read_parquet(p), pd.read_parquet(ROOT / f"meta/x/xkg_queries_{sub}.parquet").set_index("qid")))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = run(a.out)
    print(R.round(4).to_string(index=False))
    R.to_csv(ROOT / f"results/xk/xkg_eval_{a.out}.csv", index=False)


if __name__ == "__main__":
    main()
