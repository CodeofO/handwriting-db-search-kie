"""DB 커버리지(DB 밖 사람): 일부 사람을 레코드 DB에서 통째로 빼고(실제 레코드 + 그 사람에게서 만든 이웃) 레코드 DB 조회 + 되돌림 규칙을 잰다.

규칙: DB 1등 거리 d1 ≤ τ면 DB 레코드 값, 아니면 VLM(또는 OCR) 출력 그대로. τ(0~20)는 같은 빠짐 비율의 tune에서 질의 완전 일치가 최대가 되게.
빠짐 비율 p = 0 · 5 · 20 %: 사람 단위(시드 1)로 실제 레코드를 DB에서 뺀다 — 그 사람의 질의는 DB 밖.
출력: results/xk/oov_{qset}.csv (방법 · p · τ · 전체 / DB 안 / DB 밖 질의 완전 일치 · 원출력 · DB만)
사용: python -m formrec.kie.oov --qset xkp --methods ocr:pvlft,ocr:ppocrv5ft,vlm:pvlkiep
"""
import argparse

import numpy as np
import pandas as pd

from formrec.kie import evaluate as E

DROPS = (0.0, 0.05, 0.2)


def dropped_people(qset, sub, p, seed=1):
    """빠짐 비율 p만큼 그 세트 질의의 사람(item_id)을 고정 시드로 고른다(우도 기반 검증 판정도 같은 함수를 쓴다)."""
    if not p:
        return set()
    ids = np.array(sorted(E.load_queries(qset, sub).item_id.unique()))
    return set(np.random.default_rng(seed).choice(ids, int(round(len(ids) * p)), replace=False))


def evaluate_drop(kind, name, sub, qset, drop, base):
    """→ (d1, DB 조회 정답, 원출력 정답, DB 밖 여부) — 질의별."""
    Q, P, _ = E.method_preds(kind, name, sub, qset)
    D = base[~base.src.isin(drop)].reset_index(drop=True) if drop else base   # 빠진 사람에게서 나온 레코드를 모두 지움
    F, conf = E.db_fix_record(P, db=D)
    return conf[:, 0], E.queries_ok(Q, F), E.queries_ok(Q, P), Q.item_id.isin(drop).values


def best_tau(d1, ok_db, ok_raw):
    return max(range(0, 21), key=lambda t: (np.where(d1 <= t, ok_db, ok_raw).mean(), -t))


def run(qset, methods):
    base = E.record_db()
    rows = []
    for m in methods:
        kind, name = m.split(":")
        for p in DROPS:
            res = {sub: evaluate_drop(kind, name, sub, qset, dropped_people(qset, sub, p), base) for sub in ("tune", "test")}
            tau = best_tau(*res["tune"][:3])
            d1, ok_db, ok_raw, oov = res["test"]
            ok = np.where(d1 <= tau, ok_db, ok_raw)
            rows.append(dict(method=m, drop=p, tau=tau, em_all=ok.mean(), em_in=ok[~oov].mean(), em_out=ok[oov].mean() if oov.any() else np.nan,
                             em_raw=ok_raw.mean(), em_db_only=ok_db.mean()))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qset", default="xkp")
    ap.add_argument("--methods", required=True)
    a = ap.parse_args(argv)
    R = run(a.qset, a.methods.split(","))
    E.RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(E.RESULTS / f"oov_{a.qset}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
