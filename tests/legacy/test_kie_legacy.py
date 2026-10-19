"""손글씨 KIE 원본 대조(LEGACY_WORK = 원본 코드, FORMREC_WORKSPACE = 원본 산출물이 있는 작업 공간 — 읽기만, 쓰기 없음).

- 채점: 저장된 예측(results/xk/*.parquet) · 질의로 새 evaluate.run(--db --auto) vs 원본 summary_{xk,xkp}_test.csv
- 레코드 DB: 새 record_db() vs 원본 xk_eval.record_db() (같은 시드)
- DB 밖: 새 oov.run vs 원본 oov_xkp.csv, 키 순서: 새 order_metrics vs 원본 order_pvlkiep.json
- 우도 기반 검증 후보: 새 verify.candidates vs 원본 results/xk/lrv/cands_xkp_ovkiep_{tune,test}.parquet
- 질의 · 학습 데이터 · 파싱: 새 함수로 다시 만든 질의 · jsonl · yaml · pred vs 저장 파일
채점 대조는 CPU로 수 분씩 걸린다(xkp 약 7분, xk 약 9분).
"""
import json
import sys

import numpy as np
import pandas as pd
import pytest

XK = "ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft,vlm:pvlkie,vlm:pvlkie2"
XKP = "ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft,vlm:pvlkiep,vlm:pvlkiep10,vlm:pvlkiep30,vlm:ovkiep"


@pytest.fixture(scope="module")
def ws(legacy_work, workspace_root):
    from formrec.common import workspace as cfg
    assert str(cfg.ROOT) == str(__import__("pathlib").Path(workspace_root).resolve())
    if not (cfg.ROOT / "results/xk/summary_xkp_test.csv").exists():
        pytest.skip("원본 KIE 산출물 없음")
    return cfg.ROOT


def same_table(new, old, text_cols):
    assert list(new.method) == list(old.method) and set(new.columns) == set(old.columns)
    new = new[old.columns]
    num = [c for c in old.columns if c not in text_cols]
    a, b = new[num].astype(float).values, old[num].astype(float).values
    assert (np.isnan(a) == np.isnan(b)).all()
    assert np.nanmax(np.abs(a - b)) < 1e-12
    for c in text_cols:
        if c in old.columns and c != "method":
            assert (new[c].fillna("") == old[c].fillna("")).all()


@pytest.mark.parametrize("qset,methods", [("xkp", XKP), ("xk", XK)])
def test_summary(ws, qset, methods):
    from formrec.kie import evaluate
    same_table(evaluate.run(qset, "test", methods.split(","), db=True, auto=True), pd.read_csv(ws / f"results/xk/summary_{qset}_test.csv"), ("method", "conf"))


def test_record_db_same_seed(ws, legacy_work):
    from formrec.kie import evaluate
    sys.path.insert(0, legacy_work)
    import xk_eval
    assert evaluate.record_db().equals(xk_eval.record_db())


def test_oov(ws):
    from formrec.kie import oov
    new = oov.run("xkp", ["ocr:pvlft", "ocr:ppocrv5ft", "vlm:pvlkiep"])
    same_table(new, pd.read_csv(ws / "results/xk/oov_xkp.csv"), ("method",))


def test_order(ws):
    from formrec.kie import order
    A = pd.read_parquet(ws / "results/xk/pvlkiep_test.parquet").set_index("qid")
    B = pd.read_parquet(ws / "results/xk/pvlkiepperm_test.parquet")
    assert order.order_metrics(A, B) == json.load(open(ws / "results/xk/order_pvlkiep.json"))


@pytest.mark.parametrize("sub", ["tune", "test"])
def test_verify_cands(ws, sub):
    from formrec.kie import evaluate, verify
    Q, P, _ = evaluate.method_preds("vlm", "ovkiep", sub, "xkp")
    C = verify.candidates(Q, P, evaluate.record_db())
    O = pd.read_parquet(ws / f"results/xk/lrv/cands_xkp_ovkiep_{sub}.parquet")
    assert list(C.columns) == list(O.columns) and len(C) == len(O)
    for c in ("qid", "keys", "free"):
        assert (C[c].values == O[c].values).all()
    assert all(list(a) == list(b) for a, b in zip(C.cand, O.cand)) and all(list(a) == list(b) for a, b in zip(C.cand_d, O.cand_d))
    assert all([list(s) for s in a] == [list(s) for s in b] for a, b in zip(C.cand_src, O.cand_src))


def test_queries_and_items(ws):
    from formrec.kie import data
    for qset in ("xk", "xkp"):
        D = pd.read_parquet(ws / f"meta/x/{qset}_items.parquet")
        n_tr, n_ot = data.N_PER[qset]
        for s in data.SUBSETS:
            Q = data.queries(D[(D.subset == s) & D.clean], n_tr if s == "train" else n_ot)
            assert Q.equals(pd.read_parquet(ws / f"meta/x/{qset}_queries_{s}.parquet")), (qset, s)
    P = data.persons()
    Pp, Dp = data.page_items(P)
    O = pd.read_parquet(ws / "meta/x/xkp_items.parquet")
    Dp = data.page_flags(Pp, Dp)
    assert (Dp.n_rrn.values == O.n_rrn.values).all() and (Dp.n_sido.values == O.n_sido.values).all()
    assert (Dp.clean.values == O.clean.values).mean() > 0.985          # 주소 칸 판정은 근사(문서 참고)


def test_vlm_train_data(ws):
    import yaml
    from formrec.kie import vlm
    for ft, qs, src, frac, vmax, run, bs, ep in (("pvl", "xk", "all", 1.0, 0, "pvl_ft", 0, 3), ("pvl3", "xk", "all3", 1.0, 800, "pvl_ft3", 16, 2),
                                                 ("pvlp", "xkp", "page", 1.0, 300, "pvl_ftp", 4, 3), ("pvlp10", "xkp", "page", 0.1, 300, "pvl_ftp10", 4, 3)):
        d = ws / f"ft/xk/{ft}"
        if not d.exists():
            continue
        for s in ("train", "val"):
            lines = [vlm.pvl_line(r, src) for r in vlm.pvl_split(qs, s, frac, vmax).itertuples()]
            assert lines == open(d / f"{s}.jsonl", encoding="utf-8").read().splitlines(), (ft, s)
        assert vlm.pvl_config(d, run, ep, bs) == yaml.safe_load(open(d / "pvl16_full.yaml", encoding="utf-8"))


def test_vlm_parse_and_val_table(ws):
    from formrec.kie import vlm
    from formrec.kie.fields import parse
    for n in ("pvlkie_test", "pvlkiep_test", "ovkiep_test", "pvlkiepperm_test"):
        V = pd.read_parquet(ws / f"results/xk/{n}.parquet")
        assert all(json.dumps(parse(r, json.loads(k)), ensure_ascii=False) == p for r, k, p in zip(V.raw, V["keys"], V.pred))
    for i in (1, 2, 3):
        V = pd.read_parquet(ws / f"results/xk/pvlkie_ep{i}_val.parquet")
        assert vlm.val_em_frame(V).equals(pd.read_parquet(ws / f"results/xk/g1/pvlft_ep{i}_val.parquet"))


def test_zeroshot_rescore(ws):
    from formrec.kie import zeroshot
    Z = pd.read_parquet(ws / "results/xk/zs_qwen20_test.parquet")
    Q = zeroshot.sample_queries("test", len(Z))
    assert list(Q.qid) == list(Z.qid)
    R = zeroshot.score_rows(Q, list(Z.raw))
    assert R.equals(Z[R.columns])
