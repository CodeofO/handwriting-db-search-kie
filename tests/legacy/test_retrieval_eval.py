"""제안 모델 원본 대조 3 — 평가 run_level(질의 · DB · 1단 · 2단 · 확신도 · λ 선택 원점수)과 요약 표.

- run_level: 원본 kie_eval3.run_level(DB 조회를 x_eval처럼 데이터셋별로 바꿔 끼움) vs 새 evaluate.run_level, xi tune 단계마다 질의 8개(스모크 DB).
- summary: 원본 kie_summary.main()을 임시 폴더(점수 파일만 심볼릭 링크)에서 돌린 summary.csv vs 새 summarize — 바이트 단위.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest


def legacy(legacy_work):
    os.environ.setdefault("HW_DATASET", "xa")
    sys.dont_write_bytecode = True
    if legacy_work not in sys.path:
        sys.path.insert(0, legacy_work)


def test_run_level(legacy_work, workspace_root):
    torch = pytest.importorskip("torch")
    legacy(legacy_work)
    import kie_eval3 as E3
    import x_eval as LXE
    from formrec.common.levels import LEVELS
    from formrec.common.workspace import ROOT
    from formrec.retrieval import evaluate as E
    path = ROOT / "runs/x/xi_prop/final.pt"
    if not path.exists():
        pytest.skip(f"{path} 없음")
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    E.CHECKED.append("skip"); E3.CHECKED.append("skip")           # 입력 점검 그림을 쓰지 않는다
    m, tok, a = E.load_model(path, dev); mo, otok, ao = E3.load_model(path, dev)
    Q = E.XQueries("xi", "tune", smoke=8); Qo = LXE.XQueries("xi", "tune")
    db_fn = E.LevelDB("xi", smoke=8)
    for lv in LEVELS["xi"]:
        pid, text, rows = Q.get(lv); p0, t0, r0 = Qo.get(lv)
        assert (pid == p0[:8]).all() and (text == t0[:8]).all() and (rows == r0[:8]).all()
    old_db = E3.db_texts
    try:
        for lv in LEVELS["xi"]:
            text = Q.get(lv)[1]
            E3.db_texts = lambda data, level, limit=0, keep=(), _t=text: db_fn(level, _t)
            Qs = type("Q", (), dict(X=Qo.X, get=lambda self, level: tuple(v[:8] for v in Qo.get(level))))()
            on, _, nn_, rn = E.run_level(m, tok, a, Q, lv, dev, db_fn, 8, lam=0.5, keep_raw=True)
            oo, _, no, ro = E3.run_level(mo, otok, ao, Qs, lv, dev, 8, lam=0.5, keep_raw=True)
            assert nn_ == no
            for tag in ("final", "t1"):
                x, y = on[tag], oo[tag]
                assert list(x.columns) == list(y.columns)
                for c in x.columns:
                    if x[c].dtype.kind == "f":
                        assert np.abs(x[c].values - y[c].values).max() <= 1e-5, (lv, tag, c)
                    else:
                        assert (x[c].values == y[c].values).all(), (lv, tag, c)
            for k in ("base", "s3", "top", "gt"):
                assert (rn[k] - ro[k]).abs().max().item() <= 1e-5, k
    finally:
        E3.db_texts = old_db


@pytest.mark.parametrize("ds", ["xi", "xc"])
def test_summary_table(legacy_work, workspace_root, ds, tmp_path, monkeypatch):
    legacy(legacy_work)
    import kie_summary as KS
    from formrec.common.levels import LEVELS
    from formrec.common.workspace import ROOT
    from formrec.retrieval.summary import summarize
    sd = ROOT / f"results/{ds}/scores"
    if not sd.exists():
        pytest.skip(f"{sd} 없음")
    (tmp_path / f"results/{ds}").mkdir(parents=True)
    (tmp_path / f"results/{ds}/scores").symlink_to(sd)
    monkeypatch.setattr(KS, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["kie_summary.py", "--data", ds])
    KS.main()
    out = tmp_path / "new.csv"
    summarize(sd, LEVELS[ds]).to_csv(out, index=False)
    assert open(out, "rb").read() == open(tmp_path / f"results/{ds}/summary.csv", "rb").read()
    assert len(pd.read_csv(out)) > 0
