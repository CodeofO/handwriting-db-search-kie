"""요약 지표 단위 테스트."""
import numpy as np
import pandas as pd

from formrec.retrieval import summary as S


def test_ned_auc():
    assert S.ned_sim(pd.Series(["abc", None]), ["abd", ""]) == (1 - 1 / 3 + 1) / 2
    assert S.auc([2, 3], [1]) == 1.0 and S.auc([1], [1]) == 0.5 and np.isnan(S.auc([], [1]))


def _frame(rng, n):
    c = rng.random(n) < 0.9
    conf = np.where(c, rng.random(n) + 0.5, rng.random(n))
    return pd.DataFrame(dict(pid=np.arange(n), text=["a"] * n, pred=np.where(c, "a", "b"), correct=c, conf=conf, s1=conf, open_pred="b",
                             open_conf=rng.random(n) * 0.6, open_s1=rng.random(n) * 0.6, margin_conf=conf * 0.5, margin_open_conf=rng.random(n) * 0.3))


def test_summarize_table(tmp_path):
    rng = np.random.default_rng(0)
    for m in ("P-x", "G1-y"):
        for lv in ("I1", "I2"):
            for sub in ("tune", "test"):
                _frame(rng, 300).to_parquet(tmp_path / f"{m}__{lv}__{sub}.parquet", index=False)
    T = S.summarize(tmp_path, ("I1", "I2"))
    assert len(T) == 6 and set(T.level) == {"I1", "I2", "ALL"}
    r = T[(T.method == "P-x") & (T.level == "ALL")].iloc[0]
    assert r.n == 600 and 0 <= r.pcov <= 1 and r.sconf99 in ("기본", "1·2등 점수 차")
    assert "cov@0.999" in T and "openFA2@0.99" in T
