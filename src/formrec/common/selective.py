"""선택적 예측(자동 처리) 지표 — 정밀도 목표에서의 자동화율 · 임계값 적용 · 부트스트랩 CI. 모든 방법이 같은 정의를 쓴다.

자동화율(coverage) = 확신도 conf ≥ thr 인 질의 비율, 정밀도 = 그중 정답 비율. 임계값은 tune에서 정해 test에 한 번 적용한다.
"""
import numpy as np

PRECISIONS = (0.99, 0.995, 0.999)
OPEN_TARGET = 0.001  # 두 임계값 규칙에서 tune open-set 오수락률 목표


def coverage_at_precision(correct, conf, targets=PRECISIONS):
    """conf >= thr 전부 수락. 동점 묶음 경계에서만 자를 수 있다. -> {t: (coverage, thr)}"""
    correct = np.asarray(correct, bool)
    conf = np.asarray(conf, float)
    if len(conf) == 0:                                   # 빈 세트(스모크 · 일부 단계 질의 0개) → 자동 처리 없음
        return {t: (0.0, np.inf) for t in targets}
    o = np.argsort(-conf, kind="stable")
    cs, cc = conf[o], np.cumsum(correct[o])
    ends = np.nonzero(np.r_[cs[1:] != cs[:-1], True])[0]
    prec = cc[ends] / (ends + 1)
    out = {}
    for t in targets:
        ok = np.nonzero(prec >= t)[0]
        if len(ok) == 0:
            out[t] = (0.0, np.inf)
        else:
            e = ends[ok[-1]]
            out[t] = ((e + 1) / len(conf), cs[e])
    return out


def apply_threshold(correct, conf, thr):
    acc = np.asarray(conf) >= thr
    n = acc.sum()
    return dict(cov=float(acc.mean()), prec=float(np.asarray(correct)[acc].mean()) if n else float("nan"), n_acc=int(n))


def bootstrap(correct, conf, n_boot=1000, seed=0, targets=PRECISIONS, by_page=None):
    """Top-1과 Cov@P의 95% CI. by_page가 주어지면 페이지 단위 클러스터 부트스트랩."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, bool); conf = np.asarray(conf, float)
    if by_page is not None:
        pages, inv = np.unique(np.asarray(by_page), return_inverse=True)
        groups = [np.nonzero(inv == k)[0] for k in range(len(pages))]
    res = {"top1": []} | {f"cov@{t}": [] for t in targets}
    for _ in range(n_boot):
        if by_page is not None:
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        else:
            idx = rng.integers(0, len(correct), len(correct))
        c, f = correct[idx], conf[idx]
        res["top1"].append(c.mean())
        for t, (cv, _) in coverage_at_precision(c, f, targets).items():
            res[f"cov@{t}"].append(cv)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}
