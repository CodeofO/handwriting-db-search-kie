"""방법 × 단계 지표 요약 (제안 모델 · 비교군 공용): tune에서 임계값을 정하고 test에 한 번 적용.

지표
  top1 = 정확도, ned_sim = 1 − 정규화 편집 거리(생성형의 부분 정답 반영), n_err_val = 그 단계 tune 오답 수(단계별 임계값 신뢰도).
  단계별 임계값(참고): cov@P · prec@P(정밀도 P 임계값의 자동화율 · 실제 정밀도), openFA@P(DB 밖 오수락),
    cov2@P · openFA2@P(DB 밖 거부 규칙: s1 하한 τs(tune DB 밖 오수락 ≤ 0.1%) + 마진 임계값).
  통합 임계값(본문): 방법마다 기본 단계 tune을 묶어 정한 임계값 1개를 모든 단계 test에 적용(운영 조건과 같음).
    단계별 tune은 오답이 몇 건뿐이라 정밀도 99.9% 임계값이 정해지지 않기 때문.
    pcov · pprec · popenFA(99.9%), pcov99 · pprec99(99%: 정답 라벨 오기 ≈0.5%가 있는 데이터에서는 99.9%에 어느 방법도 닿지 못함),
    pcov2 · pprec2 · popenFA2(DB 밖 거부 규칙), auc_open(정답 확신도 > DB 밖 1등 확신도일 확률), fa_at95(정답 95% 수락 임계값의 DB 밖 오수락).
  확신도 선택(scov · sprec · sconf · sfa95_ · sopenFA, 접미 99 · 99.9): 확신도 종류(기본 · 1·2등 점수 차 · 토큰 최저 확률)를
    tune(기본 단계 합침)에서 목표 정밀도의 자동화율이 가장 큰 것으로 고른다 — 모든 방법에 같은 규칙.
  ALL 행 = 기본 단계 test를 합친 시스템 단위(통합 임계값의 정밀도 조건은 이 행으로 확인).
입력: results/{data}/scores/{method}__{level}__{tune,test}.parquet
출력: results/{data}/summary.csv (+ 화면)
사용: python -m formrec.retrieval.summary --data xa
"""
import argparse

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from formrec.common.levels import LEVELS
from formrec.common.selective import OPEN_TARGET, apply_threshold, coverage_at_precision
from formrec.common.workspace import ROOT

PT = 0.999
ALL = "ALL"
CANDS = (("conf", "open_conf", "기본"), ("margin_conf", "margin_open_conf", "1·2등 점수 차"), ("conf_min", "open_conf_min", "토큰 최저 확률"))
SHOW = ["method", "level", "n", "top1", "cov@0.999", "prec@0.999", "pcov", "pprec", "popenFA", "pcov2", "auc_open", "fa_at95", "n_err_val"]


def ned_sim(pred, text):
    """1 − NED (편집 거리 / 두 문자열 중 긴 길이). 필드 구분 기호 포함, 레코드 단위 평균."""
    return float(np.mean([1 - Levenshtein.distance(p, t) / max(len(p), len(t), 1) for p, t in zip(pred.fillna(""), text)]))


def auc(pos, neg):
    """P(pos > neg) + ½P(=) (순위 기반)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().values; n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def pooled_thresholds(ds):
    """방법 하나의 기본 단계 tune 프레임 목록 → (thr, τs, thr2, thr99) 통합 임계값."""
    D = pd.concat(ds, ignore_index=True)
    thr = coverage_at_precision(D.correct, D.conf, (PT,))[PT][1]
    ts = float(np.quantile(D.open_s1, 1 - OPEN_TARGET))
    k = D.s1 >= ts
    thr2 = coverage_at_precision(D.correct[k], D.conf[k], (PT,))[PT][1]
    thr99 = coverage_at_precision(D.correct, D.conf, (0.99,))[0.99][1]
    return thr, ts, thr2, thr99


def pooled(v, P):
    thr, ts, thr2, thr99 = P
    r = apply_threshold(v.correct, v.conf, thr); r99 = apply_threshold(v.correct, v.conf, thr99)
    acc2 = (v.s1 >= ts) & (v.conf >= thr2)
    ok = v.conf[v.correct]
    t95 = float(np.quantile(ok, 0.05)) if len(ok) else np.inf
    return {"pcov": r["cov"], "pprec": r["prec"], "pcov99": r99["cov"], "pprec99": r99["prec"], "popenFA": float((v.open_conf >= thr).mean()),
            "pcov2": float(acc2.mean()), "pprec2": float(v.correct[acc2].mean()) if acc2.any() else float("nan"),
            "popenFA2": float(((v.open_s1 >= ts) & (v.open_conf >= thr2)).mean()),
            "auc_open": auc(ok, v.open_conf), "fa_at95": float((v.open_conf >= t95).mean())}


def choose(ds, t):
    """확신도 종류를 tune(기본 단계 합침)에서 고름 — 목표 정밀도 t에서 자동화율 최대(동률이면 앞의 것)."""
    D = pd.concat(ds, ignore_index=True); best = None
    for c, oc, nm in CANDS:
        if c in D and oc in D:
            cov, thr = coverage_at_precision(D.correct, D[c], (t,))[t]
            if best is None or cov > best[0] + 1e-12:
                best = (cov, thr, c, oc, nm)
    return best


def pooled_sel(v, S):
    out = {}
    for t, (cov, thr, c, oc, nm) in S.items():
        k = f"{100 * t:g}"
        r = apply_threshold(v.correct, v[c], thr); ok = v[c][v.correct]
        t95 = float(np.quantile(ok, 0.05)) if len(ok) else np.inf
        out.update({f"scov{k}": r["cov"], f"sprec{k}": r["prec"], f"sconf{k}": nm, f"sfa95_{k}": float((v[oc] >= t95).mean()),
                    f"sopenFA{k}": float((v[oc] >= thr).mean())})
    return out


def one(v, d):
    """단계 하나: v = test, d = tune → 단계별 임계값 지표."""
    row = {"n": len(v), "top1": float(v.correct.mean()), "ned_sim": ned_sim(v.pred, v.text), "n_err_val": int((~d.correct.astype(bool)).sum())}
    for t, (c, thr) in coverage_at_precision(d.correct, d.conf).items():
        r = apply_threshold(v.correct, v.conf, thr)
        row[f"cov@{t}"] = r["cov"]; row[f"prec@{t}"] = r["prec"]; row[f"openFA@{t}"] = float((v.open_conf >= thr).mean())
    ts = float(np.quantile(d.open_s1, 1 - OPEN_TARGET))
    pd_, pv = d.s1 >= ts, v.s1 >= ts
    for t, (c, thr) in coverage_at_precision(d.correct[pd_], d.conf[pd_]).items():
        acc = pv & (v.conf >= thr)
        row[f"cov2@{t}"] = float(acc.mean()); row[f"openFA2@{t}"] = float(((v.open_s1 >= ts) & (v.open_conf >= thr)).mean())
    return row


def score_pairs(sd):
    """scores 폴더 → [(방법, 단계, test 경로, tune 경로)] (tune이 있는 것만, 파일 이름 순)."""
    pairs = []
    for p in sorted(sd.glob("*__test.parquet")):
        m, lv, _ = p.stem.split("__")
        tp = sd / f"{m}__{lv}__tune.parquet"
        if tp.exists():
            pairs.append((m, lv, p, tp))
    return pairs


def summarize(sd, base_levels):
    """scores 폴더 → 요약 표(DataFrame)."""
    base_levels = tuple(base_levels)
    pairs = score_pairs(sd)
    vals = {}                                        # 방법별 기본 단계 tune → 통합 임계값
    for m, lv, _, tp in pairs:
        if lv in base_levels:
            vals.setdefault(m, []).append(pd.read_parquet(tp))
    POOL = {m: pooled_thresholds(ds) for m, ds in vals.items() if len(ds) == len(base_levels)}
    SEL = {m: {t: choose(ds, t) for t in (0.99, 0.999)} for m, ds in vals.items() if len(ds) == len(base_levels)}
    rows = []
    for m, lv, p, tp in pairs:
        v, d = pd.read_parquet(p), pd.read_parquet(tp)
        r = dict(method=m, level=lv, **one(v, d))
        if m in POOL:
            r.update(pooled(v, POOL[m])); r.update(pooled_sel(v, SEL[m]))
        rows.append(r)
    tests = {}
    for m, lv, p, tp in pairs:
        if lv in base_levels and m in POOL:
            tests.setdefault(m, []).append(pd.read_parquet(p))
    for m, vs in tests.items():
        if len(vs) == len(base_levels):
            V = pd.concat(vs, ignore_index=True)
            r = {"method": m, "level": ALL, "n": len(V), "top1": float(V.correct.mean())}
            r.update(pooled(V, POOL[m])); r.update(pooled_sel(V, SEL[m])); rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(LEVELS))
    a = ap.parse_args(argv)
    S = summarize(ROOT / f"results/{a.data}/scores", LEVELS[a.data])
    out = ROOT / f"results/{a.data}/summary.csv"
    S.to_csv(out, index=False)
    cols = [c for c in SHOW if c in S]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(S[cols].round(4).to_string(index=False))
    print("DONE kie_summary", out)


if __name__ == "__main__":
    main()
