"""비교군 채점: 생성형 OCR의 필드 이미지별 생성 결과(ppocr.infer · vlm.infer) → 단계별 레코드 점수 (G1 = 생성만, Exact Match).

레코드 = 필드별 생성 문자열을 SEP로 이음, 모든 필드가 정답과 완전히 같아야 정답. conf = 필드 확신도(score)의 최솟값.
DB 밖 열: open_pred = 생성 결과, open_conf = 정답이면 −1e9 · 아니면 conf.
확신도 후보 2(VLM만): conf_min = 필드별 생성 토큰 최저 확률의 최솟값. 본문 자동화율은 tune에서 후보 중 고른다(요약 단계).
xa는 칼럼 누락 질의 F3#d{k}(F3에서 필드 k를 비움)도 낸다 — 제안 모델과 같은 질의 · 같은 정답(blank).
입력: results/{ds}/g1/{tag}_{subset}.parquet (행 = 그 세트 item, cfg.queries 순서)
출력: results/{ds}/scores/G1-{tag}__{단계}__{subset}.parquet
사용: env HW_DATASET=xa python -m formrec.baselines.evaluate --tag ppocrv5ft
"""
import argparse

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.levels import LEVELS, SEP, blank, level_table


def score_frame(pid, text, gen, ps, pm=None):
    ok = gen == text
    o = pd.DataFrame(dict(pid=pid, text=text, pred=gen, correct=ok, conf=ps, s1=ps, open_pred=gen,
                          open_conf=np.where(ok, -1e9, ps), open_s1=ps))
    if pm is not None:
        o["conf_min"], o["open_conf_min"] = pm, np.where(ok, -1e9, pm)
    return o


def score_levels(ds, L, pred, score, smin=None):
    """{파일 단계 이름: 점수 표}. L = level_table(ds, sub), pred/score/smin = item_id → 값."""
    out = {}
    for lv in LEVELS[ds]:
        t = L[L.level == lv]
        if len(t) == 0:
            continue
        gen = np.array([SEP.join(pred[i] for i in its) for its in t["items"]])
        ps = np.array([min(score[i] for i in its) for its in t["items"]])
        pm = np.array([min(smin[i] for i in its) for its in t["items"]]) if smin is not None else None
        out[lv] = score_frame(t.pid.values, t.text.values, gen, ps, pm)
    if ds == "xa":
        t = L[L.level == "F3"]
        for k in range(3):
            txt = np.array([blank(x, k) for x in t.text.values])
            gen = np.array([SEP.join("" if j == k else pred[i] for j, i in enumerate(its)) for its in t["items"]])
            ps = np.array([min(score[i] for j, i in enumerate(its) if j != k) for its in t["items"]])
            pm = np.array([min(smin[i] for j, i in enumerate(its) if j != k) for its in t["items"]]) if smin is not None else None
            out[f"F3#d{k}"] = score_frame(t.pid.values, txt, gen, ps, pm)
    return out


def load_generation(path):
    g = pd.read_parquet(path)
    pred = dict(zip(g.image_id, g.pred.fillna(""))); score = dict(zip(g.image_id, g.score.fillna(0.0)))
    smin = dict(zip(g.image_id, np.exp(g.lp_min.fillna(-1e9).clip(upper=0)))) if "lp_min" in g else None
    return pred, score, smin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subsets", default="tune,test")
    ap.add_argument("--partial", action="store_true", help="스모크: 생성 결과가 없는 이미지를 포함한 레코드는 건너뜀")
    a = ap.parse_args(argv)
    ds = cfg.DATASET
    sd = cfg.RESULTS / "scores"; sd.mkdir(parents=True, exist_ok=True)
    for sub in a.subsets.split(","):
        pred, score, smin = load_generation(cfg.RESULTS / f"g1/{a.tag}_{sub}.parquet")
        L = level_table(ds, sub)
        if a.partial:
            L = L[[all(i in pred for i in its) for its in L["items"]]]
        for lv, o in score_levels(ds, L, pred, score, smin).items():
            o.to_parquet(sd / f"G1-{a.tag}__{lv}__{sub}.parquet", index=False)
            print(f"{sub} {lv}: n={len(o)} 생성만 정확도 {o.correct.mean():.4f}", flush=True)
    print("DONE x_base_eval", ds, a.tag)


if __name__ == "__main__":
    main()
