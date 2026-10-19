"""우도 기반 레코드 검증(Likelihood-based Record Verification) — 편집 거리 레코드 매칭을 대신하는 후보 방법.

DB 후보 레코드 r마다 그 레코드 값이 정답이라고 놓고
  ℓ(r) = log p(키1: r[키1] \\n 키2: r[키2] … <끝> | 페이지 이미지, 키 목록)       (교사 강제 · 한 번의 순전파)
를 계산해 이미지 증거로 후보를 고른다(레코드 = 숨은 원인, 이미지 = 관측).
  · 후보: 자유 생성 출력과 요청 키 편집 거리 합이 작은 레코드 K개(요청 키로 투영해 중복 제거) — 1차 거름.
  · 선택: r* = argmax ℓ(r).
  · DB 밖 판정: 우도비 Λ = ℓ(자유 생성) − ℓ(r*) 가 θ보다 크면 자유 생성 출력을 쓴다. θ는 tune에서.
  · 자동 처리 확신도: 1 · 2등 우도 차, −Λ.
단계: cands(CPU) → score(GPU, vLLM prompt_logprobs) → decide(CPU, DB 밖 비율 0 · 5 · 20 %, 편집 거리 규칙 'ed'와 같은 틀로 비교).
출력: results/xk/lrv/{cands,ll}_{qset}_{name}_{subset}.parquet, results/xk/lrv_{qset}_{name}.csv
사용:
  python -m formrec.kie.verify cands --qset xkp --name ovkiep
  python -m formrec.kie.verify score --qset xkp --name ovkiep --model ovis --model_path runs/xk/ovis_ftp/export/best --src page   (vllm 환경)
  python -m formrec.kie.verify decide --qset xkp --name ovkiep
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.common.selective import apply_threshold, coverage_at_precision
from formrec.kie import evaluate as E
from formrec.kie.fields import fmt, norm, prompt, target_dict
from formrec.kie.oov import DROPS, dropped_people

ROOT = cfg.ROOT
OUT = ROOT / "results/xk/lrv"
END = {"ovis": "<|im_end|>", "pvl": "</s>"}


def candidates(Q, P, D, k=8, pool=64, chunk=64):
    """질의마다 편집 거리 합 상위 pool개 → 요청 키 투영 기준 중복 제거 후 K개. 같은 투영의 출처 사람(src)은 모두 남긴다
    (DB 밖 실험에서 출처가 전부 빠질 때만 후보 삭제)."""
    cols, src = E.db_columns(D), D.src.astype(str).values
    rows = []
    for keys in sorted({tuple(json.loads(x)) for x in Q["keys"]}):
        ii = [i for i, x in enumerate(Q["keys"]) if tuple(json.loads(x)) == keys]
        for j in range(0, len(ii), chunk):
            ch = ii[j:j + chunk]
            tot = E.distance_sum([(c, [" ".join(str(P[i].get(c) or "").split()) for i in ch]) for c in keys], cols)
            top = np.argsort(tot, 1, kind="stable")[:, :pool]
            for i, t, row in zip(ch, top, tot):
                seen, ct, cs, cd = {}, [], [], []
                for b in t:
                    txt = fmt(keys, {c: cols[c][b] for c in keys})
                    if txt in seen:
                        cs[seen[txt]].append(src[b])
                        continue
                    seen[txt] = len(ct)
                    ct.append(txt)
                    cs.append([src[b]])
                    cd.append(int(row[b]))
                    if len(ct) >= k:
                        break
                rows.append(dict(qid=Q.qid[i], keys=json.dumps(list(keys), ensure_ascii=False), free=fmt(keys, P[i]), cand=ct, cand_src=cs, cand_d=cd))
    return pd.DataFrame(rows).set_index("qid").loc[Q.qid].reset_index()


def cmd_cands(a):
    D = E.record_db()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("tune", "test"):
        Q, P, _ = E.method_preds("vlm", a.name, sub, a.qset)
        C = candidates(Q, P, D, a.k, a.pool)
        C.to_parquet(OUT / f"cands_{a.qset}_{a.name}_{sub}.parquet", index=False)
        print(sub, len(C), "질의 · 후보 평균", round(C.cand.str.len().mean(), 2), flush=True)


def cmd_score(a):
    """vLLM prompt_logprobs로 (자유 생성 · 후보 K개) 정답 문자열의 로그 우도 합. 끝 표식까지 포함(짧은 답이 유리해지지 않게)."""
    import io

    from PIL import Image
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from formrec.kie.data import blob_bytes
    tok = AutoTokenizer.from_pretrained(str(ROOT / a.model_path), trust_remote_code=True)
    kw = dict(gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, max_model_len=16384)
    if a.model == "ovis":
        llm = LLM(model=str(ROOT / a.model_path), gdn_prefill_backend="triton", **kw)
        mmk = {"images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}}
    else:
        llm = LLM(model=str(ROOT / a.model_path), trust_remote_code=True, enable_prefix_caching=False, mm_processor_cache_gb=0, **kw)
        mmk = None
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    for sub in a.subsets.split(","):
        C = pd.read_parquet(OUT / f"cands_{a.qset}_{a.name}_{sub}.parquet")
        Q = E.load_queries(a.qset, sub).set_index("qid")
        if a.limit:
            C = C.head(a.limit)
        res, t0 = [], time.time()
        for i in range(0, len(C), a.batch):
            ch = C.iloc[i:i + a.batch]
            reqs, meta = [], []
            for r in ch.itertuples():
                keys = json.loads(r.keys)
                u = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(keys)}]}]
                pre = tok.apply_chat_template(u, add_generation_prompt=True, enable_thinking=False, tokenize=False)
                n0 = len(tok(pre, add_special_tokens=False).input_ids)
                img = Image.open(io.BytesIO(blob_bytes(a.src, Q.at[r.qid, "row"]))).convert("RGB")
                for j, txt in enumerate([r.free] + list(r.cand)):
                    full = pre + txt + END[a.model]
                    n = len(tok(full, add_special_tokens=False).input_ids) - n0
                    req = {"prompt": full, "multi_modal_data": {"image": img}}
                    if mmk:
                        req["mm_processor_kwargs"] = mmk
                    reqs.append(req)
                    meta.append((r.qid, j, n))
            for (qid, j, n), o in zip(meta, llm.generate(reqs, sp, use_tqdm=False)):
                lp = [next(iter(d.values())).logprob for d in o.prompt_logprobs[-n:]]
                res.append(dict(qid=qid, j=j, ll=float(np.sum(lp)), n_tok=n, lp_min=float(np.min(lp))))
            print(f"{sub} {i + len(ch)}/{len(C)} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(res).to_parquet(OUT / f"ll_{a.qset}_{a.name}_{sub}.parquet", index=False)
        print(f"DONE xk_verify score {sub} n={len(C)} {1000 * (time.time() - t0) / max(1, len(C)):.0f} ms/질의", flush=True)


def load(qset, name, sub):
    C = pd.read_parquet(OUT / f"cands_{qset}_{name}_{sub}.parquet")
    L = pd.read_parquet(OUT / f"ll_{qset}_{name}_{sub}.parquet")
    ll = {(q, j): v for q, j, v in zip(L.qid, L.j, L.ll)}
    Q = E.load_queries(qset, sub).set_index("qid")
    C = C[C.qid.isin({q for q, _ in ll})].reset_index(drop=True)
    return C, ll, Q


def decide_one(C, ll, Q, drop, theta, rule):
    """질의마다 (정답 여부, 확신도들). rule = 'lrv'(우도) | 'ed'(같은 후보 풀에서 편집 거리 1등 — 기존 방법 재현)."""
    ok, m12, lam = [], [], []
    for r in C.itertuples():
        keys, tg = json.loads(r.keys), target_dict(Q.at[r.qid, "target"])
        live = [j for j, s in enumerate(r.cand_src) if not set(s) <= drop]     # DB 밖 실험: 출처가 모두 빠진 후보는 DB에 없음
        lf = ll[(r.qid, 0)]
        if live:
            if rule == "lrv":
                sc = sorted(((ll[(r.qid, j + 1)], j) for j in live), reverse=True)
                best = sc[0][1]
                L = lf - sc[0][0]
                mm = sc[0][0] - (sc[1][0] if len(sc) > 1 else sc[0][0] - 50)
            else:
                dd = sorted((r.cand_d[j], j) for j in live)
                best = dd[0][1]
                L = float(dd[0][0])
                mm = float((dd[1][0] if len(dd) > 1 else dd[0][0] + 10) - dd[0][0])
            txt = r.cand[best] if L <= theta else r.free
        else:
            txt, L, mm = r.free, 99.0, 0.0
        pr = dict(line.split(": ", 1) if ": " in line else (line, "") for line in txt.split("\n"))
        ok.append(all(norm(k, pr.get(k, "")) == norm(k, tg[k]) for k in keys))
        m12.append(mm)
        lam.append(-L)
    return np.array(ok), {"1·2등 차": np.array(m12), "−Λ(또는 −d1)": np.array(lam)}


GRIDS = (("ed", list(range(0, 21))), ("lrv", [x / 2 for x in range(0, 81)] + [1e9]))


def cmd_decide(a):
    rows = []
    T = {s: load(a.qset, a.name, s) for s in ("tune", "test")}
    for p in DROPS:
        drops = {s: set(map(str, dropped_people(a.qset, s, p))) for s in ("tune", "test")}    # oov와 같은 시드 · 같은 빠짐
        for rule, grid in GRIDS:
            best = max(grid, key=lambda th: (decide_one(*T["tune"], drops["tune"], th, rule)[0].mean(), -th))
            ok, conf = decide_one(*T["test"], drops["test"], best, rule)
            C, _, Q = T["test"]
            oov = Q.loc[C.qid, "item_id"].astype(str).isin(drops["test"]).values
            okt, conft = decide_one(*T["tune"], drops["tune"], best, rule)
            cb = max(conft, key=lambda c: coverage_at_precision(okt, conft[c], (0.99,))[0.99][0])
            _, thr = coverage_at_precision(okt, conft[cb], (0.99,))[0.99]
            r = apply_threshold(ok, conf[cb], thr)
            rows.append(dict(rule=rule, drop=p, theta=best, em=ok.mean(), em_in=ok[~oov].mean(), em_out=ok[oov].mean() if oov.any() else np.nan,
                             auto_conf=cb, auto_cov=r["cov"], auto_prec=r["prec"], n=len(ok)))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / f"results/xk/lrv_{a.qset}_{a.name}.csv", index=False)
    with pd.option_context("display.width", 220):
        print(R.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "score", "decide"])
    ap.add_argument("--qset", default="xkp")
    ap.add_argument("--name", default="ovkiep")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--model", default="ovis", choices=["ovis", "pvl"])
    ap.add_argument("--model_path", default="runs/xk/ovis_ftp/export/best")
    ap.add_argument("--src", default="page")
    ap.add_argument("--subsets", default="tune,test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"cands": cmd_cands, "score": cmd_score, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
