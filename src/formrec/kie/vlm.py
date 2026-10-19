"""손글씨 KIE용 VLM 파인튜닝 데이터 · 추론 (PaddleOCR-VL-1.6 · OvisOCR2).

과제: 서식 이미지 1장(검출 · crop 없음) + 키 목록 → 키 순서대로 "키: 값" 한 줄씩. 프롬프트 = "<image>KIE: 키1, 키2, …".
  data      ft/xk/{ft}/{train,val}.jsonl + pvl16_full.yaml (기준 = ft/kie/pvl/pvl16_full.yaml, 에폭마다 저장 · val loss) — PaddleFormers
  data_ovis ft/xk/{ft}/{train,val}.jsonl (이미지 base64) — ms-swift
  data_mix  일반 KIE(xkg, 인쇄 항목명 키 + '[없음]' 네거티브) + 5개 키 xkp 혼합(OvisOCR2). 서식 분리 평가(xkg test_tpl) 서식의 페이지는 xkp 학습에서도 뺀다.
  infer     vLLM 생성 → results/xk/{out}_{subset}.parquet (qid, raw, pred JSON, lp_mean, lp_min)
            --val_em_out: 체크포인트별 val 질의 완전 일치용 results/xk/g1/{이름}.parquet(pred · text = 키 순서로 다시 쓴 답) — 디스크 보호의 지표 기준 선정에 쓴다.
사용: HW_DATASET=xk python -m formrec.kie.vlm data --epochs 3
      HW_DATASET=xk VLLM_USE_FLASHINFER_SAMPLER=0 python -m formrec.kie.vlm infer --subset test --model_path runs/xk/pvl_ft/export/best --out pvlkie
"""
import argparse
import base64
import json
import math
import shutil
import time

import pandas as pd

from formrec.common import workspace as cfg
from formrec.kie.data import blob_bytes
from formrec.kie.fields import fmt, parse, prompt

ROOT = cfg.ROOT
FT, RUNS, RESULTS = ROOT / "ft/xk", ROOT / "runs/xk", ROOT / "results/xk"
BASE_PVL = ROOT / "ft/kie/pvl"          # PaddleOCR-VL 기준 설정 · 템플릿(비교군 학습과 같은 것)


def load_queries(qset, sub):
    return pd.read_parquet(ROOT / f"meta/x/{qset}_queries_{sub}.parquet")


def messages(keys_json, target):
    return [{"role": "user", "content": "<image>" + prompt(json.loads(keys_json))}, {"role": "assistant", "content": target}]


def pvl_line(r, src):
    return json.dumps({"messages": messages(r.keys, r.target), "images": [f"blob:{src}/{r.row}"]}, ensure_ascii=False)


def ovis_line(r, src):
    return json.dumps({"messages": messages(r.keys, r.target), "images": [base64.b64encode(blob_bytes(src, r.row)).decode()]}, ensure_ascii=False)


def pvl_split(qset, s, train_frac=1.0, val_max=0):
    """PaddleOCR-VL 학습 · val 질의. train_frac < 1 = 학습 페이지(이미지) 단위 고정 표본(데이터량 곡선), val_max = val loss 표본 상한."""
    Q = load_queries(qset, s)
    if s == "train" and train_frac < 1:
        import numpy as np
        it = np.array(sorted(Q.item_id.unique()))
        keep = set(np.random.default_rng(0).choice(it, max(1, round(len(it) * train_frac)), replace=False))
        Q = Q[Q.item_id.isin(keep)]
    if s == "val" and val_max and len(Q) > val_max:
        Q = Q.sample(val_max, random_state=0).sort_index()
    return Q


def pvl_config(d, run, epochs, bs):
    import yaml
    c = yaml.safe_load(open(BASE_PVL / "pvl16_full.yaml", encoding="utf-8"))
    run = RUNS / run
    c.update(train_dataset_path=str(d / "train.jsonl"), custom_register_path=str(d / "paddleocr_vl_v15_template.py"), num_train_epochs=epochs,
             do_eval=True, evaluation_strategy="epoch", save_strategy="epoch", eval_dataset_type="messages", eval_dataset_path=str(d / "val.jsonl"),
             eval_dataset_prob="1.0", output_dir=str(run), logging_dir=str(run / "vdl"))
    if bs:                                              # GPU당 배치를 줄이고 누적으로 전체 배치 유지
        acc = max(1, c["per_device_train_batch_size"] // bs)
        c.update(per_device_train_batch_size=bs, per_device_eval_batch_size=bs, gradient_accumulation_steps=acc)
    return c


def cmd_data(a):
    import yaml
    d = FT / a.ft
    d.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val"):
        Q = pvl_split(a.qset, s, a.train_frac, a.val_max)
        with open(d / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in Q.itertuples():
                f.write(pvl_line(r, a.src) + "\n")
        print(s, len(Q), flush=True)
    shutil.copy(BASE_PVL / "paddleocr_vl_v15_template.py", d / "paddleocr_vl_v15_template.py")
    yaml.safe_dump(pvl_config(d, a.run, a.epochs, a.bs), open(d / "pvl16_full.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("DONE xk_vlm data", d)


def cmd_data_ovis(a):
    d = FT / a.ft
    d.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val"):
        Q = load_queries(a.qset, s)
        if s == "val" and a.val_max and len(Q) > a.val_max:
            Q = Q.sample(a.val_max, random_state=0).sort_index()
        with open(d / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in Q.itertuples():
                f.write(ovis_line(r, a.src) + "\n")
        print(s, len(Q), flush=True)
    print("DONE xk_vlm data_ovis", d)


def mix_split(s, val_max=0):
    """일반 KIE(xkg, 이미지 gpage) + xkp(이미지 page). 분리 서식(test_tpl)의 페이지는 xkp에서도 뺀다."""
    G = pd.read_parquet(ROOT / "meta/x/xkg_items.parquet")
    hold = set(G[G.subset == "test_tpl"].tpl)
    page_tpl = dict(zip(G.page_id, G.tpl))
    Qg = load_queries("xkg", s).assign(src="gpage")
    Qp = load_queries("xkp", s).assign(src="page")
    P = pd.read_parquet(ROOT / "meta/x/xkp_items.parquet").set_index("item_id")
    Qp = Qp[~Qp.item_id.map(lambda i: page_tpl.get(P.at[i, "page_id"], "") in hold)]
    Q = pd.concat([Qg, Qp], ignore_index=True)
    if s == "val" and val_max and len(Q) > val_max:
        Q = Q.sample(val_max, random_state=0)
    Q = Q.sample(frac=1, random_state=1) if s == "train" else Q
    return Q, hold


def cmd_data_mix(a):
    d = FT / a.ft
    d.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val"):
        Q, hold = mix_split(s, a.val_max)
        with open(d / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in Q.itertuples():
                f.write(ovis_line(r, r.src) + "\n")
        print(s, len(Q), "(xkg", int((Q.src == "gpage").sum()), "· xkp", int((Q.src == "page").sum()), ")", flush=True)
    print("DONE xk_vlm data_mix", d, "분리 서식", sorted(hold))


def logprob_stats(c):
    """생성 토큰 로그 확률의 평균 · 최소(유한값만)."""
    lps = [x for x in (next(iter(d.values())).logprob for d in (c.logprobs or []) if d) if math.isfinite(x)]
    return (sum(lps) / len(lps) if lps else float("nan")), (min(lps) if lps else float("nan"))


def val_em_frame(Q):
    """체크포인트 선정용: 출력 · 정답을 같은 형식(키 순서 · 공백 정규화)으로 → (pred, text)."""
    return pd.DataFrame(dict(pred=[fmt(json.loads(k), parse(r, json.loads(k))) for r, k in zip(Q.raw, Q["keys"])],
                             text=[fmt(json.loads(k), parse(t, json.loads(k))) for t, k in zip(Q.target, Q["keys"])]))


def cmd_infer(a):
    import io

    from PIL import Image
    from vllm import LLM, SamplingParams
    Q = load_queries(a.qset, a.subset)
    if a.limit:
        Q = Q.sample(a.limit, random_state=0).reset_index(drop=True)
    if a.model == "ovis":
        llm = LLM(model=str(ROOT / a.model_path), gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, max_model_len=16384, gdn_prefill_backend="triton")
        chat_kw = dict(chat_template_kwargs={"enable_thinking": False}, mm_processor_kwargs={"images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}})
    else:
        llm = LLM(model=str(ROOT / a.model_path), gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, trust_remote_code=True,
                  max_model_len=16384, enable_prefix_caching=False, mm_processor_cache_gb=0)
        chat_kw = {}
    sp = SamplingParams(temperature=0.0, max_tokens=256, logprobs=0)
    raws, lpm, lpn = [], [], []
    t0 = time.time()
    for i in range(0, len(Q), a.batch):
        ch = Q.iloc[i:i + a.batch]
        msgs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": Image.open(io.BytesIO(blob_bytes(a.src, r.row))).convert("RGB")},
                                              {"type": "text", "text": prompt(json.loads(r.keys))}]}] for r in ch.itertuples()]
        for o in llm.chat(msgs, sp, use_tqdm=False, **chat_kw):
            c = o.outputs[0]
            raws.append(c.text)
            m, n = logprob_stats(c)
            lpm.append(m)
            lpn.append(n)
        print(f"{i + len(ch)}/{len(Q)} {time.time() - t0:.0f}s", flush=True)
    Q = Q.assign(raw=raws, lp_mean=lpm, lp_min=lpn)
    Q["pred"] = [json.dumps(parse(r, json.loads(k)), ensure_ascii=False) for r, k in zip(Q.raw, Q["keys"])]
    RESULTS.mkdir(parents=True, exist_ok=True)
    Q.to_parquet(RESULTS / f"{a.out}_{a.subset}.parquet", index=False)
    if a.val_em_out:
        g = RESULTS / "g1"
        g.mkdir(parents=True, exist_ok=True)
        val_em_frame(Q).to_parquet(g / f"{a.val_em_out}.parquet", index=False)
    print(f"DONE xk_vlm infer {a.out} {a.subset} n={len(Q)} {1000 * (time.time() - t0) / max(1, len(Q)):.1f} ms/질의")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "data_ovis", "data_mix", "infer"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--out", default="pvlkie")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val_em_out", default="")
    ap.add_argument("--src", default="all", help="이미지 묶음(all = 1240폭, all3 = 1860폭, page = 전체 페이지)")
    ap.add_argument("--ft", default="pvl")
    ap.add_argument("--run", default="pvl_ft")
    ap.add_argument("--bs", type=int, default=0, help="GPU당 배치(0 = 기준 32), 누적으로 전체 배치 유지")
    ap.add_argument("--val_max", type=int, default=0, help="val loss용 표본 상한(0 = 전체)")
    ap.add_argument("--model", default="pvl", choices=["pvl", "ovis"])
    ap.add_argument("--qset", default="xk", help="질의 세트: xk = 사람 영역, xkp = 전체 페이지, xkpperm = 키 순서 바꿈")
    ap.add_argument("--train_frac", type=float, default=1.0, help="학습 페이지 비율(데이터량 곡선)")
    a = ap.parse_args(argv)
    {"data": cmd_data, "data_ovis": cmd_data_ovis, "data_mix": cmd_data_mix, "infer": cmd_infer}[a.cmd](a)


if __name__ == "__main__":
    main()
