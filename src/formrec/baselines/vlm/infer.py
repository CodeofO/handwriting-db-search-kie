"""VLM OCR 추론(vLLM) — PaddleOCR-VL-1.6 · OvisOCR2, 파인튜닝 모델(--model_path) 또는 원본. crop 1장 = 요청 1건.

모델별 설정은 공식 문서/코드 기준:
- PaddleOCR-VL-1.6: 프롬프트 "OCR:", temperature 0, max_model_len 16384, prefix/mm 캐시 끔
- OvisOCR2: 페이지 단위 공식 프롬프트 전문(앞 \\n 포함), min/max pixels 448²/2880², enable_thinking false, gdn_prefill_backend triton
확신도 score = 생성 토큰 평균 확률 exp(lp_mean) (PP-OCRv5 rec_score와 같은 성격), lp_min = 최저 토큰 로그확률(확신도 후보 2).
출력: results/{ds}/g1/{out_name or tag_subset[_n{limit}]}.parquet (질의 순서 = cfg.queries)
사용: env HW_DATASET=xa VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 python -m formrec.baselines.vlm.infer \
        --model paddlevl --subset tune --model_path runs/xa/pvl_ft/export/best --out_name pvlft_tune
"""
import argparse
import io
import math
import time

import numpy as np

from formrec.common import workspace as cfg
from formrec.baselines.normalize import allowed_chars, make_normalizer

OVIS_PROMPT = ("\nExtract all readable content from the image in natural human reading order and output the result "
               "as a single Markdown document. For charts or images, represent them using an HTML image tag: "
               "<img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are "
               "bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: "
               "<table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text "
               "without translation or paraphrasing.")

MODELS = {
    "paddlevl": dict(hf="PaddlePaddle/PaddleOCR-VL-1.6", local="PaddleOCR-VL-1.6", prompt="OCR:"),
    "ovis": dict(hf="ATH-MaaS/OvisOCR2", local="OvisOCR2", prompt=OVIS_PROMPT),
}
COLS = ["image_id", "order", "text", "raw", "pred", "score", "lp_mean", "lp_min", "finish", "model"]


def model_path(key):
    p = cfg.MODELS / MODELS[key]["local"]
    return str(p) if p.exists() else MODELS[key]["hf"]


def llm_kwargs(model, mp):
    """(LLM 생성 인자, SamplingParams 인자, chat 인자)."""
    kw = dict(model=mp, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1})
    sp = dict(temperature=0.0, max_tokens=256, logprobs=0)
    chat_kw = {}
    if model == "paddlevl":
        kw.update(trust_remote_code=True, max_model_len=16384, enable_prefix_caching=False, mm_processor_cache_gb=0)
    else:  # ovis
        kw.update(max_model_len=16384, gdn_prefill_backend="triton")
        chat_kw = dict(chat_template_kwargs={"enable_thinking": False},
                       mm_processor_kwargs={"images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}})
    return kw, sp, chat_kw


def logprob_summary(logprobs):
    """vLLM 토큰 logprob 목록 → (평균, 최솟값). 유한값만."""
    lps = [next(iter(d.values())).logprob for d in (logprobs or []) if d]
    lps = [x for x in lps if math.isfinite(x)]
    return (sum(lps) / len(lps) if lps else float("nan")), (min(lps) if lps else float("nan"))


def out_name(a):
    return a.out_name or "_".join(x for x in [a.tag or a.model, a.subset, f"n{a.limit}" if a.limit else ""] if x)


def finish_frame(m, ds, tag):
    """raw · lp_mean → pred(정리) · score · model. 정리 함수가 정답을 망가뜨리지 않는지 1회 점검."""
    from formrec.common import input_check
    norm = make_normalizer(ds, allowed_chars(ds, cfg.queries("train").text), markup=True)
    if not input_check.check_normalize(m.text, norm, f"{ds}_{tag}_normalize")["pass"]:
        raise SystemExit("생성 결과 정리 점검 실패 — logs/checks/ 확인")
    m["pred"] = m.raw.map(norm)
    m["score"] = np.exp(m.lp_mean.fillna(-1e9).clip(upper=0))
    m["model"] = tag
    return m


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--subset", required=True, choices=list(cfg.SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="스모크: 무작위 N개(시드 0)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--model_path", default="", help="파인튜닝 모델 폴더(작업 공간 상대, 없으면 models/ 원본)")
    ap.add_argument("--tag", default="", help="출력 파일명의 모델 이름 (기본 = --model)")
    ap.add_argument("--out_name", default="", help="출력 파일명(확장자 제외)을 직접 지정 (예: pvlft_tune)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from PIL import Image
    from formrec.baselines.ppocr.infer import load_images
    mc = MODELS[a.model]
    m, pngs = load_images(a.subset)
    m = m.assign(png=pngs)
    if a.limit:
        m = m.sample(a.limit, random_state=0).reset_index(drop=True)

    from vllm import LLM, SamplingParams
    mp = str(cfg.ROOT / a.model_path) if a.model_path else model_path(a.model)
    print("model:", mp, flush=True)
    kw, spk, chat_kw = llm_kwargs(a.model, mp)
    sp = SamplingParams(**spk)
    llm = LLM(**kw)

    raws, lp_mean, lp_min, fin = [], [], [], []
    t0 = time.time()
    for i in range(0, len(m), a.batch):
        chunk = m.png.iloc[i:i + a.batch]
        msgs = [[{"role": "user", "content": [
            {"type": "image_pil", "image_pil": Image.open(io.BytesIO(p)).convert("RGB")},
            {"type": "text", "text": mc["prompt"]}]}] for p in chunk]
        outs = llm.chat(msgs, sp, use_tqdm=False, **chat_kw)
        for o in outs:
            c = o.outputs[0]
            raws.append(c.text)
            mean, mn = logprob_summary(c.logprobs)
            lp_mean.append(mean); lp_min.append(mn); fin.append(c.finish_reason)
        print(f"{i + len(chunk)}/{len(m)}  {time.time() - t0:.0f}s", flush=True)
    m["raw"], m["lp_mean"], m["lp_min"], m["finish"] = raws, lp_mean, lp_min, fin
    m = finish_frame(m, cfg.DATASET, a.tag or a.model)
    out = cfg.RESULTS / "g1"
    out.mkdir(parents=True, exist_ok=True)
    name = out_name(a)
    cfg.atomic_write(out / f"{name}.parquet", lambda t: m[COLS].to_parquet(t, index=False))
    el = time.time() - t0
    print(f"DONE {name} exact={(m.pred == m.text).mean():.4f} n={len(m)} {el:.0f}s ({1000 * el / len(m):.1f} ms/img)")


if __name__ == "__main__":
    main()
