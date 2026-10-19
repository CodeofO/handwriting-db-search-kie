"""VLM 파인튜닝 데이터 (PaddleOCR-VL-1.6 = 공식 PaddleFormers, OvisOCR2 = ms-swift), 분할표 meta/x/{ds}_items.parquet 기준.

- pvl : ft/{ds}/pvl/{train,val}.jsonl + pvl16_full.yaml (기준 = configs/pvl/pvl16_full_base.yaml, do_eval + 에폭마다 val loss)
        {"messages":[{"role":"user","content":"<image>OCR:"},{"role":"assistant","content":정답}],"images":["blob:{src}/{row}"]}
        (이미지 = blob 참조, formrec.baselines.vlm.pf_train이 PaddleFormers 이미지 읽기를 패치)
- ovis: ft/{ds}/ovis/{train,val}.jsonl (base64 PNG, 공식 페이지 프롬프트) + epochs.txt · swift_args.txt (scripts/ovis_train.sh에 넘길 인자)
VLM 예산: --vlm_samples(기본 = 한국어 3에폭 ≈ 31만 샘플)를 넘지 않게 train을 고정 시드로 뽑고 에폭 수를 정한다(데이터셋 간 같은 학습량).
최종 실험 인자: xa 기본 · xh --save_splits 3 · xi --vlm_epochs 10 · xc --vlm_epochs 3
사용: env HW_DATASET=xa python -m formrec.baselines.vlm.ft_data vlm      (IO 작업: ovis base64)
"""
import base64
import io
import os
import shutil

import numpy as np
import yaml

from formrec.common import blob
from formrec.common import workspace as cfg
from formrec.baselines.ft_common import PVL_BASE, PVL_REGISTER, items, main, write_jsonl  # noqa: F401
from formrec.baselines.vlm.infer import OVIS_PROMPT

FONT = os.environ.get("FORMREC_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")   # 점검 그림 글꼴(없으면 기본 글꼴)


def input_check_vlm(D):
    """입력 1회 점검: train 표본 12장을 정답과 함께 격자로 저장(logs/checks/{ds}_vlm_ft_data.png), 빈 라벨 확인."""
    from PIL import Image, ImageDraw, ImageFont
    rng = np.random.default_rng(0); idx = rng.choice(len(D), size=min(12, len(D)), replace=False)
    ims = [Image.open(io.BytesIO(blob.read(D.ref.iloc[int(k)]))).convert("RGB") for k in idx]
    W_ = 1000; H_ = sum(min(80, im.height) + 34 for im in ims)
    g = Image.new("RGB", (W_, H_), "white"); d = ImageDraw.Draw(g); y = 0
    try:
        font = ImageFont.truetype(FONT, 18)
    except OSError:
        font = None
    for k, im in zip(idx, ims):
        s_ = min(1.0, 80 / im.height, W_ / im.width); t = im.resize((max(1, int(im.width * s_)), max(1, int(im.height * s_))))
        g.paste(t, (0, y)); d.text((4, y + t.height + 4), f"[{D.text.iloc[int(k)]}]  {im.size[0]}×{im.size[1]}", fill=(200, 60, 0), font=font)
        y += t.height + 34
    out = cfg.LOGS / "checks"; out.mkdir(parents=True, exist_ok=True); g.save(out / f"{cfg.DATASET}_vlm_ft_data.png")
    empty = int((D.text.str.len() == 0).sum()); ok = empty == 0
    print(f"[input_check] {cfg.DATASET}_vlm_ft_data: {'PASS' if ok else 'FAIL'} 빈 라벨 {empty} · 그림 logs/checks/{cfg.DATASET}_vlm_ft_data.png", flush=True)
    if not ok:
        raise SystemExit("VLM 학습 데이터 입력 점검 실패")


def sample_budget(tr, va, a):
    """(train, val, 에폭 수): 작은 데이터 = 여러 에폭, 큰 데이터 = 예산만큼만 뽑아 1에폭, val = 상한까지 표본(시드 23)."""
    rng = np.random.default_rng(23)
    ep = max(1, round(a.vlm_samples / len(tr))) if len(tr) * 1.5 < a.vlm_samples else 1
    if len(tr) * ep > a.vlm_samples * 1.5:
        tr = tr.iloc[np.sort(rng.choice(len(tr), a.vlm_samples, replace=False))].reset_index(drop=True)
    if a.val_max and len(va) > a.val_max:
        va = va.iloc[np.sort(rng.choice(len(va), a.val_max, replace=False))].reset_index(drop=True)
    return tr, va, ep


def pvl_row(ref, text):
    return {"messages": [{"role": "user", "content": "<image>OCR:"}, {"role": "assistant", "content": text}], "images": [ref]}


def ovis_row(png, text):
    return {"messages": [{"role": "user", "content": "<image>" + OVIS_PROMPT}, {"role": "assistant", "content": text}],
            "images": [base64.b64encode(png).decode()]}


def pvl_config(d, run, epochs, nsv=0, base=PVL_BASE):
    """기준 설정 → 이 데이터셋 PaddleFormers 설정(dict). 경로는 작업 공간 절대 경로로."""
    c = yaml.safe_load(open(base, encoding="utf-8"))
    c["model_name_or_path"] = str(cfg.ROOT / c["model_name_or_path"])
    c.update(train_dataset_path=str(d / "train.jsonl"), custom_register_path=str(d / "paddleocr_vl_v15_template.py"),
             num_train_epochs=epochs, do_eval=True, evaluation_strategy="epoch", save_strategy="epoch",
             eval_dataset_type="messages", eval_dataset_path=str(d / "val.jsonl"), eval_dataset_prob="1.0",
             output_dir=str(run), logging_dir=str(run / "vdl"))
    if nsv:                                                       # 큰 데이터 1회 순회: N등분 스텝마다 저장 · val loss
        c.update(evaluation_strategy="steps", save_strategy="steps", eval_steps=nsv, save_steps=nsv)
    return c


def swift_args(epochs, nsv=0):
    return f"--num_train_epochs {epochs}" + (f" --save_strategy steps --save_steps {nsv} --eval_strategy steps --eval_steps {nsv}" if nsv else " --eval_strategy epoch")


def build(a):
    tr, va = items("train"), items("val")
    tr, va, ep = sample_budget(tr, va, a)
    print(f"{cfg.DATASET} VLM: train {len(tr)} × {ep}에폭 = {len(tr) * ep} 샘플 · val {len(va)}", flush=True)
    input_check_vlm(tr)
    epochs = a.vlm_epochs or ep
    nsv = -(-len(tr) * epochs // (128 * a.save_splits)) if a.save_splits else 0     # 전역 배치 128 기준 저장 · 평가 간격(스텝)
    # PaddleOCR-VL
    d = cfg.FT / "pvl"; d.mkdir(parents=True, exist_ok=True)
    for s, D in (("train", tr), ("val", va)):
        write_jsonl(d / f"{s}.jsonl", (pvl_row(r, t) for r, t in zip(D.ref, D.text)))
    shutil.copy(PVL_REGISTER, d / "paddleocr_vl_v15_template.py")
    c = pvl_config(d, cfg.RUNS / "pvl_ft", epochs, nsv)
    cfg.atomic_write(d / "pvl16_full.yaml", lambda p: yaml.safe_dump(c, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False))
    # OvisOCR2
    d = cfg.FT / "ovis"; d.mkdir(parents=True, exist_ok=True)
    for s, D in (("train", tr), ("val", va)):
        write_jsonl(d / f"{s}.jsonl", (ovis_row(blob.read(r), t) for r, t in zip(D.ref, D.text)))
    (d / "epochs.txt").write_text(str(epochs))
    (d / "swift_args.txt").write_text(swift_args(epochs, nsv))
    print(f"DONE vlm {cfg.DATASET}: pvl/ovis train {len(tr)} val {len(va)} 에폭 {epochs}")


if __name__ == "__main__":
    main()
