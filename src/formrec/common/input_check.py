"""학습·평가 시작 전 입력 처리 1회 점검.

왜: 학습 · 추론 입력 처리가 어긋나는 문제(예: CTC 시간축 칸 수가 학습 40칸 ↔ 추론 120칸, 짧은 필드 칸 부족, BN 누적 통계 어긋남,
생성 결과 정리에서 숫자 · 공백 삭제)는 실제 모델 입력과 출력 모양을 한 번만 봐도 잡힌다.
점검 항목 (결과: logs/checks/{name}.json + 입력 그림 {name}_{train,eval}.png, 실패면 종료 코드 1):
  PP-OCR 학습 전 (ppocr --cfg): 공식 Train/Eval 데이터로더의 실제 배치 그림·라벨, 값 범위, 여백 비율,
      같은 입력에서 학습 모드와 추론 모드의 CTC 시간축 칸 수가 같은가, 짧은 필드의 글자당 칸 수(≥ 1.5)
  PP-OCR 평가 전 (ppocr --cfg --ckpt): 위 + 같은 train 배치에서 학습 모드 vs 추론 모드 정확도 차이(≤ 10%p), 필드 종류별
  생성 결과 정리 (check_normalize): 정리 후 문자열이 라벨 글자 집합을 지우지 않는가(숫자·공백 보존)
  제안 모델 (kie_batch / kie_eval_inputs): 레코드별 이미지 수 = 비지 않은 필드 수, 값 범위, 학습·평가 입력 크기 동일, 그림 저장
사용: HW_DATASET=xa CUDA_VISIBLE_DEVICES=0 KIE_PPOCR_POOL=prop python -m formrec.common.input_check ppocr --cfg ft/xa/ppocr/korean_v5_ft.yml --name ppocr_xa [--ckpt ...]
"""
import argparse
import json
import os
import sys

import numpy as np

from formrec.common.workspace import ROOT, THIRD_PARTY

OUT = os.path.join(ROOT, "logs/checks")


def _grid(imgs, labels, path, ncol=2, maxn=16):
    """imgs: (N, C, H, W) 정규화(−1~1) 또는 (N, H, W) 0~1 → PNG (라벨은 옆 txt)."""
    from PIL import Image
    x = np.asarray(imgs[:maxn], np.float32)
    if x.ndim == 4:
        x = x.mean(1)
    if x.min() < -0.01:
        x = (x + 1) / 2
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    n, h, w = x.shape
    rows = (n + ncol - 1) // ncol
    canvas = np.full((rows * (h + 6), ncol * (w + 6)), 128, np.uint8)
    for i in range(n):
        r, c = divmod(i, ncol)
        canvas[r * (h + 6):r * (h + 6) + h, c * (w + 6):c * (w + 6) + w] = x[i]
    Image.fromarray(canvas).save(path)
    with open(path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{i}: {t}" for i, t in enumerate(labels[:maxn])))


def _content_frac(x):
    """정규화 입력 (N, C, H, W)에서 여백(정확히 0)이 아닌 열의 비율."""
    col = np.abs(np.asarray(x)).sum((1, 2)) > 0
    return col.mean(1)


def _finish(name, res):
    os.makedirs(OUT, exist_ok=True)
    res["pass"] = not res.get("fail")
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(f"[input_check] {name}: {'PASS' if res['pass'] else 'FAIL'} " + "; ".join(res.get("fail", [])) + (" — 경고: " + "; ".join(res["warn"]) if res.get("warn") else ""), flush=True)
    return res


def field_kind(t):
    return f"addr{len(t.split(' '))}" if " " in t else ("birth" if t.isdigit() else "name")


# ------------------------------------------------------------------ PP-OCR
def ppocr(cfg_path, name, ckpt=None, gap_max=0.10):
    sys.path.insert(0, str(THIRD_PARTY / "PaddleOCR"))
    import logging
    import yaml
    from formrec.common import blob
    from formrec.baselines.ppocr import patch as ppocr_patch
    ppocr_patch.apply()
    import ppocr.data.simple_dataset as sd
    _l, _e = sd._load_image_bytes, sd._img_path_exists
    sd._load_image_bytes = lambda p: blob.read(p) if blob.is_ref(p) else _l(p)
    sd._img_path_exists = lambda p: True if blob.is_ref(p) else _e(p)
    import paddle
    from ppocr.data import build_dataloader
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    c = yaml.safe_load(open(cfg_path))
    for m_ in ("Train", "Eval"):
        c[m_]["loader"]["num_workers"] = 0
    wid = next((t["RecResizeImg"]["image_shape"][-1] for t in c["Eval"]["dataset"]["transforms"] if "RecResizeImg" in t), 960)
    c["Eval"]["loader"]["batch_size_per_card"] = 64 if wid <= 960 else 16      # 폭 1920(IAM)에서 64장 점검은 GPU 메모리 초과
    post = build_post_process(c["PostProcess"], c["Global"]); ch = len(post.character)
    for h in c["Architecture"]["Head"]["head_list"]:
        if "CTCHead" in h:
            h["CTCHead"]["out_channels"] = ch
        if "NRTRHead" in h:
            h["NRTRHead"]["out_channels"] = ch + 3
    c["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": ch, "NRTRLabelDecode": ch + 3}
    dev = paddle.set_device("gpu" if paddle.is_compiled_with_cuda() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    log = logging.getLogger("input_check")
    res = {"name": name, "cfg": cfg_path, "pool": os.environ.get("KIE_PPOCR_POOL", ""), "fail": [], "warn": []}
    batches = {}
    for mode in ("Train", "Eval"):
        b = next(iter(build_dataloader(c, mode, dev, log)))
        x = b[0].numpy(); lab = [t[0] for t in post(np.zeros((len(x), 1, ch), np.float32), b[1].numpy())[1]]
        batches[mode] = (b, lab)
        cf = _content_frac(x)
        res[mode] = dict(shape=list(x.shape), min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                         content_frac_min=float(cf.min()), content_frac_med=float(np.median(cf)), labels=lab[:6])
        os.makedirs(OUT, exist_ok=True)
        _grid(x, lab, os.path.join(OUT, f"{name}_{mode.lower()}.png"))
    m = build_model(c["Architecture"])
    if ckpt:
        m.set_state_dict(paddle.load(ckpt if ckpt.endswith(".pdparams") else ckpt + ".pdparams"))
    for mode in ("Train", "Eval"):                               # 같은 입력의 시간축 칸 수: 학습 모드 vs 추론 모드
        x = batches[mode][0][0]
        m.backbone.train(); tt = m.backbone(x).shape[-1]
        m.backbone.eval(); te = m.backbone(x).shape[-1]
        W = x.shape[-1]
        res[mode].update(T_train=int(tt), T_eval=int(te), px_per_step_train=W / tt)
        if tt != te:
            res["fail"].append(f"{mode} 입력 폭 {W}: 학습 {tt}칸 ≠ 추론 {te}칸 (ppocr_patch prop/fix40 필요)")
        lab = batches[mode][1]; cf = _content_frac(x.numpy())
        spc = [cf[i] * tt / max(1, len(lab[i])) for i in range(len(lab))]
        res[mode]["steps_per_char_min"] = float(min(spc)); res[mode]["steps_per_char_p10"] = float(np.percentile(spc, 10))
        if np.percentile(spc, 10) < 1.5:
            res["fail"].append(f"{mode}: 글자당 칸 수 하위 10% = {np.percentile(spc, 10):.2f} < 1.5 (짧은/긴 필드가 칸 부족 — 폭·풀링 확인)")
    if ckpt:                                                     # 같은 train 배치: 학습 모드 vs 추론 모드 정확도
        # 비교 배치 = 증강 없는 val 배치(추론과 같은 높이·전처리). train 배치는 증강(색 반전·여러 높이)이 섞여
        # 추론 모드(BN 누적 통계)가 원래 볼 일이 없는 입력이라 비교가 무의미하다(반전된 짧은 필드에서 거짓 실패).
        b, lab = batches["Eval"]
        acc = {}
        for mode in ("train", "eval"):
            (m.train() if mode == "train" else m.eval())
            with paddle.no_grad():
                o = m(b[0], data=b[1:]) if mode == "train" else m(b[0])
            o = o["ctc"] if isinstance(o, dict) else o
            pr = [p[0] for p in post(o.numpy())]
            acc[mode] = float(np.mean([p == t for p, t in zip(pr, lab)]))
            byk = {}
            for p, t in zip(pr, lab):
                v = byk.setdefault(field_kind(t), [0, 0]); v[0] += p == t; v[1] += 1
            res[f"acc_{mode}_by_field"] = {k: v[0] / v[1] for k, v in sorted(byk.items())}
            res[f"examples_{mode}"] = list(zip(pr[:6], lab[:6]))
        res["acc_train_mode"], res["acc_eval_mode"] = acc["train"], acc["eval"]
        if acc["train"] - acc["eval"] > gap_max:
            res["fail"].append(f"같은 val 배치 정확도: 학습 모드 {acc['train']:.3f} vs 추론 모드 {acc['eval']:.3f} (차이 > {gap_max:.2f}) — BN 통계·풀링 확인")
    return _finish(name, res)


# ------------------------------------------------------------------ 생성 결과 정리
def check_normalize(labels, normalize, name):
    """라벨에 정리 함수를 적용해 그대로 남는지 (정답 문자열이 정리 과정에서 망가지면 생성형이 부당하게 오답 처리된다)."""
    labels = list(labels)
    bad = [(t, normalize(t)) for t in labels if normalize(t) != t]
    res = {"name": name, "n": len(labels), "changed": len(bad), "examples": bad[:8], "fail": [], "warn": []}
    if len(bad) > max(0.001 * len(labels), 2):        # 표본이 작을 때(스모크 100개) 희귀 글자 1개로 실패하지 않게
        res["fail"].append(f"정리 함수가 정답 라벨 {len(bad):,}/{len(labels):,}개를 바꿈 (예: {bad[:3]})")
    return _finish(name, res)


# ------------------------------------------------------------------ 제안 모델
def kie_batch(x, mask, texts, name, sep="␞"):
    """x: (B, M, 1, H, W) 학습 입력, mask: (B, M) 실제 이미지, texts: 레코드 문자열."""
    x = np.asarray(x); mask = np.asarray(mask)
    res = {"name": name, "shape": list(x.shape), "min": float(x[mask].min()), "max": float(x[mask].max()), "fail": [], "warn": []}
    nimg = mask.sum(1); nf = np.array([sum(1 for f in t.split(sep) if f) for t in texts])
    res["img_count_mismatch"] = int((nimg != nf).sum())
    if (nimg != nf).any():
        res["fail"].append(f"레코드 {int((nimg != nf).sum())}개: 이미지 수 ≠ 비지 않은 필드 수")
    if res["min"] < -1.01 or res["max"] > 1.01:
        res["fail"].append(f"값 범위 {res['min']:.2f}~{res['max']:.2f} (−1~1 기대)")
    ims, labs = [], []
    for i in range(min(6, len(x))):
        for j in range(x.shape[1]):
            if mask[i, j]:
                ims.append(x[i, j, 0]); labs.append(f"rec{i} img{j}: {texts[i]}")
    os.makedirs(OUT, exist_ok=True)
    _grid(np.stack(ims), labs, os.path.join(OUT, f"{name}.png"), ncol=1, maxn=18)
    return _finish(name, res)


def kie_eval_inputs(x, texts, name, train_shape=None):
    x = np.asarray(x)
    res = {"name": name, "shape": list(x.shape), "min": float(x.min()), "max": float(x.max()), "fail": [], "warn": []}
    if train_shape is not None and list(x.shape[-2:]) != list(train_shape[-2:]):
        res["fail"].append(f"평가 입력 크기 {list(x.shape[-2:])} ≠ 학습 {list(train_shape[-2:])}")
    os.makedirs(OUT, exist_ok=True)
    _grid(x.reshape(-1, x.shape[-2], x.shape[-1])[:12], [t for t in texts for _ in range(x.shape[1] if x.ndim == 5 else 1)], os.path.join(OUT, f"{name}.png"), ncol=1, maxn=12)
    return _finish(name, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ppocr"])
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--warn_only", action="store_true")
    a = ap.parse_args()
    r = ppocr(a.cfg, a.name, a.ckpt or None)
    sys.exit(0 if r["pass"] or a.warn_only else 1)


if __name__ == "__main__":
    main()
