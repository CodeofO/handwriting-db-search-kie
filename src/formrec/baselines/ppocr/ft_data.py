"""PP-OCRv5 파인튜닝 데이터: ft/{ds}/ppocr/{train,val,tune,test}_list.txt + dict.txt + {korean,en}_v5_ft.yml.

기준 설정 = configs/ppocr/korean_v5_ft_base.yml(비례 칸 수 재학습용, 폭 960 · 최대 40자 · GPU당 48 · 20에폭)에서 경로 · 에폭 · 폭 · 길이 · 배치만 바꾼다.
에폭마다 가중치 저장(save_epoch_step) → bn_select가 BN 재추정 후 val CTC loss(선정) · val 정확도(부록)를 잰다.
학습 중 공식 평가는 끔(BN 누적 통계 문제로 무의미).
라벨 목록 = "blob:{src}/{row}\\t정답" (공식 SimpleDataSet/MultiScaleDataSet 형식), 사전 = 공식 추론 모델 inference.yml의 character_dict.
데이터셋별 인자(최종 실험): xa 기본 · xc --max_len 48 --width 1280 · xh --epochs 3 --max_len 80 --width 640 ·
  xi --epochs 100 --max_len 100 --width 2560 --bs 8 --save_every 5
사용: env HW_DATASET=xa python -m formrec.baselines.ppocr.ft_data ppocr [--epochs 20 --width 960 ...]
"""
import yaml

from formrec.common import workspace as cfg
from formrec.baselines.ft_common import GPUS, PPOCR_BASE, items, main  # noqa: F401


def model_name():
    return "korean_PP-OCRv5_mobile_rec" if cfg.LANG == "ko" else "en_PP-OCRv5_mobile_rec"


def config(a, n_train, out):
    """기준 설정 → 이 데이터셋 설정(dict)."""
    c = yaml.safe_load(open(PPOCR_BASE, encoding="utf-8"))
    name = model_name(); md = cfg.MODELS / name
    run = cfg.RUNS / "ppocrv5_ft"
    g = c["Global"]
    g["model_name"] = name                               # en 모델에 korean 이름이 남으면 추론기가 거부
    g.update(epoch_num=a.epochs, save_model_dir=str(run), save_res_path=str(run / "predicts.txt"), save_epoch_step=a.save_every,
             eval_batch_step=[10 ** 9, 10 ** 9], pretrained_model=str(md / f"{name}_pretrained"), character_dict_path=str(out / "dict.txt"))
    if a.bs:                                              # 넓은 입력: 메모리 → 배치 축소, 학습률 1e-4 × 전체 배치/128
        c["Train"]["sampler"]["first_bs"] = a.bs; c["Train"]["loader"]["batch_size_per_card"] = a.bs
        c["Optimizer"]["lr"]["learning_rate"] = 1e-4 * GPUS * a.bs / 128
    c["Optimizer"]["lr"]["warmup_epoch"] = min(5, max(1, a.epochs // 4))     # 공식 5에폭 워밍업 — 짧은 학습은 전체의 ¼ 이하
    if a.max_len:
        g["max_text_length"] = a.max_len
        for lh in c["Loss"]["loss_config_list"]:
            for v in lh.values():
                if isinstance(v, dict) and "max_text_length" in v:
                    v["max_text_length"] = a.max_len
        for d in c["Architecture"]["Head"]["head_list"]:
            for v in d.values():
                if isinstance(v, dict) and "max_text_length" in v:
                    v["max_text_length"] = a.max_len
        for t in c["Train"]["dataset"]["transforms"]:
            for v in t.values():
                if isinstance(v, dict) and "max_text_length" in v:
                    v["max_text_length"] = a.max_len
    if a.width and a.width != 960:                       # 입력 폭. 칸 수 = 폭/8 (비례 칸 수 패치)
        g["d2s_train_image_shape"] = [3, 48, a.width]
        for t in c["Train"]["dataset"]["transforms"]:
            for v in t.values():
                if isinstance(v, dict) and "image_shape" in v:
                    v["image_shape"] = [48, a.width, 3]
        c["Train"]["sampler"]["scales"] = [[a.width, 32], [a.width, 48], [a.width, 64]]
        for t in c["Eval"]["dataset"]["transforms"]:
            for v in t.values():
                if isinstance(v, dict) and "image_shape" in v:
                    v["image_shape"] = [3, 48, a.width]
    c["Train"]["dataset"]["label_file_list"] = [str(out / "train_list.txt")]
    c["Eval"]["dataset"]["label_file_list"] = [str(out / "val_list.txt")]
    return c


def build(a):
    out = cfg.FT / "ppocr"; out.mkdir(parents=True, exist_ok=True)
    chars = yaml.safe_load(open(cfg.MODELS / model_name() / "infer/inference.yml", encoding="utf-8"))["PostProcess"]["character_dict"]
    (out / "dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    cs = set(chars) | {" "}
    n = {}
    for s in ("train", "val", "tune", "test"):
        D = items(s)
        bad = sum(any(c not in cs for c in t) for t in D.text)
        if s == "train" and bad:
            print(f"경고: train 정답 중 사전에 없는 글자 포함 {bad}개", flush=True)
        cfg.atomic_write(out / f"{s}_list.txt", lambda p, D=D: open(p, "w", encoding="utf-8").write("".join(f"{r}\t{t}\n" for r, t in zip(D.ref, D.text))))
        n[s] = len(D)
    c = config(a, n["train"], out)
    spe = n["train"] // (GPUS * c["Train"]["loader"]["batch_size_per_card"])
    fn = "korean_v5_ft.yml" if cfg.LANG == "ko" else "en_v5_ft.yml"
    cfg.atomic_write(out / fn, lambda p: yaml.safe_dump(c, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False))
    print(f"DONE ppocr {cfg.DATASET}: {n} 사전 {len(chars)} 에폭 {a.epochs} × {spe}스텝 · 설정 {out / fn}")


if __name__ == "__main__":
    main()
