"""비교군 원본 대조(LEGACY_WORK · FORMREC_WORKSPACE = 원본 산출물이 있는 작업 공간, 읽기 전용).

- 채점: 저장된 생성 결과(results/{ds}/g1) → 새 evaluate.score_levels vs 원본 점수 파일(results/{ds}/scores/G1-*)
- 정리: 저장된 raw → 새 정리 함수 vs 저장된 pred · score
- FT 데이터: 새 함수로 만든 PP-OCR 목록 · 설정, PaddleOCR-VL jsonl · 설정 vs 원본 ft/{ds}/
- 선정: 학습 로그의 val loss 해석 vs 원본 runs/{ds}/pvl_ft/best.json
- PP-OCR 전처리(paddle 환경이 있을 때): 새 bn_select.load_imgs vs 원본 ppocr_bn_select.load_imgs
데이터셋마다 HW_DATASET이 다르므로 하위 프로세스로 돈다.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[2] / "src")
DSS = ("xa", "xh", "xi", "xc")


def run(code, ws, ds, py=sys.executable, **env):
    e = dict(os.environ, PYTHONPATH=SRC, FORMREC_WORKSPACE=ws, HW_DATASET=ds, CUDA_VISIBLE_DEVICES="", **env)
    r = subprocess.run([py, "-c", code], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


SCORES = """
import pandas as pd
from formrec.common import workspace as cfg
from formrec.common.levels import level_table
from formrec.baselines.evaluate import load_generation, score_levels
bad = n = 0
for tag in ("ppocrv5ft", "pvlft", "ovisft"):
    for sub in ("tune", "test"):
        if not (cfg.RESULTS / f"g1/{tag}_{sub}.parquet").exists(): continue
        out = score_levels(cfg.DATASET, level_table(cfg.DATASET, sub), *load_generation(cfg.RESULTS / f"g1/{tag}_{sub}.parquet"))
        ref = {p.name.split("__")[1] for p in (cfg.RESULTS / "scores").glob(f"G1-{tag}__*__{sub}.parquet")}
        bad += ref != set(out)
        for lv, o in out.items():
            n += 1
            try: pd.testing.assert_frame_equal(o, pd.read_parquet(cfg.RESULTS / f"scores/G1-{tag}__{lv}__{sub}.parquet"), check_exact=True)
            except AssertionError: bad += 1
print("FILES", n, "MISMATCH", bad)
"""

NORM = """
import numpy as np, pandas as pd
from formrec.common import workspace as cfg
from formrec.baselines.normalize import allowed_chars, make_normalizer
allowed = allowed_chars(cfg.DATASET, cfg.queries("train").text); bad = 0
for p in sorted((cfg.RESULTS / "g1").glob("*.parquet")):
    g = pd.read_parquet(p); vlm = "lp_mean" in g
    bad += int((g.raw.map(make_normalizer(cfg.DATASET, allowed, markup=vlm)) != g.pred).sum())
    if vlm: bad += int((np.exp(g.lp_mean.fillna(-1e9).clip(upper=0)) != g.score).sum())
print("MISMATCH", bad)
"""

FT = """
import json, yaml
from formrec.common import workspace as cfg
from formrec.baselines import ft_common as C
from formrec.baselines.ppocr import ft_data as P
from formrec.baselines.vlm import ft_data as V
PP = {"xa": "", "xc": "--max_len 48 --width 1280", "xh": "--epochs 3 --max_len 80 --width 640", "xi": "--epochs 100 --max_len 100 --width 2560 --bs 8 --save_every 5"}
VL = {"xa": "", "xh": "--save_splits 3", "xi": "--vlm_epochs 10", "xc": "--vlm_epochs 3"}
ds = cfg.DATASET; bad = 0; n = {}
for s in ("train", "val", "tune", "test"):
    D = C.items(s); n[s] = len(D)
    bad += "".join(f"{r}\\t{t}\\n" for r, t in zip(D.ref, D.text)) != open(cfg.FT / f"ppocr/{s}_list.txt", encoding="utf-8").read()
fn = "korean_v5_ft.yml" if cfg.LANG == "ko" else "en_v5_ft.yml"
c = P.config(C.parser().parse_args(["ppocr"] + PP[ds].split()), n["train"], cfg.FT / "ppocr")
bad += yaml.safe_dump(c, allow_unicode=True, sort_keys=False) != open(cfg.FT / f"ppocr/{fn}", encoding="utf-8").read()
a = C.parser().parse_args(["vlm"] + VL[ds].split())
tr, va, ep = V.sample_budget(C.items("train"), C.items("val"), a); epochs = a.vlm_epochs or ep
nsv = -(-len(tr) * epochs // (128 * a.save_splits)) if a.save_splits else 0
for s, D in (("train", tr), ("val", va)):
    bad += [json.dumps(V.pvl_row(r, t), ensure_ascii=False) + "\\n" for r, t in zip(D.ref, D.text)] != open(cfg.FT / f"pvl/{s}.jsonl", encoding="utf-8").readlines()
c = V.pvl_config(cfg.FT / "pvl", cfg.RUNS / "pvl_ft", epochs, nsv)
bad += yaml.safe_dump(c, allow_unicode=True, sort_keys=False) != open(cfg.FT / "pvl/pvl16_full.yaml", encoding="utf-8").read()
bad += V.swift_args(epochs, nsv) != (cfg.FT / "ovis/swift_args.txt").read_text()
print("MISMATCH", bad)
"""

LOSS = """
import json, re
from formrec.common import workspace as cfg
from formrec.baselines.vlm import ckpt
ref = json.load(open(cfg.RUNS / "pvl_ft/best.json"))["val_loss"]; rd = str(cfg.RUNS / "pvl_ft"); got = None
for lg in sorted(cfg.LOGS.glob("pdist/*/workerlog.0"), key=lambda p: -p.stat().st_mtime):
    t = lg.read_text(errors="ignore")
    if re.search(r"output_dir\\s*:\\s*" + re.escape(rd) + r"\\b", t) or re.search(r"output_dir\\s*:\\s*" + re.escape(rd.replace(str(cfg.ROOT) + "/", "")) + r"\\b", t):
        got = ckpt.losses_from_log(t, len(ref)); break
print("SKIP" if got is None else ("MISMATCH %d" % ({str(k): v for k, v in got.items()} != ref)))
"""

BN = """
import os, sys
import numpy as np
from formrec.common import workspace as cfg
from formrec.baselines.ppocr import bn_select as new
sys.path.append(os.environ["LEGACY_WORK"])
import ppocr_bn_select as old
lst = [l.split("\\t")[0] for l, _ in zip(open(cfg.FT / "ppocr/val_list.txt"), range(32))]
shape = {"xa": [3, 48, 960], "xc": [3, 48, 1280], "xh": [3, 48, 640], "xi": [3, 48, 2560]}[cfg.DATASET]
a, b = new.load_imgs(lst, shape), old.load_imgs(lst, shape)
print("MISMATCH", int(a.shape != b.shape or not np.array_equal(a, b)))
"""


@pytest.mark.parametrize("ds", DSS)
def test_scores_match_original(legacy_work, workspace_root, ds):
    out = run(SCORES, workspace_root, ds)
    assert "MISMATCH 0" in out and "FILES 0" not in out, out


@pytest.mark.parametrize("ds", DSS)
def test_normalize_matches_saved_predictions(legacy_work, workspace_root, ds):
    assert "MISMATCH 0" in run(NORM, workspace_root, ds)


@pytest.mark.parametrize("ds", DSS)
def test_ft_data_matches_original(legacy_work, workspace_root, ds):
    pytest.importorskip("yaml")
    assert "MISMATCH 0" in run(FT, workspace_root, ds)


@pytest.mark.parametrize("ds", DSS)
def test_vlm_val_loss_parse(legacy_work, workspace_root, ds):
    out = run(LOSS, workspace_root, ds)
    if "SKIP" in out:
        pytest.skip("학습 로그 없음")
    assert "MISMATCH 0" in out


@pytest.mark.parametrize("ds", DSS)
def test_ppocr_preprocess_matches_original(legacy_work, workspace_root, ds):
    py = os.environ.get("FORMREC_PY_PADDLE", os.path.join(workspace_root, "envs/paddle/bin/python"))
    if not os.path.exists(py):
        pytest.skip("paddle 환경 없음")
    xs = {"KIE_PPOCR_XSTRETCH": "2"} if ds in ("xh", "xi") else {}
    assert "MISMATCH 0" in run(BN, workspace_root, ds, py=py, LEGACY_WORK=legacy_work, PYTHONDONTWRITEBYTECODE="1", **xs)
