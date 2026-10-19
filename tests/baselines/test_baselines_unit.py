"""비교군 단위 테스트 (작업 공간 · GPU 불필요)."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formrec.baselines import evaluate, ft_common, normalize
from formrec.baselines.ppocr import patch
from formrec.baselines.vlm import ckpt, infer, pf_train

REPO = Path(__file__).resolve().parents[2]


def test_normalizer_korean_keeps_digits_spaces_and_rare_hangul():
    allowed = normalize.allowed_chars("xa", ["김민준", "900101", "서울특별시 강남구"])
    norm = normalize.make_normalizer("xa", allowed)
    assert norm("  샨  900101!! ") == "샨 900101"          # train에 없는 희귀 음절도 남김, 기호 제거, 공백 하나
    assert norm("abc") == ""


def test_normalizer_hana_lowercase_and_vlm_markup():
    allowed = normalize.allowed_chars("xh", ["dahl carl"])
    norm = normalize.make_normalizer("xh", allowed, markup=True)
    assert norm('<img src="x.jpg" />Dahl\n<b>Carl</b>') == "dahl carl"
    assert normalize.make_normalizer("xh", allowed)("<b>Dahl</b>") == "dahl"   # PP-OCR 경로는 마크업을 따로 지우지 않는다(글자 거르기만)


def test_xstretch_identity_when_off():
    x = np.zeros((4, 5, 3), np.uint8)
    if patch.XS == 1:
        assert patch.xstretch(x) is x
    assert patch.xstretch(None) is None


def test_score_levels_single_and_column_drop():
    L = pd.DataFrame(dict(level=["I1", "I1"], pid=["a", "b"], items=[["a"], ["b"]], text=["x y", "z"]))
    out = evaluate.score_levels("xi", L, {"a": "x y", "b": "q"}, {"a": 0.9, "b": 0.4}, {"a": 0.5, "b": 0.1})
    o = out["I1"]
    assert o.correct.tolist() == [True, False] and o.conf.tolist() == [0.9, 0.4]
    assert o.open_conf.tolist() == [-1e9, 0.4] and o.conf_min.tolist() == [0.5, 0.1]
    sep = evaluate.SEP
    L = pd.DataFrame(dict(level=["F3"], pid=["p"], items=[["n", "b", "a"]], text=[sep.join(["김", "900101", "서울"])]))
    out = evaluate.score_levels("xa", L, {"n": "김", "b": "900102", "a": "서울"}, {"n": 0.9, "b": 0.2, "a": 0.8})
    assert not out["F3"].correct[0]
    assert out["F3#d1"].correct[0] and out["F3#d1"].conf[0] == 0.8 and out["F3#d1"].text[0] == sep.join(["김", "", "서울"])
    assert "conf_min" not in out["F3"]


def test_ft_parser_matches_plan_args():
    a = ft_common.parser().parse_args("vlm --save_splits 3".split())
    assert (a.cmd, a.save_splits, a.vlm_samples, a.val_max, a.vlm_epochs) == ("vlm", 3, 308247, 20000, 0)
    a = ft_common.parser().parse_args("ppocr --epochs 100 --max_len 100 --width 2560 --bs 8 --save_every 5".split())
    assert (a.epochs, a.max_len, a.width, a.bs, a.save_every) == (100, 100, 2560, 8, 5)


def test_ppocr_config_from_template():
    yaml = pytest.importorskip("yaml")
    from formrec.baselines.ppocr import ft_data
    a = ft_common.parser().parse_args("ppocr --epochs 3 --max_len 80 --width 640 --bs 8".split())
    c = ft_data.config(a, 1000, Path("ft/xx/ppocr"))
    assert c["Global"]["epoch_num"] == 3 and c["Global"]["max_text_length"] == 80 and c["Global"]["save_epoch_step"] == 1
    assert c["Global"]["eval_batch_step"] == [10 ** 9, 10 ** 9] and c["Optimizer"]["lr"]["warmup_epoch"] == 1
    assert c["Optimizer"]["lr"]["learning_rate"] == pytest.approx(2.5e-5) and c["Train"]["sampler"]["scales"][0] == [640, 32]
    assert c["Eval"]["dataset"]["label_file_list"] == ["ft/xx/ppocr/val_list.txt"]
    base = yaml.safe_load(open(ft_common.PPOCR_BASE))
    assert base["Global"]["max_text_length"] == 40 and base["Train"]["loader"]["batch_size_per_card"] == 48


def test_vlm_ft_rows_and_budget():
    pytest.importorskip("yaml")
    from formrec.baselines.vlm import ft_data
    assert ft_data.pvl_row("blob:all/3", "abc") == {"messages": [{"role": "user", "content": "<image>OCR:"}, {"role": "assistant", "content": "abc"}],
                                                   "images": ["blob:all/3"]}
    r = ft_data.ovis_row(b"\x89PNG", "abc")
    assert r["messages"][0]["content"].startswith("<image>\nExtract all readable") and r["images"] == ["iVBORw=="]
    a = ft_common.parser().parse_args(["vlm"])
    tr, va, ep = ft_data.sample_budget(pd.DataFrame(dict(x=range(1000))), pd.DataFrame(dict(x=range(10))), a)
    assert ep == 308 and len(tr) == 1000 and len(va) == 10                      # 작은 데이터: 여러 에폭
    a.vlm_samples, a.val_max = 100, 5
    tr, va, ep = ft_data.sample_budget(pd.DataFrame(dict(x=range(1000))), pd.DataFrame(dict(x=range(10))), a)
    assert ep == 1 and len(tr) == 100 and tr.x.is_monotonic_increasing and len(va) == 5
    assert ft_data.swift_args(1, 803) == "--num_train_epochs 1 --save_strategy steps --save_steps 803 --eval_strategy steps --eval_steps 803"
    assert ft_data.swift_args(3) == "--num_train_epochs 3 --eval_strategy epoch"
    c = ft_data.pvl_config(Path("/w/ft/xa/pvl"), Path("/w/runs/xa/pvl_ft"), 3)
    assert c["model_name_or_path"].endswith("models/PaddleOCR-VL-1.6") and Path(c["model_name_or_path"]).is_absolute()
    assert c["do_eval"] and c["evaluation_strategy"] == "epoch" and c["output_dir"] == "/w/runs/xa/pvl_ft"


def test_vlm_infer_settings():
    a = infer.parse_args("--model ovis --subset tune --model_path runs/xa/ovis_ft/export/best --out_name ovisft_tune".split())
    assert infer.out_name(a) == "ovisft_tune"
    a = infer.parse_args("--model paddlevl --subset test --limit 100".split())
    assert infer.out_name(a) == "paddlevl_test_n100"
    kw, sp, chat = infer.llm_kwargs("paddlevl", "m")
    assert kw["max_model_len"] == 16384 and kw["trust_remote_code"] and not kw["enable_prefix_caching"] and sp["temperature"] == 0.0 and chat == {}
    kw, sp, chat = infer.llm_kwargs("ovis", "m")
    assert kw["gdn_prefill_backend"] == "triton" and chat["mm_processor_kwargs"]["images_kwargs"]["max_pixels"] == 2880 * 2880

    class L:
        def __init__(self, v): self.logprob = v
    assert infer.logprob_summary([{1: L(-1.0)}, {2: L(-3.0)}, {3: L(float("-inf"))}, None]) == (-2.0, -3.0)
    m, n = infer.logprob_summary([])
    assert np.isnan(m) and np.isnan(n)


def test_ckpt_losses(tmp_path, monkeypatch):
    assert ckpt.losses_from_log("x eval_loss: 0.5 y eval_loss: 0.25\n eval_loss: 1e-1", 2) == {1: 0.5, 2: 0.25}
    import json
    for s in (10, 20, 30):
        (tmp_path / f"checkpoint-{s}").mkdir()
    hist = [{"step": 10, "eval_loss": 0.3}, {"step": 20, "eval_loss": 0.2}, {"step": 30, "eval_loss": 0.25}, {"step": 30, "loss": 1.0}]
    json.dump({"log_history": hist}, open(tmp_path / "checkpoint-30/trainer_state.json", "w"))
    monkeypatch.setattr(ckpt, "run_dir", lambda kind: tmp_path)
    L = ckpt.eval_losses("ovis")
    assert L == {1: 0.3, 2: 0.2, 3: 0.25} and min(L, key=lambda e: (L[e], e)) == 2


def test_pf_train_ref_and_cmd():
    assert pf_train.norm_ref("/w/ft/xa/pvl/blob:train/5") == "blob:train/5" and pf_train.norm_ref("a.png") == "a.png"
    cmd = pf_train.launch_cmd(["ft/xa/pvl/pvl16_full.yaml", "max_steps=20"], {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert cmd[1:3] == ["-m", "paddle.distributed.launch"] and cmd[-3:] == ["train", "ft/xa/pvl/pvl16_full.yaml", "max_steps=20"]
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_bn_select_args_and_kind():
    from formrec.baselines.ppocr import bn_select
    a = bn_select.parse_args("--run runs/xa/ppocrv5_ft --cfg ft/xa/ppocr/korean_v5_ft.yml --select loss --last 0".split())
    assert (a.select, a.last, a.calib, a.bs) == ("loss", 0, 8192, 64)
    assert [bn_select.field_kind(t) for t in ("김민준", "900101", "서울 강남구 역삼로")] == ["name", "birth", "addr3"]


@pytest.mark.parametrize("name", ["ppocr_ft.sh", "ppocr_ft_post.sh", "ovis_train.sh", "setup_third_party.sh", "setup_envs.sh"])
def test_scripts_syntax(name):
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / name)]).returncode == 0
