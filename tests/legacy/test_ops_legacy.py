"""실험 계획 원본 대조: 새 plan.build() vs 원본 xplan.build() — 작업 이름 · 순서 · deps · after · gpus · disk_gb · stall · io 동일,
명령은 '원본 스크립트 → 새 모듈' 치환 규칙(RULES)을 적용하면 같아야 한다. 원본의 리포트 작업(*_report)은 공개 저장소에서 뺐다."""
import sys

import pytest

RULES = [
    ("envs/train/bin/torchrun --nproc_per_node 4 --master_port 29751 work/x_train.py",
     '"$FORMREC_PY_TRAIN" -m torch.distributed.run --nproc_per_node 4 --master_port 29751 -m formrec.retrieval.train'),
    ("envs/train/bin/torchrun --nproc_per_node 4 --master_port 29741 work/x_pretrain.py",
     '"$FORMREC_PY_TRAIN" -m torch.distributed.run --nproc_per_node 4 --master_port 29741 -m formrec.retrieval.pretrain'),
    ("envs/base/bin/python", '"$FORMREC_PY_BASE"'), ("envs/train/bin/python", '"$FORMREC_PY_TRAIN"'), ("envs/paddle/bin/python", '"$FORMREC_PY_PADDLE"'),
    ("envs/vllm/bin/python", '"$FORMREC_PY_VLLM"'), ("envs/pf/bin/python", '"$FORMREC_PY_PF"'),
    ("work/x_data.py", "-m formrec.data_prep.build"), ("work/x_ft_data.py", "-m formrec.baselines.vlm.ft_data"),
    ("work/pf_train.py", "-m formrec.baselines.vlm.pf_train"), ("work/vlm_ckpt.py", "-m formrec.baselines.vlm.ckpt"),
    ("work/g1_vlm.py", "-m formrec.baselines.vlm.infer"), ("work/ppocr_bn_select.py", "-m formrec.baselines.ppocr.bn_select"),
    ("work/g1_ppocr.py", "-m formrec.baselines.ppocr.infer"), ("work/x_base_eval.py", "-m formrec.baselines.evaluate"),
    ("work/x_eval.py", "-m formrec.retrieval.evaluate"), ("work/kie_summary.py", "-m formrec.retrieval.summary"),
    ("work/xk_vlm.py", "-m formrec.kie.vlm"), ("work/xk_eval.py", "-m formrec.kie.evaluate"), ("work/xk_oov.py", "-m formrec.kie.oov"),
    ("work/xk_order.py", "-m formrec.kie.order"), ("work/xkg_ocr.py", "-m formrec.kie_general.ocr"),
    ("bash work/ppocr_ft.sh", 'bash "$FORMREC_REPO"/scripts/ppocr_ft.sh'), ("bash work/ppocr_ft_post.sh", 'bash "$FORMREC_REPO"/scripts/ppocr_ft_post.sh'),
    ("bash work/ovis_train.sh", 'bash "$FORMREC_REPO"/scripts/ovis_train.sh'), ("bash work/x_smoke_clean.sh", 'bash "$FORMREC_REPO"/scripts/smoke_clean.sh'),
]
FIELDS = ("name", "deps", "after", "gpus", "disk_gb", "stall", "io")


def translate(cmd):
    for a, b in RULES:
        cmd = cmd.replace(a, b)
    return cmd


@pytest.mark.parametrize("smoke", [False, True])
def test_plan_matches_xplan(legacy_work, smoke):
    sys.path.insert(0, legacy_work)
    import xplan

    from formrec.ops import plan
    old = xplan.build(smoke=smoke)
    rep = {j["name"] for j in old if j["name"].endswith("_report")}
    assert len(rep) == 5
    old = [dict(j, after=[d for d in j["after"] if d not in rep]) for j in old if j["name"] not in rep]
    new = plan.build(smoke=smoke)
    assert len(new) == len(old)
    for a, b in zip(old, new):
        assert all(a[k] == b[k] for k in FIELDS), (a["name"], b["name"])
        assert translate(a["cmd"]) == b["cmd"], a["name"]


def test_plan_smoke_xc(legacy_work, monkeypatch):
    """원본 plan_smoke_xc(스모크 중 xc 단계만) = 새 plan_smoke + FORMREC_SMOKE_ONLY=S_xc"""
    sys.path.insert(0, legacy_work)
    import xplan

    from formrec.ops import jobsched
    old = [j for j in xplan.build(smoke=True) if j["name"].startswith("S_xc") and not j["name"].endswith("_report")]
    names = {j["name"] for j in old}
    monkeypatch.setenv("FORMREC_SMOKE_ONLY", "S_xc")
    _, new = jobsched.load_plan("formrec.ops.plan_smoke")
    assert [j["name"] for j in new] == [j["name"] for j in old]
    for a, b in zip(old, new):
        assert [d for d in a["deps"] if d in names] == b["deps"] and [d for d in a["after"] if d in names] == b["after"]
