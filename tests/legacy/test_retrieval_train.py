"""제안 모델 원본 대조 2 — 학습(레코드 표집 · 합성 음성 · 증강 · batch_loss · 기울기, 사전학습 표집 · val_loss). CPU · gloo 1프로세스.

원본 x_train.py / x_pretrain.py의 main() 안 닫힘 함수(build · batch_loss · val_loss · producer 본문)를 ast로 꺼내 같은 입력 · 같은 시드로 부른다.
"""
import ast
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
TOL = 1e-6


def legacy(legacy_work):
    os.environ.setdefault("HW_DATASET", "xa")
    sys.dont_write_bytecode = True
    if legacy_work not in sys.path:
        sys.path.insert(0, legacy_work)
    import g4_train, kie_model3, kie_train, kie_train3, x_common  # noqa: E401
    return g4_train, kie_model3, kie_train, kie_train3, x_common


def main_closures(path, names, ns):
    tree = ast.parse(open(path).read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    for n in ast.walk(main):
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])), path, "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def pg():
    import torch.distributed as dist
    for k, v in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29981"), ("RANK", "0"), ("WORLD_SIZE", "1")):
        os.environ.setdefault(k, v)
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    yield dist
    dist.destroy_process_group()


@pytest.mark.parametrize("ds", ["xa", "xi"])
def test_sampling_and_negatives(legacy_work, workspace_root, ds):
    from formrec.retrieval import data as X
    LG, L3, LK, LT3, LX = legacy(legacy_work)
    n, o = X.XData(ds, "train"), LX.XData(ds, "train")
    assert n.pids == o.pids and n.by == o.by
    rn, ro = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(20):
        assert n.sample(rn, 64, 1, 0.15) == o.sample(ro, 64, 1, 0.15)
    pn, po = X.char_pools(n.text), LX.char_pools(o.text)
    tn, to = X.token_vocab(n), LX.token_vocab_x(o)
    assert pn == po and tn.keys() == to.keys() and all((tn[k] == to[k]).all() for k in tn)
    rn, ro = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(20):
        recs = n.sample(rn, 64, 1, 0.15); o.sample(ro, 64, 1, 0.15)
        txt = [n.record(p, f)[0] for p, f in recs]; keys = [X.field_kinds(f) for _, f in recs]
        a = X.synth_negatives(txt, keys, rn, pn, tn); b = []
        for i, t in enumerate(txt):
            n1 = LX.perturb_char_x(t, ro, po); n2 = LK.swap_field(t, txt[ro.integers(len(txt))], ro); n3 = LK.perturb_token(t, keys[i], ro, to)
            b += [n1 or t + "x", n2 or (LX.perturb_char_x(t, ro, po) or t + "y"), n3 or LX.perturb_char_x(t, ro, po) or t + "z"]
        assert a == b


@pytest.mark.parametrize("train", [True, False])
def test_build_batch_loss_grad(legacy_work, workspace_root, pg, train):
    import torch.distributed.nn as dnn
    import torch.nn.functional as F
    from formrec.common.workspace import ROOT
    from formrec.retrieval import data as X
    from formrec.retrieval import train as T
    LG, L3, LK, LT3, LX = legacy(legacy_work)
    path = ROOT / "runs/x/xi_prop/final.pt"
    if not path.exists():
        pytest.skip(f"{path} 없음")
    torch.set_num_threads(8)
    a = T.parse_args(["--name", "t", "--data", "xi", "--cpu", "--bs", "4"])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tok, otok = X.load_tok(), LX.load_tok()
    mn = T.make_model(a, len(tok.vocab), tok.maxu); mn.load_state_dict(ck["model"], strict=True)
    mo = L3.RecordEncoder3(len(tok.vocab), tok.maxu, 16, 3, 64, 1280, d=384, dim=64, layers=6, vfeat=14, read_layers=3); mo.load_state_dict(ck["model"], strict=True)
    trn, tro = X.XData("xi", "train"), LX.XData("xi", "train")
    pn, po = X.char_pools(trn.text), LX.char_pools(tro.text); tvn, tvo = X.token_vocab(trn), LX.token_vocab_x(tro)
    fu_o = lambda t: [[u for u in otok.encode(f)[0] if u > 2] for f in t.split(L3.SEP) if f]
    ns = dict(np=np, torch=torch, F=F, dnn=dnn, dist=pg, X=LX, augment=LG.augment, swap_field=LK.swap_field, perturb_token=LK.perturb_token,
              flens=LT3.flens, SEP=L3.SEP, RecordEncoder3=L3.RecordEncoder3, imgs=LX.XImages("xi"), pools=po, tokv=tvo, dev=torch.device("cpu"), M=3,
              a=a, tok=otok, model=mo, ac=dict(device_type="cpu", dtype=torch.bfloat16, enabled=False), world=1, field_units=fu_o)
    C = main_closures(os.path.join(legacy_work, "x_train.py"), {"build", "batch_loss"}, ns)
    mn.train(train); mo.train(train)
    recs = trn.sample(np.random.default_rng(3), a.bs, 1, 0.15)
    torch.manual_seed(5); bn = T.build_batch(trn, recs, np.random.default_rng(9), train, X.XImages("xi"), pn, tvn, torch.device("cpu"))
    torch.manual_seed(5); bo = C["build"](tro, recs, np.random.default_rng(9), train)
    assert bn[0] == bo[0] and bn[1] == bo[1] and torch.equal(bn[2], bo[2]) and torch.equal(bn[3], bo[3])
    torch.manual_seed(21); ln, partn = T.batch_loss(mn, tok, a, T.FieldUnits(tok), *bn, train, 1)
    torch.manual_seed(21); lo, parto = C["batch_loss"](*bo, train)
    assert abs(ln.item() - lo.item()) <= TOL and all(abs(float(partn[k]) - float(parto[k])) <= TOL for k in partn)
    if train:
        ln.backward(); lo.backward()
        gn = dict(mn.named_parameters())
        for k, p in mo.named_parameters():
            assert (p.grad is None) == (gn[k].grad is None), k
            if p.grad is not None:
                assert (p.grad - gn[k].grad).abs().max().item() <= TOL, k


def test_pretrain_sampling_val_loss(legacy_work, workspace_root, pg):
    import torch.nn.functional as F
    from formrec.common.workspace import ROOT
    from formrec.retrieval import data as X
    from formrec.retrieval import pretrain as P
    LG, L3, LK, LT3, LX = legacy(legacy_work)
    path = ROOT / "runs/x/pretrain_ctc/final.pt"
    if not path.exists():
        pytest.skip(f"{path} 없음")
    dss = ("xa", "xi")
    a = P.parse_args(["--cpu", "--bs", "8", "--val_cap", "24", "--dss", ",".join(dss)])
    tr = {ds: X.items(ds, "train") for ds in dss}; imgs = {ds: X.XImages(ds) for ds in dss}
    n = np.array([len(tr[ds]) for ds in dss], float); p_ds = P.dataset_probs(n, a.alpha)
    # 원본 producer 본문(q.put 제외)
    tree = ast.parse(open(os.path.join(legacy_work, "x_pretrain.py")).read())
    prod = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == "producer")
    body = [s for s in prod.body[0].body if "q.put" not in ast.unparse(s)]
    src = "def orig_batch():\n" + "\n".join("    " + l for s in body for l in ast.unparse(s).splitlines()) + "\n    return arr, txt, dsi\n"
    ns = dict(np=np, torch=torch, rng=np.random.default_rng(42), DSS=dss, a=a, X=LX, tr=tr, imgs={ds: LX.XImages(ds) for ds in dss},
              p_ds=n ** a.alpha / (n ** a.alpha).sum())
    exec(compile(src, "x_pretrain.py", "exec"), ns)
    rng = np.random.default_rng(42)
    for _ in range(3):
        A, B = P.sample_batch(rng, dss, p_ds, tr, imgs, a.bs), ns["orig_batch"]()
        assert (A[0] == B[0]).all() and A[1] == B[1] and (A[2] == B[2]).all()
    tok = X.load_tok(); ck = torch.load(path, map_location="cpu", weights_only=False)
    mn = P.make_model(len(tok.vocab), tok.maxu, 3); mn.load_state_dict(ck["model"], strict=True)
    mo = L3.RecordEncoder3(len(tok.vocab), tok.maxu, 16, 3, 64, 1280, d=384, dim=64, layers=6, read_layers=3); mo.load_state_dict(ck["model"], strict=True)
    val = P.val_samples(dss, a.val_cap)
    vn = P.val_loss(mn, val, imgs, tok.units, torch.device("cpu"))
    ns2 = dict(np=np, torch=torch, F=F, dist=pg, model=mo, DSS=dss, val=val, rank=0, world=1, dev=torch.device("cpu"), imgs=ns["imgs"],
               augment=LG.augment, X=LX, a=a, units=lambda t: [u for u in tok.encode(t)[0] if u > 2])
    vo = main_closures(os.path.join(legacy_work, "x_pretrain.py"), {"val_loss"}, ns2)["val_loss"]()
    assert all(abs(vn[d][k] - vo[d][k]) <= TOL for d in dss for k in ("loss", "em", "n"))
