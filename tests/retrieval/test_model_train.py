"""작은 모델로 구조 · 손실 · 평가 도우미 점검 (CPU, 작업 공간 불필요)."""
import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from formrec.common.levels import SEP  # noqa: E402
from formrec.retrieval.augment import augment  # noqa: E402
from formrec.retrieval.evaluate import select_grid  # noqa: E402
from formrec.retrieval.model import RecordEncoder3  # noqa: E402
from formrec.retrieval.tokenizer import UnitTok3, field_char_lengths  # noqa: E402
from formrec.retrieval.train import FieldUnits, batch_loss  # noqa: E402

H, W = 16, 64


def tiny(tok):
    torch.manual_seed(0)
    return RecordEncoder3(len(tok.vocab), tok.maxu, 4, 3, H, W, d=32, dim=8, layers=1, pdim=16, read_layers=1)


def images(B=4):
    g = torch.Generator().manual_seed(0)
    arr = torch.full((B * 3, H, W), 255, dtype=torch.uint8)
    arr[:, 4:12, 8:40] = torch.randint(0, 120, (B * 3, 8, 32), generator=g).to(torch.uint8)
    return arr


def test_state_dict_keys_match_checkpoint_layout():
    tok = UnitTok3(["가나"])
    keys = set(tiny(tok).state_dict())
    for k in ("img.cnn.0.0.weight", "img.fold.0.weight", "img.pos", "img_id.weight", "ipool.q", "tpool.attn.in_proj_weight", "read.layers.0.linear1.weight",
              "ctc_head.weight", "mix_ctc", "vhead.0.weight", "len_head.2.weight", "mix_len", "logit_scale", "pli_scale", "mix", "iproj2.weight", "tabs.weight"):
        assert k in keys, k


def test_forward_shapes():
    tok = UnitTok3([f"가나다{SEP}12 ab"])
    m = tiny(tok).eval()
    x = torch.ones(2, 3, 1, H, W); mk = torch.tensor([[1, 1, 0], [1, 0, 0]]).bool()
    x[mk] = augment(images(1)[:3], H, W, False)
    with torch.no_grad():
        hi = m.image_tokens(x, mk); assert hi.shape == (2, 3, W // 4, 32) and hi[~mk].abs().sum() == 0
        zi = m.slots_image(hi, mk); assert zi.shape == (2, 4, 8)
        cands = [f"가나{SEP}12", f"가다{SEP}1", "가", f"나{SEP}ab"]
        ids, fld, pos, role = tok.batch(cands, "cpu"); ht = m.text_tokens(ids, fld, pos, role)
        assert RecordEncoder3.flat(m.slots_text(ht, ids)).norm(dim=1).allclose(torch.ones(4))
        cand = torch.tensor([[0, 1], [2, 3]])
        ext = m.ink_extent(x, mk)
        s, ps, pm = m.pli(hi, ext, mk, ht, fld, pos, ids, field_char_lengths(cands), cand, per_slot=True)
        assert s.shape == (2, 2) and ps.shape == (2, 2, 3) and pm.dtype == torch.bool
        g, gs, gm = m.cand_ctc(hi, mk, [tok.field_units(t) for t in cands], cand, per_slot=True)
        assert g.shape == (2, 2) and (g <= 0).all()
        f, c1 = m.verify_feats(s, s, g, ps, pm, gs, gm, s)
        assert f.shape == (2, RecordEncoder3.NVF) and m.verify(f).shape == (2,)


def test_augment_ranges():
    torch.manual_seed(0)
    a = augment(images(2), H, W, True, keep_ink=True)
    e = augment(images(2), H, W, False)
    assert a.shape == e.shape == (6, 1, H, W) and a.min() >= -1 and a.max() <= 1


def test_batch_loss_backward():
    txt = [f"가나{SEP}12", f"다라{SEP}34", f"마바{SEP}56", f"사아{SEP}78"]
    tok = UnitTok3(txt + [SEP]); m = tiny(tok)
    a = argparse.Namespace(slot_drop=0.1, div_w=0.1, pli_w=1.0, pli_hard=2, verify_w=1.0, mix=1, ctc_mix=1, ctc_w=1.0)
    x = torch.ones(4, 3, 1, H, W); mk = torch.tensor([[1, 1, 0]] * 4).bool()
    x[mk] = augment(images(4)[:8], H, W, True, keep_ink=True)
    negs = [t + s for t in txt for s in "xyz"]
    loss, p = batch_loss(m, tok, a, FieldUnits(tok), txt, negs, x, mk, True)
    assert torch.isfinite(loss) and set(p) == {"l1", "pli", "ctc", "ver", "vacc", "div", "scale"}
    loss.backward()
    assert m.vhead[0].weight.grad is not None and m.img.pos.grad is not None


def test_select_grid_tie_break():
    top = torch.tensor([[5, 7, 9]]); gt = torch.tensor([7])
    raw = dict(base=torch.tensor([[1.0, 0.5, 0.0]]), s3=torch.tensor([[0.0, 1.0, 0.0]]), top=top, gt=gt)
    grid, best = select_grid([raw], ks=(2, 3), lams=(0.0, 1.0, 2.0))
    assert grid[(2, 0.0)] == 0.0 and grid[(2, 1.0)] == 1.0 and best == (2, 1.0)   # 동률이면 작은 K · 작은 λ
