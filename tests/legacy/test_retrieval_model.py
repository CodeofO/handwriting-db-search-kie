"""제안 모델 원본 대조 1 — 추론 경로(LEGACY_WORK = 원본 코드, FORMREC_WORKSPACE = 체크포인트 · 캐시가 있는 작업 공간, 읽기만).

같은 체크포인트를 원본 RecordEncoder3와 새 모델에 strict 로드하고, 캐시 레코드 4개로
image_tokens · slots_image · text_tokens · slots_text · ink_extent · pli · cand_ctc · final_score · verify_feats · verify · ctc_loss를 비교한다.
CPU fp32. 결과는 같아야 한다(허용 1e-5 = CPU 스레드 스케줄 차이로 드물게 생기는 fp32 반올림 수준).
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
TOL = 1e-5


def legacy_modules(legacy_work):
    os.environ.setdefault("HW_DATASET", "xa")
    sys.dont_write_bytecode = True                      # 원본 폴더에 __pycache__를 만들지 않는다
    if legacy_work not in sys.path:
        sys.path.insert(0, legacy_work)
    import g4_train, kie_model3, kie_train3  # noqa: E401
    return kie_model3, kie_train3, g4_train


@pytest.mark.parametrize("run", ["xa_prop", "xi_prop"])
def test_inference_paths(legacy_work, workspace_root, run):
    from formrec.common.levels import items, level_table
    from formrec.common.workspace import ROOT
    from formrec.retrieval import data as X
    from formrec.retrieval.augment import augment
    from formrec.retrieval.evaluate import load_model
    from formrec.retrieval.tokenizer import field_char_lengths
    L3, LT3, LG = legacy_modules(legacy_work)
    path = ROOT / f"runs/x/{run}/final.pt"
    if not path.exists():
        pytest.skip(f"{path} 없음")
    torch.set_num_threads(8)
    m, tok, a = load_model(path, torch.device("cpu"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    o = L3.RecordEncoder3(len(tok.vocab), tok.maxu, a["slots"], 3, a["H"], a["W"], d=a["d"], dim=a["dim"], layers=a["layers"],
                          vfeat=a.get("vfeat", 14), read_layers=a.get("read_layers", 0))
    o.load_state_dict(ck["model"], strict=True); o.eval()
    otok = L3.UnitTok3([]); otok.vocab = ck["vocab"]; otok.id = {u: i for i, u in enumerate(otok.vocab)}
    ds = run.split("_")[0]; D = items(ds, "test"); L = level_table(ds, "test", D)
    pick = (list(L[L.level == "F3"].index[:2]) + list(L[L.level == "F2"].index[:1]) + list(L[L.level == "L1"].index[:1])) if ds == "xa" else list(L.index[:4])
    row_of = {i: r for r, i in enumerate(D.item_id)}
    rows = np.full((4, 3), -1, np.int64)
    for b, i in enumerate(pick):
        its = L.loc[i, "items"]; rows[b, :len(its)] = [row_of[x] for x in its]
    texts = list(L.loc[pick, "text"])
    flat = rows[rows >= 0]
    arr = torch.from_numpy(X.XImages(ds).get(list(D.src.values[flat]), list(D.row.values[flat])))
    mk = torch.from_numpy(rows >= 0)
    x = torch.ones(4, 3, 1, a["H"], a["W"]); x[mk] = augment(arr, a["H"], a["W"], False)
    xo = torch.ones(4, 3, 1, a["H"], a["W"]); xo[mk] = LG.augment(arr, a["H"], a["W"], False)
    assert torch.equal(x, xo)
    cands = []
    for k, t in enumerate(texts):
        cands += [t, t[:-1] + ("가" if ds == "xa" else "a"), texts[(k + 1) % 4], t + "x"]
    cand = torch.arange(16).view(4, 4)
    d = lambda p, q: (p - q).abs().max().item()
    with torch.no_grad():
        hn, ho = m.image_tokens(x, mk), o.image_tokens(x, mk); assert d(hn, ho) <= TOL
        assert d(m.slots_image(hn, mk), o.slots_image(ho, mk)) <= TOL
        bn, bo = tok.batch(cands, "cpu"), otok.batch(cands, "cpu"); assert all(torch.equal(p, q) for p, q in zip(bn, bo))
        htn, hto = m.text_tokens(*bn), o.text_tokens(*bo); assert d(htn, hto) <= TOL
        assert d(m.slots_text(htn, bn[0]), o.slots_text(hto, bo[0])) <= TOL
        fl = field_char_lengths(cands); assert torch.equal(fl, LT3.flens(cands))
        en, eo = m.ink_extent(x, mk), o.ink_extent(x, mk); assert torch.equal(en, eo)
        pn = m.pli(hn, en, mk, htn, bn[1], bn[2], bn[0], fl, cand, per_slot=True)
        po = o.pli(ho, eo, mk, hto, bo[1], bo[2], bo[0], fl, cand, per_slot=True)
        assert d(pn[0], po[0]) <= TOL and d(pn[1], po[1]) <= TOL and torch.equal(pn[2], po[2])
        un = [tok.field_units(t) for t in cands]
        assert un == [[[u for u in otok.encode(f)[0] if u > 2] for f in t.split("␞") if f] for t in cands]
        cn, co = m.cand_ctc(hn, mk, un, cand, per_slot=True), o.cand_ctc(ho, mk, un, cand, per_slot=True)
        assert d(cn[0], co[0]) <= TOL and d(cn[1], co[1]) <= TOL and torch.equal(cn[2], co[2])
        s1c = (m.flat(m.slots_image(hn, mk)) @ m.flat(m.slots_text(htn, bn[0])).T).gather(1, cand)
        sn = m.final_score(s1c, pn[0]) + m.mix_ctc * cn[0]; so = o.final_score(s1c, po[0]) + o.mix_ctc * co[0]
        assert d(sn, so) <= TOL
        fn, _ = m.verify_feats(sn, pn[0], cn[0], pn[1], pn[2], cn[1], cn[2], s1c)
        fo, _ = o.verify_feats(so, po[0], co[0], po[1], po[2], co[1], co[2], s1c, pl=po[3])
        assert d(fn, fo) <= TOL and d(m.verify(fn), o.verify(fo)) <= 10 * TOL
        tg = [u for t in texts for u in tok.field_units(t)]
        assert abs(m.ctc_loss(hn, mk, tg).item() - o.ctc_loss(ho, mk, tg).item()) <= TOL
