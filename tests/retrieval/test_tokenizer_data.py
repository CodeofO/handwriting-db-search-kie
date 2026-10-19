"""토크나이저 · 레코드 표집 · 합성 음성 단위 테스트 (작업 공간 불필요)."""
import numpy as np
import pytest

from formrec.common.levels import SEP
from formrec.retrieval import data as X
from formrec.retrieval.tokenizer import ROLE, UnitTok3, field_char_lengths

torch = pytest.importorskip("torch")


def test_encode_roles_positions():
    tok = UnitTok3(["각 1", "가"])
    ids, fld, pos, role = tok.encode(f"각a{SEP}가")
    n = sum(1 for v in ids if v)
    # 각 = C · V · T (글자 순번 0), a = 글자 1개(순번 1, unk), SEP, 가 = C · V (다음 필드 순번 0)
    assert n == 7
    assert role[:7] == [ROLE["C"], ROLE["V"], ROLE["T"], ROLE["d"], ROLE["sep"], ROLE["C"], ROLE["V"]]
    assert pos[:7] == [0, 0, 0, 1, 2, 0, 0] and fld[:7] == [0, 0, 0, 0, 0, 1, 1]
    assert ids[3] == 1 and ids[4] == 2 and len(ids) == tok.maxu


def test_batch_and_units():
    tok = UnitTok3(["가나다", "12"])
    ids, fld, pos, role = tok.batch(["가나다", "1"], "cpu")
    assert ids.shape == (2, 8)              # 실제 최대 6단위 → 8의 배수
    assert tok.field_units(f"가{SEP}{SEP}12") == [tok.units("가"), tok.units("12")]
    assert all(u > 2 for u in tok.units(f"가{SEP}1"))
    t2 = UnitTok3.from_vocab(tok.vocab)
    assert t2.encode("가나") == tok.encode("가나")


def test_field_char_lengths():
    fl = field_char_lengths([f"홍길동{SEP}{SEP}서울 강남", "abc"])
    assert fl.tolist() == [[3, 0, 5], [3, 0, 0]]


class _Toy:
    """sample_multi_field가 쓰는 최소 인터페이스."""
    def __init__(self):
        self.by = {0: {"name": 0, "birth": 1, "addr2": 2, "addr3": 3}, 1: {"name": 4, "birth": 5, "addr2": 6}}
        self.pids = [0, 1]


def test_sample_multi_field_deterministic():
    D = _Toy()
    a = X.sample_multi_field(D, np.random.default_rng(0), 200, 1, 0.15)
    b = X.sample_multi_field(D, np.random.default_rng(0), 200, 1, 0.15)
    assert a == b
    for pid, f in a:
        assert 1 <= len(f) <= 3 and all(x == "" or x in D.by[pid] for x in f)
    assert any("" in f for _, f in a) and any(len(f) == 1 for _, f in a)


def test_negatives():
    pools = X.char_pools(["가나다라", "ABCabc123", "-/"])
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = X.perturb_char("가나1", rng, pools)
        if n is not None:
            assert len(n) == 3 and n != "가나1" and sum(a != b for a, b in zip(n, "가나1")) == 1
    assert X.swap_field(f"a{SEP}b", f"a{SEP}c", np.random.default_rng(0)) in (None, f"a{SEP}c")
    assert X.swap_field("a", f"a{SEP}c", rng) is None
    tokv = {("addr", 1): np.array(["강남구", "서초구"])}
    out = X.perturb_token(f"홍{SEP}서울 강남구", ["name", "addr"], np.random.default_rng(3), tokv)
    assert out in (None, f"홍{SEP}서울 서초구")
    txt = [f"홍길동{SEP}900101", f"김철수{SEP}851231"]
    negs = X.synth_negatives(txt, [["name", "birth"]] * 2, np.random.default_rng(0), X.char_pools(txt), {})
    assert len(negs) == 6 and all(n not in txt for n in negs)
    assert X.field_kinds(["name", "addr4", ""]) == ["name", "addr", ""]
