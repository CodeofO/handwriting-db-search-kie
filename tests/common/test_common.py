import numpy as np

from formrec.common import jamo, levels, selective


def test_jamo_roundtrip():
    assert jamo.decompose("라아심") == [("ㄹ", "ㅏ", ""), ("ㅇ", "ㅏ", ""), ("ㅅ", "ㅣ", "ㅁ")]
    assert jamo.jamo_string("각") == "ㄱㅏㄱ" and jamo.jamo_string("가") == "ㄱㅏ_"


def test_parse_level():
    assert levels.parse_level("F3@2M") == ("F3", "_2M", None)
    assert levels.parse_level("F3#d1") == ("F3", "", 1)
    assert levels.parse_level("F1") == ("L1", "", None)


def test_coverage_at_precision():
    correct = np.array([1, 1, 1, 0, 1, 0], bool)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    cov, thr = selective.coverage_at_precision(correct, conf, (0.99,))[0.99]
    assert cov == 0.5 and thr == 0.7
    r = selective.apply_threshold(correct, conf, thr)
    assert r["n_acc"] == 3 and r["prec"] == 1.0
