"""formrec.kie_general 단위 테스트 — 합성 입력만. 칸 검출(cells_of)은 cv2가 있을 때만."""
import json

import numpy as np
import pandas as pd
import pytest

from formrec.kie_general import build, cells, evaluate


def test_norm_key():
    assert cells.norm_key("① 성 명 (Name)") == "성명"
    assert cells.norm_key("주소1)") == "주소"
    assert cells.norm_key("성명 (법인명)") == "성명(법인명)"
    assert cells.norm_key("(현거주지)") == "현거주지"
    assert cells.norm_key("8 1일통상임금") == "1일통상임금"


def test_geometry_helpers():
    assert cells.ov1(0, 10, 5, 20) == 5 and cells.ov1(0, 1, 2, 3) == 0
    C = [(0, 0, 100, 100), (10, 10, 50, 50)]
    assert cells.cell_of(C, 20, 20) == 1 and cells.cell_of(C, 80, 80) == 0 and cells.cell_of(C, 200, 200) is None


def test_core():
    assert build.core("신청인 성명(대표자)") == "성명"


def _items():
    kv = {"성명": "홍길동", "주소": "서울 종로구", "전화번호": "010", "생년월일": "800101"}
    return pd.DataFrame([dict(page_id=f"pg{i}", row=i, kv=json.dumps(kv, ensure_ascii=False), present=json.dumps(sorted(kv) + ["비고"], ensure_ascii=False),
                              empty=json.dumps(["비고"], ensure_ascii=False)) for i in range(40)])


def test_queries_negatives():
    Q = build.queries(_items(), 1, ["성명", "주소", "계좌번호", "신청인 성명"])
    assert len(Q) == 40 and build.queries(_items(), 1, ["성명", "계좌번호"]).equals(build.queries(_items(), 1, ["성명", "계좌번호"]))
    for r in Q.itertuples():
        tg = dict(l.split(": ", 1) for l in r.target.split("\n"))
        negs = [k for k, v in tg.items() if v == build.NONE]
        assert len(negs) == (1 if r.neg else 0)
        if r.neg == "A":
            assert negs == ["계좌번호"]                       # 페이지에 없는 키만 · 구역 접두 키 제외
        if r.neg == "B":
            assert negs == ["비고"]
    assert set(Q.neg) <= {"", "A", "B"} and (Q.neg != "").any()


def test_drop_rare_and_group_templates():
    R = [("서식1", {"성명": "a", "희귀": "b"}, ["희귀"], ["성명", "주소"])] * 3 + [("서식1", {"성명": "a"}, [], ["성명", "주소"])] * 17 + [("?", {"성명": "c"}, [], ["성명", "주소"]), ("?", {"성명": "d"}, [], ["다른", "항목"])]
    R2 = build.group_templates(build.drop_rare(R, min_pages=20))
    assert R2[0][1] == {"성명": "a"} and R2[0][2] == []            # 희귀 키(20페이지 미만) 제거
    assert R2[20][0] == "서식1" and R2[21][0] == "기타:?"


def test_evaluate_subset_metrics():
    V = pd.DataFrame(dict(qid=["a", "b"], keys=[json.dumps(["성명", "계좌번호"], ensure_ascii=False), json.dumps(["주소"], ensure_ascii=False)],
                          target=["성명: 김\n계좌번호: [없음]", "주소: 서울"],
                          pred=[json.dumps({"성명": "김", "계좌번호": "[없음]"}, ensure_ascii=False), json.dumps({"주소": "[없음]"}, ensure_ascii=False)]))
    Q = pd.DataFrame(dict(qid=["a", "b"], neg=["A", ""])).set_index("qid")
    m = evaluate.subset_metrics("test", V, Q)
    assert m["query_em"] == 0.5 and m["neg_A_acc"] == 1.0 and m["n_negB"] == 0 and m["key_acc_pos"] == 0.5 and m["miss_rate"] == 0.5


def test_cells_of_grid():
    cv2 = pytest.importorskip("cv2")
    g = np.full((600, 900), 255, np.uint8)
    for y in (100, 200, 300):
        cv2.line(g, (50, y), (850, y), 0, 3)
    for x in (50, 300, 850):
        cv2.line(g, (x, 100), (x, 300), 0, 3)
    C = cells.cells_of(g)
    assert len(C) == 4                                          # 2행 × 2열
    assert cells.cell_of(C, 150, 150) is not None and cells.cell_of(C, 150, 150) != cells.cell_of(C, 500, 250)
