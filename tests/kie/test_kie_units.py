"""formrec.kie 단위 테스트 — 작은 합성 입력만(작업 공간 불필요)."""
import json

import numpy as np
import pandas as pd

from formrec.kie import data, evaluate, oov, order, verify
from formrec.kie.fields import fmt, norm, parse, parse_loose, prompt, query_ok, target_dict


def test_prompt_parse():
    keys = ["이름", "주소"]
    assert prompt(keys) == "KIE: 이름, 주소"
    raw = "이름: 홍길동\n주소：서울특별시 종로구\n이름: 두번째\n기타: x"
    assert parse(raw, keys) == {"이름": "홍길동", "주소": "서울특별시 종로구"}
    assert parse("- 이름: 김", keys) == {}
    assert parse_loose("- 이름: 김\n* 주소: 부산", keys) == {"이름": "김", "주소": "부산"}


def test_norm_fmt_target():
    assert norm("생년월일", "78-07 05") == "780705"
    assert norm("주소", "  서울  종로구 ") == "서울 종로구"
    assert norm("이름", None) == ""
    assert fmt(["주소", "이름"], {"이름": "김  철수", "주소": "a  b"}) == "주소: a b\n이름: 김 철수"
    assert target_dict("이름: 김\n주소: a: b") == {"이름": "김", "주소": "a: b"}
    assert query_ok(["이름"], "이름: 김", {"이름": " 김 "}) and not query_ok(["이름", "주소"], "이름: 김\n주소: a", {"이름": "김"})


def _items(n=5):
    rows = []
    for i in range(n):
        v = {"이름": f"이{i}", "생년월일": f"80010{i}", "주소": f"서울특별시 구{i} 동{i}", "시도": "서울특별시", "시군구": f"구{i}"}
        rows.append(dict(item_id=f"p{i}", row=i, values=json.dumps(v, ensure_ascii=False)))
    return pd.DataFrame(rows)


def test_queries_deterministic():
    D = _items()
    a, b = data.queries(D, 3), data.queries(D, 3)
    assert a.equals(b) and len(a) == 15
    for r in a.itertuples():
        keys = json.loads(r.keys)
        assert 1 <= len(keys) <= 4 and len(set(keys)) == len(keys)
        assert [l.split(": ")[0] for l in r.target.split("\n")] == keys


def test_values_address_tokens():
    r = {"name": "김", "birth": "800101", "a1": "서울특별시", "a2": "종로구", "a3": "", "a4": "", "a5": "", "a6": ""}
    assert data.values(r) == {"이름": "김", "생년월일": "800101", "주소": "서울특별시 종로구", "시도": "서울특별시", "시군구": "종로구"}
    r["a2"] = ""
    assert "시군구" not in data.values(r)


def test_page_checks(tmp_path):
    def b(t, x0, x1, y0=100, y1=140):
        return {"data": t, "x": [x0, x1], "y": [y0, y1]}
    bl = [b("123456", 0, 50), b("1234567", 60, 120), b("서울특별시", 200, 300), b("종로구", 310, 380), b("101호", 420, 480), b("02", 200, 240, 300, 340)]
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"bbox": bl}, ensure_ascii=False))
    assert data.page_checks((str(p), [2, 3], ["서울특별시"])) == (1, 1, False)       # 같은 줄 오른쪽에 이어지는 토큰 → 주소 칸 일부
    bl[4] = b("101호", 700, 760)
    p.write_text(json.dumps({"bbox": bl}, ensure_ascii=False))
    assert data.page_checks((str(p), [2, 3], ["서울특별시"]))[2]


def _db():
    real = pd.DataFrame({"이름": ["김철수", "이영희", "박민수"], "생년월일": ["800101", "810202", "820303"],
                         "주소": ["서울 종로구 a", "부산 해운대구 b", "대구 수성구 c"], "src": ["p0", "p1", "p2"]})
    a = real["주소"].str.split()
    return real.assign(시도=a.str[0], 시군구=a.str[1])


def test_db_fix_record_and_conf():
    D = _db()
    preds = [{"이름": "김철스", "주소": "서울 종로구 a"}, {"이름": "이영희"}, {"생년월일": "820308"}]
    out, conf = evaluate.db_fix_record(preds, db=D)
    assert out[0] == {"이름": "김철수", "주소": "서울 종로구 a"} and out[1] == {"이름": "이영희"} and out[2] == {"생년월일": "820303"}
    assert conf[0].tolist()[0] == 1 and conf[1].tolist()[0] == 0 and conf.shape == (3, 2)


def test_score():
    Q = pd.DataFrame(dict(qid=["a", "b"], keys=[json.dumps(["이름"]), json.dumps(["이름", "생년월일"], ensure_ascii=False)],
                          target=["이름: 김", "이름: 이\n생년월일: 800101"]))
    s = evaluate.score(Q, [{"이름": "김"}, {"이름": "이", "생년월일": "80-01-01"}])
    assert s["query_em"] == 1.0 and s["em_2keys"] == 1.0 and s["n"] == 2
    s = evaluate.score(Q, [{"이름": "박"}, {"이름": "이"}])
    assert s["query_em"] == 0.0 and s["key_이름"] == 0.5


def test_best_tau():
    d1 = np.array([0, 1, 5, 9])
    ok_db = np.array([1, 1, 0, 0], bool)
    ok_raw = np.array([0, 0, 1, 1], bool)
    assert oov.best_tau(d1, ok_db, ok_raw) == 1


def test_permuted_queries():
    Q = pd.DataFrame(dict(qid=["a#0", "b#0"], item_id=["a", "b"], row=[0, 1], keys=[json.dumps(["이름", "주소", "시도"]), json.dumps(["이름"])],
                          target=["이름: 김\n주소: x y\n시도: x", "이름: 이"]))
    P1, P2 = order.permuted_queries(Q), order.permuted_queries(Q)
    assert P1.equals(P2) and len(P1) == 1 and P1.qid[0] == "a#0~p"
    assert json.loads(P1["keys"][0]) != ["이름", "주소", "시도"]
    assert target_dict(P1.target[0]) == target_dict(Q.target[0])


def test_order_metrics_binding():
    A = pd.DataFrame(dict(qid=["q"], pred=[json.dumps({"이름": "김", "시도": "서울"}, ensure_ascii=False)])).set_index("qid")
    B = pd.DataFrame(dict(qid=["q~p"], keys=[json.dumps(["시도", "이름"])], target=["시도: 서울\n이름: 김"],
                          pred=[json.dumps({"이름": "서울", "시도": "서울"}, ensure_ascii=False)]))
    m = order.order_metrics(A, B)
    assert m["n"] == 1 and m["order_consistency"] == 0.0 and m["binding_error"] == 0.25 and m["em_original"] == 1.0 and m["em_permuted"] == 0.0


def test_candidates_and_decide():
    D = pd.concat([_db(), _db().assign(이름=["김철순", "이영회", "박민서"], src=["p0", "p1", "p2"])], ignore_index=True)
    Q = pd.DataFrame(dict(qid=["q0", "q1"], keys=[json.dumps(["이름"])] * 2, item_id=["p0", "p9"], target=["이름: 김철수", "이름: 최없음"]))
    P = [{"이름": "김철스"}, {"이름": "최없음"}]
    C = verify.candidates(Q, P, D, k=3, pool=6)
    assert list(C.qid) == ["q0", "q1"] and C.cand[0][0] in ("이름: 김철수", "이름: 김철순") and len(C.cand[0]) == 3
    assert C.free[1] == "이름: 최없음"
    ll = {("q0", 0): -5.0, ("q1", 0): -1.0}
    for q, c in zip(C.qid, C.cand):
        for j, t in enumerate(c):
            ll[(q, j + 1)] = -0.5 if (q == "q0" and t == "이름: 김철수") else -20.0
    Qi = Q.set_index("qid")
    ok, conf = verify.decide_one(C, ll, Qi, set(), 2.0, "lrv")
    assert ok.tolist() == [True, True]                            # q0 = 우도 1등(정답), q1 = 우도비가 커서 자유 생성 유지
    ok, _ = verify.decide_one(C, ll, Qi, {"p0"}, 2.0, "lrv")       # p0이 DB 밖이면 그 출처 후보가 빠진다
    assert set(conf) == {"1·2등 차", "−Λ(또는 −d1)"} and len(ok) == 2
