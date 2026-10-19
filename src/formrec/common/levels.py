"""레코드 문자열 · 실험 단계(사다리) 정의 — torch 없이 쓰는 공용 모듈(제안 모델 · 비교군 채점 · 요약이 함께 쓴다).

레코드 문자열 = 칼럼 값을 SEP(␞)로 이은 것. 단계(level): 길이 · 칼럼 수 사다리(xa L1~L6 · F2~F3, xh H2~H5+, xi I1~I5, xc C2~C3).
서브 단계 표기: 'F3@2M' = DB 규모 변형, 'F3#d1' = 칼럼 1을 질의 · DB 양쪽에서 비움(칼럼 누락).
"""
import numpy as np
import pandas as pd

from formrec.common.workspace import ROOT

SEP = "␞"
LV_KIE = ["L1", "L2", "L3", "L4", "L5", "L6", "F2", "F3"]
LV_HANA = ["H2", "H3", "H4", "H5+"]


def parse_level(level):
    """서브 실험 단계 이름 → (기본 단계, DB 접미사, 비운 필드 번호).
    'F3@2M' = F3 질의 + meta/kie_db_F3_2M (DB 규모), 'F3#d1' = F3에서 필드 1(생년월일)을 질의·DB 양쪽에서 비움 (필드 누락)."""
    drop = None
    if "#d" in level:
        level, d = level.split("#d"); drop = int(d)
    suf = ""
    if "@" in level:
        level, suf = level.split("@"); suf = "_" + suf
    return ("L1" if level == "F1" else level), suf, drop


def blank(t, j):
    f = t.split(SEP); f[j] = ""
    return SEP.join(f)


def db_texts(data, level, limit=0, keep=()):
    """단계 DB 문자열(정렬된 고유값). kie = meta/kie_db_{단계}, hana = 전체 이름을 '성␞이름␞중간이름'으로.
    필드 누락 단계는 DB 쪽도 같은 필드를 비우고 중복을 합친다(남은 필드가 같은 레코드는 구분 불가 → 같은 항목)."""
    base, suf, drop = parse_level(level)
    if data == "kie":
        t = pd.read_parquet(ROOT / f"meta/kie_db_{base}{suf}.parquet").text.values
    else:
        d = pd.read_parquet(ROOT / "meta/hana_db.parquet")
        t = (d["last"] + SEP + d["first"] + SEP + d["middle"]).values
    if drop is not None:
        t = np.array(sorted({blank(x, drop) for x in t}))
    if limit:
        s = set(keep); t = np.array(sorted(s | set(t[:limit])))
    return t


LEVELS = {"xa": ["L1", "L2", "L3", "L4", "L5", "L6", "F2", "F3"], "xh": ["H2", "H3", "H4", "H5+"], "xi": ["I1", "I2", "I3", "I4", "I5"], "xc": ["C2", "C3"]}


def items(ds, sub=None):
    D = pd.read_parquet(ROOT / f"meta/x/{ds}_items.parquet")
    return D if sub is None else D[D.subset == sub].reset_index(drop=True)


def xi_level(n):
    return "I1" if n <= 5 else "I2" if n <= 7 else "I3" if n <= 9 else "I4" if n <= 11 else "I5"


def level_table(ds, sub, D=None):
    """단계 질의 표 (level, pid, items, text) — 그 세트의 레코드. D = 이미지가 있는 item 집합(표본) 제한."""
    D = items(ds, sub) if D is None else D
    if ds == "xa":
        A = items("xa"); sub_of = dict(zip(A.rec_id, A.subset))
        L = pd.read_parquet(ROOT / "meta/kie_levels.parquet")
        L = L[L.pid.map(sub_of) == sub].drop_duplicates(["level", "pid"])
        have = set(D.item_id)
        return L[[all(i in have for i in its) for its in L["items"]]].reset_index(drop=True)
    lv = D.level.values if ds in ("xh", "xc") else np.array([xi_level(n) for n in D.n_words])
    return pd.DataFrame(dict(level=lv, pid=D.rec_id.values, items=[[i] for i in D.item_id], text=D.text.values))
