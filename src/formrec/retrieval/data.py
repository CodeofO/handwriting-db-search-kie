"""제안 모델 데이터: 분할표 · 통합 어휘 · 이미지 저장소 · 레코드 표집 · 합성 음성.

레코드
  xa : 사람 1명 = 필드 이미지 여러 장(이름 · 생년월일 · 주소 k토큰). 표집 = 단일 필드 50% · 복수 필드(2~3개) 50%, 순서 섞기, 필드 누락.
  xh : 이름 줄 이미지 1장 = 필드 1개(칼럼을 쓰인 순서대로 공백으로 이은 문자열). 한 이미지 여러 칼럼을 구조 변경 없이 다루는 일반형.
  xi : IAM 줄 이미지 1장 = 필드 1개.        xc : xa 칼럼을 한 줄로 이어 붙인 이미지 1장 = 필드 1개.
어휘: xa · xh · xi의 train · val · tune 정답 문자열(test 제외)로 만든 UnitTok3 하나(meta/x/vocab.json, 만들기 = formrec.data_prep.vocab)
  — 사전학습 CTC 머리와 FT가 같은 어휘.
이미지: 64×1280 uint8(흰색 255). xh 캐시는 폭 448(나머지는 흰색으로 채움).
"""
import numpy as np

from formrec.common.levels import SEP, items  # noqa: F401  (items: 다른 모듈이 data.items로 부른다)
from formrec.common.workspace import ROOT
from formrec.retrieval.tokenizer import UnitTok3

DSS = ("xa", "xh", "xi")            # 어휘 · 사전학습에 쓰는 데이터셋 (xc는 FT · 평가만)
H, W = 64, 1280
CACHE = {"xa": {f"src_{s}": f"cache/kie/{s}_64x1280.npy" for s in ("train", "tune", "test")},
         "xh": {"all": "cache/xh/all_64x448.npy"}, "xi": {"all": "cache/xi/all_64x1280.npy"},
         "xc": {"all": "cache/xc/all_64x1280.npy"}}


def vocab_path():
    return ROOT / "meta/x/vocab.json"


def load_tok():
    return UnitTok3.load(vocab_path())


class XImages:
    """(src, row) → uint8 (n, 64, 1280). mmap 배열을 CPU에서 모은다(큰 캐시를 GPU에 다 올리지 않는다)."""
    def __init__(self, ds):
        self.ds = ds
        self.arr = {k: np.load(ROOT / p, mmap_mode="r") for k, p in CACHE[ds].items() if (ROOT / p).exists()}

    def get(self, srcs, rows):
        out = np.full((len(rows), H, W), 255, np.uint8)
        for k in set(srcs):
            sel = np.array([i for i, s in enumerate(srcs) if s == k])
            a = self.arr[k]; r = np.asarray(rows)[sel]
            o = np.argsort(r)                                   # 정렬된 읽기(mmap 순차성)
            out[sel[o], :, :a.shape[2]] = a[r[o]]
        return out


class XData:
    """학습 · 평가용 레코드 집합: pids · by(pid → {필드 종류: 행}) · text · record · sample."""
    def __init__(self, ds, sub):
        self.ds, self.sub = ds, sub
        D = items(ds, sub)
        self.q = D
        self.src, self.rows, self.text = D.src.values, D.row.values, D.text.values
        self.by = {}
        for i, (p, k) in enumerate(zip(D.rec_id.values, self.kinds())):
            self.by.setdefault(p, {})[k] = i
        self.pids = sorted(self.by)
        self.single = ds != "xa"

    def kinds(self):
        """행별 필드 종류: xa = name · birth · addr{k}, 그 밖 = line."""
        return self.q.kind.values if self.ds == "xa" else np.array(["line"] * len(self.q))

    def record(self, pid, fields):
        """fields: 필드 종류 목록(빈 문자열 = 누락 필드) → (레코드 문자열, 이미지 행 목록)."""
        d = self.by[pid]
        return SEP.join(self.text[d[f]] if f else "" for f in fields), [d[f] for f in fields if f]

    def sample(self, rng, B, shuffle=1, field_drop=0.15):
        if self.single:
            return [(self.pids[i], ["line"]) for i in rng.integers(len(self.pids), size=B)]
        return sample_multi_field(self, rng, B, shuffle, field_drop)


def sample_multi_field(D, rng, B, shuffle, field_drop):
    """xa 레코드 표집: 50% 단일 필드(아무 종류), 50% 이름 · 생년월일 · 주소 k 중 2~3개(순서 섞기, field_drop 확률로 한 필드 비움)."""
    recs = []
    for _ in range(B):
        pid = D.pids[rng.integers(len(D.pids))]; d = D.by[pid]
        addrs = [k for k in d if k.startswith("addr")]
        if rng.random() < 0.5:
            f = [list(d)[rng.integers(len(d))]]
        else:
            a = addrs[rng.integers(len(addrs))]
            pool = ["name", "birth", a]
            n = 2 + int(rng.random() < 0.5)
            f = [pool[i] for i in sorted(rng.choice(3, n, replace=False))]
            if shuffle:
                f = [f[i] for i in rng.permutation(len(f))]
            if field_drop and rng.random() < field_drop:
                f[rng.integers(len(f))] = ""
        recs.append((pid, f))
    return recs


def field_kinds(fields):
    """필드 종류 목록 → 토큰 교체 음성의 종류 키(주소 k토큰은 모두 addr)."""
    return ["addr" if x.startswith("addr") else x for x in fields]


# ---------------------------------------------------------------- 합성 음성
def char_pools(texts):
    """같은 종류 글자 치환용 풀: 한글 음절 · 대문자 · 소문자 · 숫자 · 기타. 데이터셋 글자 집합에서."""
    cs = sorted({c for t in texts for c in t if c not in (" ", SEP)})
    P = {"hangul": [c for c in cs if "가" <= c <= "힣"], "upper": [c for c in cs if c.isupper()],
         "lower": [c for c in cs if c.islower()], "digit": [c for c in cs if c.isdigit()]}
    P["other"] = [c for c in cs if not any(c in v for v in P.values())]
    return P


def char_class(c):
    return "hangul" if "가" <= c <= "힣" else "upper" if c.isupper() else "lower" if c.islower() else "digit" if c.isdigit() else "other"


def perturb_char(t, rng, pools):
    """글자 1개를 같은 종류의 다른 글자로 치환(필드 종류 · 언어 무관). 바꿀 수 없으면 None."""
    pos = [i for i, ch in enumerate(t) if ch not in (" ", SEP)]
    if not pos:
        return None
    i = pos[rng.integers(len(pos))]; p = pools.get(char_class(t[i])) or []
    if len(p) < 2:
        return None
    new = p[rng.integers(len(p))]
    return None if new == t[i] else t[:i] + new + t[i + 1:]


def token_vocab(data):
    """{(필드 종류, 토큰 자리): 토큰 배열} — train에서 같은 종류 · 같은 자리에 나온 토큰(2종 이상인 자리만)."""
    tv = {}
    for k, t in zip(data.kinds(), data.text):
        k = "addr" if k.startswith("addr") else k
        for i, w in enumerate(t.split(" ")):
            tv.setdefault((k, i), set()).add(w)
    return {k: np.array(sorted(v)) for k, v in tv.items() if len(v) > 1}


def perturb_token(t, keys, rng, tokv):
    """토큰(띄어쓰기 단위) 1개를 train에서 같은 자리에 나온 다른 토큰으로 교체 — DB 이웃(토큰 1개만 다름)과 같은 종류의 음성.
    keys: 필드별 종류, tokv: token_vocab. 바꿀 곳이 없으면 None."""
    f = t.split(SEP)
    cand = [(j, i) for j, (x, k) in enumerate(zip(f, keys)) if x for i in range(len(x.split(" "))) if (k, i) in tokv]
    if not cand:
        return None
    j, i = cand[rng.integers(len(cand))]
    w = f[j].split(" "); v = tokv[(keys[j], i)]; u = v[rng.integers(len(v))]
    if u == w[i]:
        return None
    w[i] = u; f[j] = " ".join(w)
    return SEP.join(f)


def swap_field(t, other, rng):
    """필드 1개를 다른 레코드의 같은 자리 값으로 교체. 필드 수가 다르거나 같은 값이면 None."""
    a, b = t.split(SEP), other.split(SEP)
    if len(a) != len(b):
        return None
    j = rng.integers(len(a))
    if a[j] == b[j]:
        return None
    a[j] = b[j]
    return SEP.join(a)


def synth_negatives(txt, keys, rng, pools, tokv):
    """레코드마다 합성 음성 3개(글자 1개 · 필드 교체 · 토큰 1개, 실패하면 글자 치환 → 접미 글자로 대체) → 길이 3·B 목록.
    난수 호출 순서가 학습 재현성의 일부다(바꾸지 않는다)."""
    negs = []
    for i, t in enumerate(txt):
        n1 = perturb_char(t, rng, pools)
        n2 = swap_field(t, txt[rng.integers(len(txt))], rng)
        n3 = perturb_token(t, keys[i], rng, tokv)
        negs += [n1 or t + "x", n2 or (perturb_char(t, rng, pools) or t + "y"), n3 or perturb_char(t, rng, pools) or t + "z"]
    return negs

