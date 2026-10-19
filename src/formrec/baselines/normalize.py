"""생성 결과 정리(PP-OCRv5 · VLM 공통): train 라벨에 나온 글자만 남기고 공백을 하나로.

- 한국어 서식(xa · xc): 한글 음절 전체도 허용 — tune · test에 train에 없는 희귀 음절이 있어 정답이 지워지는 것을 막는다.
  (한글만 남기는 규칙을 쓰면 생년월일 · 번지 숫자와 띄어쓰기가 지워져 전부 오답이 된다)
- HANA(xh): 라벨이 소문자 → 생성 결과도 소문자로.
- VLM: 마크업(<img …>, 태그)과 줄바꿈을 공백으로 바꾼 뒤 같은 규칙.
"""
import re

HANGUL = {chr(c) for c in range(0xAC00, 0xD7A4)}


def allowed_chars(ds, train_texts):
    allowed = set("".join(train_texts))
    if ds in ("kie", "xa", "xc"):
        allowed |= HANGUL
    return allowed


def strip_markup(r):
    return re.sub(r"<[^>]+>", " ", re.sub(r"<img[^>]*>", " ", r)).replace("\n", " ")


def make_normalizer(ds, allowed, markup=False):
    low = ds in ("hnf", "xh")
    pre = strip_markup if markup else (lambda r: r)

    def norm(r):
        r = pre(r)
        return " ".join("".join(c for c in (r.lower() if low else r) if c in allowed).split())
    return norm
