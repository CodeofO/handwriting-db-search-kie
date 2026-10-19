"""통합 어휘 meta/x/vocab.json: 세 데이터(xa · xh · xi)의 train · val · tune 정답 문자열(test 제외) + 필드 구분 기호로 만든 단위 어휘.

사전학습 CTC 머리와 FT가 같은 어휘를 쓴다. xc(한 줄 여러 칼럼)는 xa와 같은 글자라 넣지 않는다.
단위 규칙 = formrec.retrieval.tokenizer.UnitTok3 (한글 음절 = 초 · 중 · 종 자모, 그 밖 = 글자 1개).
사용: python -m formrec.data_prep.vocab      (train 환경 — 토크나이저가 torch를 불러온다)
"""
import json

from formrec.common import workspace as cfg
from formrec.common.levels import SEP, items

DSS = ("xa", "xh", "xi")
PATH = cfg.META / "x/vocab.json"


def vocab_texts():
    texts = []
    for ds in DSS:
        D = items(ds); texts += list(D[D.subset != "test"].text)
    return texts + [SEP]


def build_vocab(path=PATH):
    from formrec.retrieval.tokenizer import UnitTok3
    tok = UnitTok3(vocab_texts())
    json.dump(tok.vocab, open(path, "w"), ensure_ascii=False)
    return tok


if __name__ == "__main__":
    print("DONE vocab", len(build_vocab().vocab))
