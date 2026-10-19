"""레코드 문자열 → 단위 토큰 (단위 id, 필드 번호, 필드 안 글자 순번, 자모 역할).

한글 음절 = 초 · 중 · (종) 자모 단위 3개(종성 없으면 2개), 그 밖(숫자 · 라틴 · 기호 · 공백) = 글자 1개.
같은 음절의 자모는 같은 글자 순번을 공유한다 → 이미지의 글자 위치와 바로 대응(PLI의 위치 사전).
id: 0 = pad(= CTC blank), 1 = unk, 2 = 필드 구분(SEP).
"""
import json

from formrec.common.jamo import decompose
from formrec.common.levels import SEP

ROLE = {"C": 1, "V": 2, "T": 3, "d": 4, "sep": 5}          # 0 = pad
MAX_UNITS = 160


class UnitTok3:
    """문자열 → (단위 id, 필드 번호, 필드 안 글자 순번, 역할) 네 줄."""
    def __init__(self, texts, maxu=MAX_UNITS):
        units = set()
        for t in texts:
            for ch in t:
                for u, _ in self.char_units(ch):
                    units.add(u)
        self.vocab = ["<pad>", "<unk>", "<sep>"] + sorted(units - {"<sep>"})
        self.id = {u: i for i, u in enumerate(self.vocab)}
        self.maxu = maxu

    @classmethod
    def from_vocab(cls, vocab, maxu=MAX_UNITS):
        """저장된 어휘 목록(체크포인트 · vocab.json)으로 만든다."""
        tok = cls.__new__(cls)
        tok.vocab = list(vocab); tok.id = {u: i for i, u in enumerate(tok.vocab)}; tok.maxu = maxu
        return tok

    @classmethod
    def load(cls, path):
        return cls.from_vocab(json.load(open(path)))

    @staticmethod
    def char_units(ch):
        if ch == SEP:
            return [("<sep>", "sep")]
        if "가" <= ch <= "힣":
            c, v, j = decompose(ch)[0]
            return [(f"C{c}", "C"), (f"V{v}", "V")] + ([(f"T{j}", "T")] if j else [])
        return [(ch, "d")]

    def encode(self, t):
        """→ ids, fld, pos, role (각 길이 maxu, 0으로 채움)."""
        ids, fld, pos, role = [], [], [], []
        f, p = 0, 0
        for ch in t:
            if ch == SEP:
                ids.append(2); fld.append(f); pos.append(p); role.append(ROLE["sep"]); f += 1; p = 0; continue
            for u, r in self.char_units(ch):
                ids.append(self.id.get(u, 1)); fld.append(f); pos.append(p); role.append(ROLE[r])
            p += 1
        L = self.maxu
        ids, fld, pos, role = ids[:L], fld[:L], pos[:L], role[:L]
        pad = L - len(ids)
        return ids + [0] * pad, fld + [0] * pad, pos + [0] * pad, role + [0] * pad

    def units(self, t):
        """CTC 정답용 단위 id 목록(pad · unk · SEP 제외)."""
        return [u for u in self.encode(t)[0] if u > 2]

    def field_units(self, t):
        """레코드 문자열 → 비지 않은 필드별 단위 id 목록."""
        return [self.units(f) for f in t.split(SEP) if f]

    def batch(self, texts, device):
        """→ 텐서 4개 (B, L). L = 배치 안 실제 최대 길이를 8의 배수로 올린 값(속도 · 메모리)."""
        import torch                     # 어휘 생성(base 환경, torch 없음)에서도 이 모듈을 불러올 수 있게
        e = [self.encode(t) for t in texts]
        L = max(1, max(sum(1 for v in x[0] if v) for x in e))
        L = -(-L // 8) * 8
        return tuple(torch.tensor([x[k][:L] for x in e], device=device) for k in range(4))


def field_char_lengths(texts, n_fields=3):
    """레코드별 필드 글자 수 (B, n_fields) long — PLI 위치 사전의 n_f (빈 필드 0)."""
    import torch
    out = torch.zeros(len(texts), n_fields, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, f in enumerate(t.split(SEP)[:n_fields]):
            out[i, j] = len(f)
    return out
