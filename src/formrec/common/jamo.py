"""한글 음절 <-> 자모 분해와 토큰화 (자모 단위 / 음절 단위)."""
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def decompose(s: str):
    """'라아심' -> [('ㄹ','ㅏ',''), ('ㅇ','ㅏ',''), ('ㅅ','ㅣ','ㅁ')]. 비한글은 (c,'','')."""
    out = []
    for c in s:
        k = ord(c) - 0xAC00
        if 0 <= k < 11172:
            out.append((CHO[k // 588], JUNG[(k % 588) // 28], JONG[k % 28]))
        else:
            out.append((c, "", ""))
    return out


# 자모 토큰: 초성/중성/종성을 위치별로 구분(같은 ㄱ이라도 초성과 종성은 다른 토큰)
JAMO_VOCAB = ["<pad>", "<unk>"] + [f"C{c}" for c in CHO] + [f"V{v}" for v in JUNG] + [f"T{t}" for t in JONG[1:]]
JAMO_ID = {t: i for i, t in enumerate(JAMO_VOCAB)}


def jamo_tokens(s: str):
    ids = []
    for c, v, t in decompose(s):
        if not v:
            ids.append(1)
            continue
        ids += [JAMO_ID[f"C{c}"], JAMO_ID[f"V{v}"]] + ([JAMO_ID[f"T{t}"]] if t else [])
    return ids


def jamo_string(s: str) -> str:
    """편집거리용: 종성은 소문자 표식 없이 이어 붙이되 위치 모호성을 피하려 초/중/종 모두 1글자씩(종성 없음은 '_')."""
    return "".join(c + v + (t or "_") for c, v, t in decompose(s))


def syllable_vocab(strings):
    chars = sorted({c for s in strings for c in s})
    return ["<pad>", "<unk>"] + chars


if __name__ == "__main__":
    assert decompose("라아심") == [("ㄹ", "ㅏ", ""), ("ㅇ", "ㅏ", ""), ("ㅅ", "ㅣ", "ㅁ")]
    assert jamo_string("각") == "ㄱㅏㄱ" and jamo_string("가") == "ㄱㅏ_"
    assert len(jamo_tokens("라아심")) == 7
    print("ok, jamo vocab", len(JAMO_VOCAB))
