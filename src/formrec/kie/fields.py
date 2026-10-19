"""KIE 공용 정의 — 키 목록 · 프롬프트 · 출력 파싱 · 채점 정규화(모든 KIE 모듈이 같은 정의를 쓴다)."""
import re

KEYS = ["이름", "생년월일", "주소", "시도", "시군구"]
QSETS = ("xk", "xkp")          # xk = 사람 영역 이미지, xkp = 전체 페이지(검출 없음)


def prompt(keys):
    """키 이름만 준다(설명 없음 — 서식 관례는 학습으로 익힌다)."""
    return "KIE: " + ", ".join(keys)


def parse(raw, keys):
    """'키: 값' 줄 → {키: 값}. 요청한 키만, 처음 나온 것만."""
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*([^:：]+)[:：]\s*(.*)$", line)
        if m and m.group(1).strip() in keys and m.group(1).strip() not in out:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def parse_loose(raw, keys):
    """parse와 같되 줄 앞의 목록 기호(-, *)를 허용(학습하지 않은 범용 VLM 출력용)."""
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*[-*]?\s*([^:：]+)[:：]\s*(.*)$", line)
        if m and m.group(1).strip() in keys and m.group(1).strip() not in out:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def norm(k, v):
    """채점 정규화: 공백 정규화, 생년월일은 숫자만. None은 빈 문자열."""
    v = " ".join(str(v or "").split())
    return re.sub(r"\D", "", v) if k == "생년월일" else v


def target_dict(target):
    """정답 문자열('키: 값' 줄) → {키: 값}."""
    return dict(line.split(": ", 1) for line in target.split("\n"))


def fmt(keys, d):
    """{키: 값} → 키 순서의 정답 형식 문자열(값 공백 정규화)."""
    return "\n".join(f"{k}: {' '.join(str(d.get(k, '')).split())}" for k in keys)


def query_ok(keys, target, pred):
    """요청 키가 모두 정답이면 True(값이 없는 키는 오답)."""
    tg = target_dict(target)
    return all(pred.get(k) is not None and norm(k, pred.get(k, "")) == norm(k, tg[k]) for k in keys)
