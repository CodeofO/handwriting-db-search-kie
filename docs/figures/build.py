"""README 그림 생성 — 그림마다 HTML을 만들고 헤드리스 Chrome으로 2배 해상도 PNG(docs/images/)로 렌더링한다.

예시 손글씨 · 이름 · 주소는 모두 가상(손글씨 글꼴로 그림)이며 원천 데이터 이미지는 쓰지 않는다(재배포 불가).
수치는 docs/results.md 와 같다.
사용: CHROME=/path/to/chrome-headless-shell python docs/figures/build.py
"""
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "html", HERE.parent / "images"
CHROME = os.environ.get("CHROME", "chrome-headless-shell")
W = 1600

BLUE, ORANGE = "#2a78d6", "#eb6834"         # 제안 / 비교군(생성형 최고) — 색각 이상 검사 통과 조합
INK, INK2, MUTED, LINE, PANEL = "#111111", "#52514e", "#8a8985", "#e7e6e2", "#f7f7f5"
GOOD, BAD = "#008300", "#e34948"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Nanum+Pen+Script&family=Caveat:wght@500&family=JetBrains+Mono:wght@400;600&display=block');
*{{box-sizing:border-box}}
body{{margin:0;background:#fff;font-family:'Noto Sans KR',sans-serif;color:{INK};-webkit-font-smoothing:antialiased}}
.fig{{width:{W}px;padding:44px 52px}}
.t{{font-size:30px;font-weight:700;margin:0 0 8px;letter-spacing:-.01em}}
.s{{font-size:18px;color:{INK2};margin:0 0 30px}}
.hw{{font-family:'Nanum Pen Script','Caveat',cursive;color:#1d2a5b;line-height:1}}
.hwl{{font-family:'Caveat',cursive;color:#1d2a5b;line-height:1}}
.field{{display:inline-flex;align-items:center;border:1.5px solid #d6d5cf;background:#fcfcf9;border-radius:6px;padding:6px 18px 2px;height:64px}}
.lab{{font-size:14px;color:{MUTED};margin-bottom:6px}}
.card{{border:1px solid {LINE};border-radius:14px;padding:22px 24px;background:#fff}}
.tag{{font-size:14px;font-weight:700;letter-spacing:.03em;color:{MUTED};margin-bottom:6px}}
.h{{font-size:21px;font-weight:700;margin:0 0 14px}}
.mono{{font-family:'JetBrains Mono',monospace}}
.bad{{color:{BAD}}} .good{{color:{GOOD}}} .blue{{color:{BLUE}}}
.note{{font-size:15px;color:{INK2};line-height:1.55}}
.chip{{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:16px;background:{PANEL};border:1px solid {LINE};border-radius:8px;padding:6px 12px}}
.arrow{{color:{MUTED};font-size:26px;text-align:center}}
table.db{{border-collapse:collapse;width:100%;font-size:15px}}
table.db td{{padding:7px 8px;border-bottom:1px solid {LINE}}}
table.db tr.hit td{{background:#e8f1fb;font-weight:700}}
.legend{{display:flex;gap:22px;font-size:15px;color:{INK2};align-items:center}}
.sw{{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:7px;vertical-align:-2px}}
"""


def page(body, h):
    return f"<!doctype html><html><head><meta charset=utf-8><style>{CSS} body{{height:{h}px;overflow:hidden}}</style></head><body><div class=fig>{body.replace('/>', ' />')}</div></body></html>", h


def hw_field(label, text, size=46, width=None, latin=False):
    w = f"width:{width}px;" if width else ""
    cls = "hwl" if latin else "hw"
    return f"<div><div class=lab>{label}</div><div class=field style='{w}'><span class={cls} style='font-size:{size}px'>{text}</span></div></div>"


# ── 1. 개요
def fig_overview():
    inp = ("<div style='display:flex;gap:18px;align-items:flex-end;margin-bottom:30px'>"
           + hw_field("이름", "김하늘", width=190) + hw_field("생년월일", "960314", width=210)
           + hw_field("주소", "서울특별시 마포구 하늘길 27", width=520)
           + f"<div class=note style='margin-left:10px;max-width:420px'>입력: 서식 칸의 손글씨(칼럼 3개). 정답 레코드는 이미 고객 · 주민 DB에 있다.</div></div>")
    gen = (f"<div class=card><div class=tag>비교군</div><div class=h>생성형 OCR — 글자를 새로 쓴다</div>"
           f"<div class=mono style='font-size:19px;line-height:1.8'>김하<span class=bad style='text-decoration:underline'>눌</span> · 960314<br>서울특별시 마포구 하늘길 27</div>"
           f"<div style='margin:16px 0 10px;font-size:17px' class=bad>✗ 한 글자 오류 → DB에 없는 사람</div>"
           f"<div class=note>모든 칼럼 · 모든 글자가 맞아야 정답. 글자 정확도 99%여도 27자면 레코드는 약 76%.</div></div>")
    rows = [("김하늘", "960314", "서울특별시 마포구 하늘길 27", "0.91", True), ("김하눌", "960314", "서울특별시 마포구 하늘길 27", "0.84", False),
            ("김하늘", "960814", "서울특별시 마포구 하늘길 27", "0.80", False), ("김하늘", "960314", "서울특별시 마포구 하늘길 21", "0.78", False)]
    tb = "".join(f"<tr class={'hit' if h else ''}><td>{a}</td><td class=mono>{b}</td><td>{c}</td><td class=mono style='text-align:right'>{s}</td></tr>" for a, b, c, s, h in rows)
    ret = (f"<div class=card style='border-color:{BLUE}'><div class=tag style='color:{BLUE}'>방법 ①</div><div class=h>DB 임베딩 검색 — 레코드를 찾는다</div>"
           f"<table class=db>{tb}</table>"
           f"<div style='margin:14px 0 8px;font-size:17px' class=good>✓ 결과는 항상 실제 DB 레코드</div>"
           f"<div class=note>칼럼 묶음 전체를 DB 벡터와 비교 → 위치 정렬 재채점 → 수락 판정. 한 글자 · 한 칼럼만 다른 이웃을 구분하도록 학습.</div></div>")
    kie = (f"<div class=card style='border-color:{BLUE}'><div class=tag style='color:{BLUE}'>방법 ②</div><div class=h>KIE — 묻고, 읽고, DB로 검증한다</div>"
           f"<div style='display:flex;gap:10px;align-items:center;margin-bottom:12px'><div style='width:64px;height:86px;border:1px solid {LINE};border-radius:4px;background:repeating-linear-gradient(#fff 0 9px,#eeede8 9px 10px)'></div>"
           f"<span class=arrow>+</span><span class=chip>KIE: 이름, 주소</span></div>"
           f"<div class=mono style='font-size:17px;line-height:1.7'>이름: 김하늘<br>주소: 서울특별시 마포구 하늘길 2<span class=bad style='text-decoration:underline'>1</span></div>"
           f"<div class=note style='margin:6px 0'>↓ 요청 키를 한 레코드로 묶어 DB와 대조</div>"
           f"<div class=mono style='font-size:17px'>주소: … 하늘길 2<span class=good>7</span> <span class=good>✓</span></div>"
           f"<div class=note style='margin-top:10px'>검출 · crop 없이 페이지 전체 + 키 목록만. 한 칸 오독을 다른 칸이 바로잡는다.</div></div>")
    body = (f"<div class=t>정답이 DB에 있는 손글씨 서식: 글자를 새로 쓰지 말고, 찾거나 검증한다</div>"
            f"<div class=s>Handwriting DB Search &amp; KIE — 임베딩 기반 레코드 검색과 검출 없는 VLM 키-값 추출</div>"
            + inp + f"<div style='display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:22px'>{gen}{ret}{kie}</div>"
            + f"<div class=note style='margin-top:18px;color:{MUTED}'>예시 손글씨 · 이름 · 주소는 설명용 가상 데이터입니다.</div>")
    return page(body, 720)


# ── 2. 검색 모델 구조
def box(title, sub, accent=False, w=None):
    st = f"border-color:{BLUE};background:#f3f8fe" if accent else ""
    ww = f"width:{w}px;" if w else ""
    return (f"<div class=card style='padding:16px 18px;{st}{ww}'><div style='font-size:18px;font-weight:700;margin-bottom:6px'>{title}</div>"
            f"<div class=note style='font-size:14.5px'>{sub}</div></div>")


def fig_retrieval_model():
    A = lambda: f"<div class=arrow>→</div>"
    img_in = ("<div style='display:flex;flex-direction:column;gap:6px'>"
              + "".join(f"<div class=field style='height:40px;padding:2px 12px;width:{w}px'><span class=hw style='font-size:30px'>{t}</span></div>"
                        for t, w in (("김하늘", 120), ("960314", 140), ("서울특별시 마포구 하늘길 27", 300))) + "</div>")
    db_in = f"<div class=chip style='font-size:14px;line-height:1.6'>김하늘␞960314␞<br>서울특별시 마포구 하늘길 27</div>"
    row = lambda *xs: "<div style='display:grid;grid-template-columns:320px 40px 300px 40px 270px;align-items:center'>" + "".join(xs) + "</div>"
    left = ("<div style='display:flex;flex-direction:column;gap:26px'>"
            + row(f"<div><div class=lab>칼럼 이미지 묶음 (칼럼 수 · 누락 무관)</div>{img_in}</div>", A(),
                  box("이미지 인코더", "64×1280 칼럼 이미지 → 열 토큰 320개. 세로를 평균하지 않고 8줄을 채널로 접어 작은 획을 보존"), A(),
                  box("공용 쿼리 16개", "칼럼을 지정하지 않는 쿼리가 역할을 스스로 나눔 → 1,024차원 벡터", True))
            + row(f"<div><div class=lab>DB 레코드 (미리 벡터로 변환)</div>{db_in}</div>", A(),
                  box("텍스트 인코더", "한글 = 초·중·종성 자모, 그 밖 = 글자. 토큰마다 칼럼 번호 · 글자 순번 · 자모 역할"), A(),
                  box("공용 쿼리 16개", "같은 구조 · 같은 1,024차원 공간 — 검색 = 내적 1번", True))
            + "</div>")
    right = ("<div style='display:flex;flex-direction:column;gap:10px'>"
             + box("① 벡터 검색", "DB 전체와 행렬곱 1번 → 상위 K 후보") + "<div class=arrow>↓</div>"
             + box("② 위치 정렬 재채점", "텍스트 토큰은 그 칼럼의 예상 위치 근처 열 토큰하고만 비교 + 글자 검증(CTC) 점수") + "<div class=arrow>↓</div>"
             + box("③ 칼럼별 수락 판정", "한 칼럼만 틀린 후보를 잡아 거부 → 자동 처리 / 사람 확인", True) + "</div>")
    body = (f"<div class=t>위치 정렬 레코드 검색 모델</div>"
            f"<div class=s>가중치 하나로 검색 → 정밀 비교 → 수락 판정. 학습 = 대칭 InfoNCE + 어려운 음성(한 글자 · 한 토큰 · 한 칼럼만 다른 가짜 레코드) 재채점 + CTC + 수락 판정</div>"
            f"<div style='display:grid;grid-template-columns:1010px 50px 1fr;align-items:center'>{left}<div class=arrow style='font-size:34px'>⟶</div>{right}</div>")
    return page(body, 640)


# ── 3. KIE 예시 (가상 서식)
def fig_kie_example():
    def r(label, val, size=40, h=70):
        return (f"<tr><td style='width:150px;background:{PANEL};font-size:16px;padding:0 14px;border:1px solid #cfcec8'>{label}</td>"
                f"<td style='border:1px solid #cfcec8;padding:6px 16px;height:{h}px'><span class=hw style='font-size:{size}px'>{val}</span></td></tr>")
    form = (f"<div style='width:600px;border:1px solid #d6d5cf;border-radius:4px;padding:30px 34px;background:#fff;box-shadow:0 1px 0 #eee'>"
            f"<div style='text-align:center;font-size:24px;font-weight:700;letter-spacing:.3em;margin-bottom:6px'>서비스 신청서</div>"
            f"<div style='text-align:center;font-size:13px;color:{MUTED};margin-bottom:20px'>※ 해당 칸에 정자로 적어 주십시오.</div>"
            f"<div style='font-size:15px;font-weight:700;margin:6px 0'>1. 신청인</div>"
            f"<table style='border-collapse:collapse;width:100%'>{r('성명', '김하늘')}{r('생년월일', '960314')}{r('주소', '서울특별시 마포구 하늘길 27', 34)}{r('연락처', '010-1234-5678', 34)}</table>"
            f"<div style='font-size:15px;font-weight:700;margin:20px 0 6px'>2. 신청 내용</div>"
            f"<table style='border-collapse:collapse;width:100%'>{r('신청 구분', '신규', 36)}{r('직장명', '', 36)}</table>"
            f"<div style='font-size:13px;color:{INK2};line-height:1.7;margin-top:18px'>위와 같이 신청합니다. 개인정보 수집 · 이용에 동의합니다. &nbsp;[ ✓ ] 동의</div>"
            f"<div style='text-align:right;margin-top:14px;font-size:15px'>신청인 <span class=hw style='font-size:34px'>김하늘</span> (서명)</div></div>")

    def q(keys, out, note=""):
        lines = "<br>".join(out)
        return (f"<div class=card style='padding:16px 20px'><div class=chip>KIE: {keys}</div>"
                f"<div class=arrow style='font-size:20px;text-align:left;margin:6px 0 2px 6px'>↓</div>"
                f"<div class=mono style='font-size:17px;line-height:1.7'>{lines}</div>"
                + (f"<div class=note style='margin-top:6px;font-size:14px'>{note}</div>" if note else "") + "</div>")
    right = ("<div style='display:flex;flex-direction:column;gap:16px;width:760px'>"
             + q("주소, 이름", ["주소: 서울특별시 마포구 하늘길 27", "이름: 김하늘"], "키 순서대로 출력 · 키 조합과 순서는 질의마다 무작위")
             + q("생년월일", ["생년월일: 960314"])
             + q("직장명, 성명", ["직장명: [없음]", "성명: 김하늘"], "빈 칸 · 서식에 없는 키는 '[없음]'(확장 데이터에서 학습 중)")
             + f"<div class=note style='padding:4px 6px'>입력은 <b>페이지 전체</b>뿐 — 칸 검출 · crop · 좌표를 쓰지 않는다. 출력 값은 요청 키를 한 레코드로 묶어 DB와 대조해 교정한다.</div></div>")
    body = (f"<div class=t>KIE: 페이지 전체 + 키 목록 → 키 순서대로 값</div>"
            f"<div class=s>키 조건부로 파인튜닝한 VLM(PaddleOCR-VL-1.6 · OvisOCR2). 서식이 바뀌어도 키 이름만 바꿔 같은 모델을 쓴다.</div>"
            f"<div style='display:flex;gap:44px;align-items:flex-start'>{form}{right}</div>"
            f"<div class=note style='margin-top:16px;color:{MUTED}'>서식 · 손글씨 · 개인정보는 설명용 가상 예시입니다.</div>")
    return page(body, 940)


# ── 차트 도우미 (인라인 SVG)
def bars(title, cats, a, b, w=700, h=380, ymax=100):
    pl, pr, pt, pb = 48, 10, 20, 64
    iw, ih = w - pl - pr, h - pt - pb
    gw = iw / len(cats); bw = 54
    y = lambda v: pt + ih * (1 - v / ymax)
    s = [f"<svg width={w} height={h} viewBox='0 0 {w} {h}' font-family='Noto Sans KR'>"]
    for g in (0, 50, 100):
        s.append(f"<line x1={pl} x2={w - pr} y1={y(g)} y2={y(g)} stroke='{LINE}' stroke-width=1/>"
                 f"<text x={pl - 10} y={y(g) + 5} font-size=14 fill='{MUTED}' text-anchor=end>{g}</text>")
    for i, c in enumerate(cats):
        cx = pl + gw * i + gw / 2
        for k, (v, col) in enumerate(((a[i], BLUE), (b[i], ORANGE))):
            x = cx - bw - 1 + k * (bw + 2)
            top = y(v); r = 4
            s.append(f"<path d='M{x},{y(0)} V{top + r} Q{x},{top} {x + r},{top} H{x + bw - r} Q{x + bw},{top} {x + bw},{top + r} V{y(0)} Z' fill='{col}'/>"
                     f"<text x={x + bw / 2} y={top - 8} font-size=15 font-weight=700 fill='{INK}' text-anchor=middle>{v:.2f}</text>")
        s.append(f"<text x={cx} y={h - pb + 26} font-size=15 fill='{INK2}' text-anchor=middle>{c[0]}</text>"
                 f"<text x={cx} y={h - pb + 46} font-size=13 fill='{MUTED}' text-anchor=middle>{c[1]}</text>")
    s.append("</svg>")
    return f"<div><div style='font-size:19px;font-weight:700;margin:0 0 6px'>{title}</div>{''.join(s)}</div>"


def legend(a="제안 모델", b="생성형 OCR 최고"):
    return (f"<div class=legend><span><span class=sw style='background:{BLUE}'></span>{a}</span>"
            f"<span><span class=sw style='background:{ORANGE}'></span>{b}</span></div>")


def fig_results():
    cats = [("한국어 서식", "AI Hub · 1,751명"), ("덴마크어 이름", "HANA · 55,822"), ("영어 문장 줄", "IAM · 2,915")]
    body = (f"<div class=t>DB 임베딩 검색 vs 파인튜닝한 생성형 OCR 3종</div>"
            f"<div class=s>test 전체 · 비교군 = PP-OCRv5 · PaddleOCR-VL-1.6 · OvisOCR2 중 지표별 최고값. 자동화율 = 정밀도 99% 목표 임계값(tune에서 결정)을 test에 적용</div>"
            + legend() + "<div style='display:flex;gap:60px;margin-top:18px'>"
            + bars("레코드 정확도 (%)", cats, [99.25, 96.88, 98.66], [96.75, 85.52, 58.83])
            + bars("자동화율 @ 정밀도 99% (%)", cats, [99.97, 86.51, 99.59], [87.99, 32.57, 9.19]) + "</div>")
    return page(body, 630)


def lines(title, xs, a, b, w=470, h=330, ylo=40, yhi=100):
    pl, pr, pt, pb = 44, 70, 16, 40
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda i: pl + iw * i / (len(xs) - 1)
    Y = lambda v: pt + ih * (1 - (v - ylo) / (yhi - ylo))
    s = [f"<svg width={w} height={h} viewBox='0 0 {w} {h}' font-family='Noto Sans KR'>"]
    for g in range(ylo, yhi + 1, 20):
        s.append(f"<line x1={pl} x2={w - pr} y1={Y(g)} y2={Y(g)} stroke='{LINE}'/><text x={pl - 8} y={Y(g) + 5} font-size=13 fill='{MUTED}' text-anchor=end>{g}</text>")
    for i, x in enumerate(xs):
        s.append(f"<text x={X(i)} y={h - 14} font-size=14 fill='{INK2}' text-anchor=middle>{x}</text>")
    for v, col, lab in ((a, BLUE, "제안"), (b, ORANGE, "생성형 최고")):
        pts = " ".join(f"{X(i)},{Y(t)}" for i, t in enumerate(v))
        s.append(f"<polyline points='{pts}' fill=none stroke='{col}' stroke-width=2.5/>")
        s += [f"<circle cx={X(i)} cy={Y(t)} r=5 fill='{col}' stroke='#fff' stroke-width=2/>" for i, t in enumerate(v)]
        s.append(f"<text x={X(len(v) - 1) + 10} y={Y(v[-1]) + 5} font-size=14 font-weight=700 fill='{INK}'>{v[-1]:.1f}</text>")
    s.append("</svg>")
    return f"<div><div style='font-size:18px;font-weight:700'>{title[0]}</div><div class=note style='margin-bottom:6px'>{title[1]}</div>{''.join(s)}</div>"


def fig_ladder():
    body = (f"<div class=t>길이 · 칼럼 수가 늘수록 생성형은 떨어지고, 검색은 버틴다</div>"
            f"<div class=s>레코드 정확도(%) · test. 생성형은 모든 글자가 맞아야 정답이라 오류가 쌓이고, 검색은 레코드 전체를 한 번에 대조한다.</div>"
            + legend("제안 모델", "생성형 OCR 최고 (단계별)") + "<div style='display:flex;gap:30px;margin-top:16px'>"
            + lines(("한국어 서식 · 길이", "이름 3자 → 주소 6토큰"), ["L1", "L2", "L3", "L4", "L5", "L6"], [98.06, 99.60, 99.37, 99.43, 99.07, 98.94], [98.63, 99.03, 96.92, 96.40, 91.67, 93.01])
            + lines(("덴마크어 이름 · 한 이미지 안 칼럼", "2단어 → 5단어 이상"), ["H2", "H3", "H4", "H5+"], [97.38, 96.86, 96.83, 93.41], [89.91, 87.59, 79.12, 59.38])
            + lines(("영어 문장 줄 · 단어 수", "5단어 이하 → 12단어 이상"), ["I1", "I2", "I3", "I4", "I5"], [93.98, 98.89, 99.40, 99.84, 97.25], [70.83, 60.82, 60.86, 55.31, 50.46])
            + "</div>")
    return page(body, 600)


def dumbbell(rows, lo, hi, w=760, label_w=250, unit="", left_lab="", right_lab=""):
    rh, pt = 52, 30
    h = pt + rh * len(rows) + 30
    X = lambda v: label_w + (w - label_w - 70) * (v - lo) / (hi - lo)
    s = [f"<svg width={w} height={h} viewBox='0 0 {w} {h}' font-family='Noto Sans KR'>"]
    for g in range(lo, hi + 1, 5 if hi - lo <= 30 else 10):
        s.append(f"<line x1={X(g)} x2={X(g)} y1={pt - 10} y2={h - 26} stroke='{LINE}'/><text x={X(g)} y={h - 8} font-size=13 fill='{MUTED}' text-anchor=middle>{g}</text>")
    for i, (name, sub, a, b, col) in enumerate(rows):
        yy = pt + rh * i + rh / 2
        s.append(f"<text x=0 y={yy - 2} font-size=15 font-weight=700 fill='{INK}'>{name}</text><text x=0 y={yy + 16} font-size=12.5 fill='{MUTED}'>{sub}</text>"
                 f"<line x1={X(a)} x2={X(b)} y1={yy} y2={yy} stroke='{col}' stroke-width=3 stroke-opacity=.45/>"
                 f"<circle cx={X(a)} cy={yy} r=7 fill='#fff' stroke='{col}' stroke-width=2.5/><circle cx={X(b)} cy={yy} r=7.5 fill='{col}' stroke='#fff' stroke-width=2/>"
                 f"<text x={X(b) + (14 if b >= a else -14)} y={yy + 5} font-size=14.5 font-weight=700 fill='{INK}' text-anchor={'start' if b >= a else 'end'}>{b:.1f}</text>"
                 f"<text x={X(a) + (-12 if b >= a else 12)} y={yy + 5} font-size=13 fill='{INK2}' text-anchor={'end' if b >= a else 'start'}>{a:.1f}</text>")
    s.append("</svg>")
    return "".join(s)


def fig_multicolumn():
    f3 = ("<div style='display:flex;gap:10px'>" + "".join(f"<div class=field style='height:46px;padding:2px 12px'><span class=hw style='font-size:32px'>{t}</span></div>"
                                                          for t in ("김하늘", "960314", "서울특별시 마포구 하늘길 27")) + "</div>")
    c3 = "<div class=field style='height:46px;padding:2px 14px'><span class=hw style='font-size:32px'>김하늘 &nbsp; 960314 &nbsp; 서울특별시 마포구 하늘길 27</span></div>"
    left = (f"<div style='width:640px'><div class=lab>F3 · 칼럼마다 이미지 1장 (서식 칸)</div>{f3}"
            f"<div class=lab style='margin-top:22px'>C3 · 같은 칼럼을 한 줄 이미지로 (표의 한 행)</div>{c3}"
            f"<div class=note style='margin-top:22px'>제안 모델은 칼럼을 지정하지 않는 공용 쿼리라 구조 변경 없이 두 경우를 같은 모델로 처리한다.<br>"
            f"칼럼 하나를 비워도 99.37–99.89% · 생성형 F3 오답의 88.8–97.3%는 '한 칼럼만' 틀린 레코드.</div></div>")
    rows = [("제안 모델", "위치 정렬 레코드 검색", 99.43, 99.54, BLUE), ("PaddleOCR-VL-1.6", "생성형", 94.40, 93.15, ORANGE),
            ("PP-OCRv5", "생성형", 92.35, 89.03, ORANGE), ("OvisOCR2", "생성형", 93.55, 87.26, ORANGE)]
    right = (f"<div><div style='font-size:18px;font-weight:700'>레코드 정확도 (%) · F3 → C3</div>"
             f"<div class=legend style='margin:6px 0 4px'><span>○ 칼럼별 이미지 (F3)</span><span>● 한 줄 이미지 (C3)</span></div>{dumbbell(rows, 85, 100)}</div>")
    body = (f"<div class=t>멀티 칼럼: 칼럼이 이미지 여러 장이든 한 줄이든 같은 모델</div>"
            f"<div class=s>AI Hub test 1,751명 · 네 모델 모두 C3 이미지로 같은 예산 재학습</div>"
            f"<div style='display:flex;gap:56px;align-items:flex-start'>{left}{right}</div>")
    return page(body, 560)


def fig_kie_results():
    rows = [("KIE · OvisOCR2", "검출 없음 · 페이지 전체", 84.8, 98.1, BLUE), ("KIE · PaddleOCR-VL-1.6", "검출 없음 · 페이지 전체", 78.6, 96.8, BLUE),
            ("crop + OvisOCR2", "정답 위치 crop(검출 완벽 가정)", 93.8, 98.6, ORANGE), ("crop + PaddleOCR-VL-1.6", "정답 위치 crop(검출 완벽 가정)", 94.5, 98.3, ORANGE),
            ("crop + PP-OCRv5", "정답 위치 crop(검출 완벽 가정)", 93.5, 98.3, ORANGE)]
    side = (f"<div class=note style='width:520px'><div style='font-size:18px;font-weight:700;color:{INK};margin-bottom:10px'>읽는 법</div>"
            f"○ VLM · OCR 출력 그대로 → ● 요청 키를 한 레코드로 묶어 DB와 대조한 뒤.<br><br>"
            f"검출 없이 페이지 전체를 읽는 KIE는 그대로는 대조군보다 낮지만, 레코드 DB 검증 뒤에는 검출이 완벽하다고 가정한 OCR 파이프라인과 거의 같다.<br><br>"
            f"키 순서를 바꿔도 같은 답 94.6% · 값이 다른 키에 붙는 결합 오류 0% · 학습 데이터 10%로도 90.9%.<br><br>"
            f"<span style='color:{MUTED}'>DB에 없는 사람에게는 VLM 자체 정확도(약 67–70%)로 떨어진다 — 한계.</span></div>")
    body = (f"<div class=t>KIE: 검출 없이도, 레코드 DB 검증으로 검출 가정 파이프라인에 근접</div>"
            f"<div class=s>질의 완전 일치(%) · AI Hub 서식 test 195장 · 585 질의</div>"
            f"<div class=legend style='margin-bottom:8px'><span><span class=sw style='background:{BLUE}'></span>제안 (KIE)</span><span><span class=sw style='background:{ORANGE}'></span>대조군 (crop + OCR)</span></div>"
            f"<div style='display:flex;gap:50px;align-items:flex-start'>{dumbbell(rows, 75, 100, w=900, label_w=280)}{side}</div>")
    return page(body, 560)


FIGS = {"overview": fig_overview, "retrieval_model": fig_retrieval_model, "kie_example": fig_kie_example,
        "results_retrieval": fig_results, "ladder": fig_ladder, "multicolumn": fig_multicolumn, "kie_results": fig_kie_results}


def main():
    SRC.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        html, h = fn(); p = SRC / f"{name}.html"; p.write_text(html, encoding="utf-8")
        subprocess.run([CHROME, "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W},{h}",
                        "--virtual-time-budget=15000", f"--screenshot={OUT / (name + '.png')}", p.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("rendered", name, f"{W * 2}x{h * 2}")


if __name__ == "__main__":
    main()
