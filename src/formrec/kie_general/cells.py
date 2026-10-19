"""일반 KIE(인쇄 항목명을 키로) 데이터의 키-값 짝: 표 칸(괘선) 구조 기반.

같은 줄 왼쪽 · 바로 위 규칙은 칸을 넘나들어 짝 정확도가 절반 남짓이었다(예시 점검) → 서식의 칸 구조를 쓴다.
  1) 페이지 이미지에서 가로 · 세로 괘선을 형태학 연산으로 찾고, 괘선으로 둘러싸인 칸(연결 요소)을 얻는다.
  2) 인쇄 항목명(OCR 상자 중 손글씨와 겹치지 않는 것) · 손글씨 토큰(라벨)을 중심점이 들어가는 칸에 배정한다.
  3) 손글씨가 있는 칸(값 칸)마다 키:
       ① 같은 칸 안의 인쇄 항목명(단위 · 체크 문구 제외) — '성명(대표자)' 칸 안 아래에 손글씨
       ② 없으면 왼쪽에 붙은 항목명 칸(행 서식)
       ③ 없으면 바로 위에 붙은(인접) 항목명 칸(열 머리) — 떨어진 머리는 쓰지 않음, 여러 행 표의 첫 행은 버림
       ④ 없으면 같은 행에서 왼쪽으로 가장 가까운 항목명 칸
       표의 두 번째 행 이후(바로 위 인접 칸도 손글씨 칸)는 열 머리가 모호해 버린다.
     값 = 그 칸 안 손글씨를 줄 · x 순으로. 한 항목명 칸에 값 칸이 여럿 붙으면 가장 가까운 것만(층 · 호 같은 보조 칸 제외).
  4) 키 정규화 · 같은 키가 한 페이지에 둘 이상이면 왼쪽 구역 이름을 앞에 붙이고, 그래도 겹치면 버림.
  5) 빈 칸(네거티브 B) = 손글씨가 없는 항목명 칸 중 오른쪽 · 아래 이웃 칸에도 손글씨가 없는 것.
모델 입력은 페이지 이미지뿐이며, 괘선 · bbox는 학습 데이터(정답) 구축에만 쓴다. cv2 필요(paddle 환경).
"""
import json
import re

import numpy as np

# 항목명이 아닌 인쇄 글자(단위 · 체크 문구 · 서명란)
STOP = {"원", "층", "호", "자가", "타가", "동의함", "동의하지않음", "예", "아니오", "부", "여", "년", "월", "일", "명", "개", "건", "부터", "까지", "서명", "(인)",
        "서명또는인", "(서명또는인)", "귀하", "확인", "해당란에", "합계", "계", "번호"}


def ov1(a0, a1, b0, b1):
    """1차원 구간 겹침 길이."""
    return max(0, min(a1, b1) - max(a0, b0))


def norm_key(t):
    """인쇄 항목명 정규화: 원문자 · 번호 · 기호 · 영문 병기 제거, 한글 사이 띄어쓰기 제거."""
    t = re.sub(r"\([^()]*[A-Za-z][^()]*\)", " ", t)                 # 영문 병기 괄호 · 영문 단어 제거(한영 병기 서식)
    t = " ".join(w for w in t.split() if not re.search(r"[A-Za-z]", w))
    t = re.sub(r"[①-⑳⑴-⒇❶-❿]", "", t)
    t = re.sub(r"^\s*[\d]+[.)]\s*", "", t); t = re.sub(r"^\d{1,2}\s+(?=[가-힣\d])", "", t)   # 번호 매김 '8 1일통상임금' → '1일통상임금'
    t = re.sub(r"(?<=[가-힣])\d\)", "", t)                                # 각주 표시 '주소1)' → '주소'
    t = re.sub(r"\s+\d$", "", t)                                         # 끝의 떨어진 숫자(':' 오인식 등)
    t = re.sub(r"[※*•·:：\[\]]", " ", t)
    t = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", t)
    t = re.sub(r"\s*\(\s*", "(", t); t = re.sub(r"\s*\)\s*(?=[가-힣(]|$)", ")", t)   # '성명 (법인명)' = '성명(법인명)'
    if t.count("(") != t.count(")"):
        t = re.sub(r"[()]", "", t)                                      # 괄호 짝이 깨지면(각주 · 줄바꿈) 괄호만 제거
    t = re.sub(r"^\(([^()]*)\)$", r"\1", t.strip())                    # '(현거주지)' → '현거주지'
    return " ".join(t.split()).strip()


def cells_of(gray, with_lines=False, hw=()):
    """괘선 → 칸 상자 목록 [(x0, y0, x1, y1)] (원본 해상도). with_lines면 가로선 마스크도.
    hw = 손글씨 상자: 안쪽을 지워 숫자 '1' · 밑줄 같은 필기 획이 괘선으로 잡히지 않게 한다.
    한국 공문서 표는 좌우 바깥 세로선이 없는 경우가 많다 → 긴 가로선 양 끝을 위 · 아래 이웃 가로선과 세로로 이어 칸을 닫는다."""
    import cv2
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)   # 연한 회색 괘선도 잡게(국소 평균 기준)
    for _, x0, y0, x1, y1 in hw:
        dx, dy = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        bw[int(y0 + dy):int(y1 - dy), int(x0 + dx):int(x1 - dx)] = 0
    hz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 30), 1)))
    vt = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 90))))
    hz = cv2.dilate(hz, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))); vt = cv2.dilate(vt, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)))   # 끊긴 선 잇기
    n, _, st, _ = cv2.connectedComponentsWithStats(hz, connectivity=8)
    L = sorted([(st[i][1], st[i][0], st[i][0] + st[i][2] - 1) for i in range(1, n) if st[i][2] >= 0.4 * W and st[i][3] < 40], key=lambda t: t[0])
    for (ya, a0, a1), (yb, b0, b1) in zip(L, L[1:]):
        if 0 < yb - ya < 0.25 * H:
            if abs(a0 - b0) < 0.03 * W:
                cv2.line(vt, (int(min(a0, b0)), int(ya)), (int(min(a0, b0)), int(yb) + 3), 255, 3)
            if abs(a1 - b1) < 0.03 * W:
                cv2.line(vt, (int(max(a1, b1)), int(ya)), (int(max(a1, b1)), int(yb) + 3), 255, 3)
    grid = cv2.bitwise_or(hz, vt)
    n, lab, st, _ = cv2.connectedComponentsWithStats(255 - grid, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 25 or h < 18 or w * h > 0.5 * W * H or x == 0 or y == 0 or x + w >= W - 1 or y + h >= H - 1:
            continue
        out.append((x, y, x + w, y + h))
    return (out, hz) if with_lines else out


def cell_of(cells, cx, cy):
    """점 (cx, cy)를 포함하는 가장 작은 칸의 번호(없으면 None)."""
    best, ba = None, 1e18
    for i, (x0, y0, x1, y1) in enumerate(cells):
        if x0 <= cx <= x1 and y0 <= cy <= y1 and (x1 - x0) * (y1 - y0) < ba:
            best, ba = i, (x1 - x0) * (y1 - y0)
    return best


def pairs_cell(args, debug=False):
    """(페이지 이미지 경로, OCR 상자 JSON) → (서식 이름, {키: 값}, 빈 칸 키 목록, 인쇄된 항목명 목록)."""
    img, boxes = args
    from PIL import Image
    lab = img.replace("원천데이터", "라벨링데이터/라벨").replace(".png", ".json")
    try:
        Hd = [(b["data"], min(b["x"]), min(b["y"]), max(b["x"]), max(b["y"])) for b in json.load(open(lab))["bbox"]]
        gray = np.asarray(Image.open(img).convert("L"))
    except Exception:
        return (None, {}, [], []) if not debug else (None, {}, [], [], {})
    B = json.loads(boxes)
    pr = []
    for t, x0, y0, x1, y1, s in B:
        if t.startswith("#ERR") or s < 0.5:
            continue
        a = max(1, (x1 - x0) * (y1 - y0))
        if any(ov1(x0, x1, h[1], h[3]) * ov1(y0, y1, h[2], h[4]) / a > 0.3 for h in Hd):
            continue
        pr.append([t, x0, y0, x1, y1])
    Hp = gray.shape[0]
    m = re.search(r"서\s*식\s*(\d+)", " ".join(b[0] for b in B if b[2] < 0.2 * Hp))   # 손글씨와 겹쳐 인쇄 목록에서 빠진 경우도 있어 전체 OCR에서
    if m:
        tpl = "서식" + m.group(1)
    else:
        tt = [b for b in pr if b[2] < 0.25 * Hp and re.search(r"[가-힣]{3}", b[0]) and not re.search(r"\d", b[0])]
        tpl = re.sub(r"[\s\[\]]+", "", max(tt, key=lambda b: b[4] - b[2])[0])[:20] if tt else "?"
    C, HZ = cells_of(gray, with_lines=True, hw=Hd)
    lab_in, hw_in = {}, {}
    for b in pr:
        c = cell_of(C, (b[1] + b[3]) / 2, (b[2] + b[4]) / 2)
        if c is not None:
            lab_in.setdefault(c, []).append(b)
    for h in Hd:
        c = cell_of(C, (h[1] + h[3]) / 2, (h[2] + h[4]) / 2)
        if c is not None:
            hw_in.setdefault(c, []).append(h)

    def label_text(c):
        bs = sorted(lab_in.get(c, []), key=lambda b: (round((b[2] + b[4]) / 2 / 40), b[1]))
        t = norm_key(" ".join(b[0] for b in bs))
        return t if 2 <= len(t) <= 20 and re.search(r"[가-힣]", t) and t not in STOP else ""
    W = gray.shape[1]
    labc = {c for c in lab_in if label_text(c) and c not in hw_in}     # 손글씨 없는 항목명 칸
    sect = {c for c in labc if C[c][2] < 0.16 * W}
    # 왼쪽 구역 이름(인적 사항 · 정정할 사항 · 신청인 …): 바깥 괘선이 없어 칸으로 안 잡히는 경우가 많다 → 왼쪽 열 인쇄 글자를 세로로 이어 묶고,
    # 구역 범위 = 그 묶음 위 · 아래로 가장 가까운 '왼쪽 열을 가로지르는 가로 괘선' 사이
    lc = sorted([b for b in pr if b[3] < 0.16 * W], key=lambda b: b[2]); grp = []
    for b in lc:
        if grp and b[2] - grp[-1][4] < 1.2 * (b[4] - b[2]) and ov1(b[1], b[3], grp[-1][1], grp[-1][3]) > 0:
            g = grp[-1]; g[0] += b[0]; g[4] = max(g[4], b[4])
        else:
            grp.append(list(b))
    zone = HZ[:, int(0.02 * W):int(0.16 * W)]; seps = np.where((zone > 0).mean(1) > 0.5)[0]
    secs = []
    for g in grp:
        t = norm_key(g[0])
        if not (2 <= len(t) <= 12) or not re.search(r"[가-힣]", t):
            continue
        up = seps[seps < g[2]]; dn = seps[seps > g[4]]
        secs.append((t, up.max() if len(up) else 0, dn.min() if len(dn) else gray.shape[0]))
    cand = {}
    for c, toks in hw_in.items():
        x0, y0, x1, y1 = C[c]; key, src = label_text(c), c; hc = y1 - y0
        if not key:
            if hc > 0.08 * Hp:                                          # 표 아래 자유 기재란 등 큰 영역은 값 칸이 아님
                continue
            LA = [d for d in labc - sect if -8 <= x0 - C[d][2] < 20 and ov1(y0, y1, C[d][1], C[d][3]) >= 0.8 * min(y1 - y0, C[d][3] - C[d][1])]
            if LA:                                                      # ② 왼쪽에 붙은 항목명(행 서식이 가장 흔함 — 위 머리글보다 우선)
                src = max(LA, key=lambda d: C[d][2]); key = label_text(src)
            else:
                # 표의 반복 행: 바로 위 인접 칸도 같은 열의 손글씨 칸이면(두 번째 행 이후) 열 머리가 모호 → 버림
                if any(0 <= y0 - C[d][3] < 15 and ov1(x0, x1, C[d][0], C[d][2]) >= 0.6 * (x1 - x0) for d in hw_in if d != c):
                    continue
                U = [d for d in labc - sect if 0 <= y0 - C[d][3] < 0.6 * hc + 15 and ov1(x0, x1, C[d][0], C[d][2]) >= 0.6 * min(x1 - x0, C[d][2] - C[d][0])]
                if U:                                                   # ③ 바로 위 인접 열 머리
                    if any(0 <= C[d][1] - y1 < 0.6 * hc + 15 and ov1(x0, x1, C[d][0], C[d][2]) >= 0.6 * (x1 - x0) for d in hw_in if d != c):
                        continue                                        # 여러 행 표의 첫 행(계좌번호 · 가족 성명 등): 같은 키에 값이 여럿 → 버림
                    src = max(U, key=lambda d: C[d][3]); key = label_text(src)
                else:                                                   # ④ 같은 행 왼쪽(떨어진) 항목명
                    L = [d for d in labc - sect if C[d][2] <= x0 + 8 and ov1(y0, y1, C[d][1], C[d][3]) >= 0.5 * min(y1 - y0, C[d][3] - C[d][1])]
                    if L:
                        src = max(L, key=lambda d: C[d][2]); key = label_text(src)
        if not key:
            continue
        toks = sorted(toks, key=lambda b: ((b[2] + b[4]) / 2, b[1])); lines = []
        for b in toks:
            if lines and ov1(b[2], b[4], lines[-1][-1][2], lines[-1][-1][4]) > 0.4 * (b[4] - b[2]):
                lines[-1].append(b)
            else:
                lines.append([b])
        val = " ".join(" ".join(x[0] for x in sorted(l, key=lambda b: b[1])) for l in lines).strip()
        if not (1 <= len(val) <= 40):
            continue
        dist = 0 if src == c else abs(C[src][2] - x0) + abs(C[src][3] - y0)
        cand.setdefault(src, []).append((dist, key, val, c))
    kv = {}
    for src, vs in cand.items():                                       # 한 항목명 칸 → 가장 가까운 값 칸 하나만
        d, key, val, c = min(vs)
        cy = (C[c][1] + C[c][3]) / 2
        sec = [t for t, a, b in secs if a <= cy <= b]
        kv.setdefault(key, []).append((val, sec[0] if sec else ""))
    nlab = {}                                                          # 페이지에 인쇄된 같은 항목명 개수(짝이 하나만 잡혀도 모호할 수 있음)
    for c in lab_in:
        t = label_text(c)
        if t:
            nlab[t] = nlab.get(t, 0) + 1
    out = {}
    for k, vs in kv.items():
        if len(vs) == 1 and nlab.get(k, 1) <= 1:
            out[k] = vs[0][0]; continue
        named = {}
        for v, sname in vs:
            named.setdefault(f"{sname} {k}".strip(), []).append(v)
        for k2, v2 in named.items():
            if len(v2) == 1 and k2 != k:
                out[k2] = v2[0]
    present = sorted({label_text(c) for c in lab_in if label_text(c)})
    used = set(cand)
    empty = []
    for c in labc - sect - used:                                       # 빈 칸: 항목명 오른쪽(같은 줄, 다음 항목명 전까지 · 최대 가로 45%) · 아래(칸 높이 2배) · 안에 손글씨 상자가 하나도 없음
        x0, y0, x1, y1 = C[c]; t = label_text(c)
        if nlab.get(t, 0) != 1:                                        # 페이지에 한 번만 인쇄된 항목명만
            continue
        nxt = [C[d][0] for d in labc if d != c and C[d][0] >= x1 - 8 and ov1(y0, y1, C[d][1], C[d][3]) > 0.5 * (y1 - y0)]
        xr = min(nxt + [x1 + 0.45 * W])
        right = any(x1 - 8 <= (h[1] + h[3]) / 2 <= xr and ov1(y0, y1, h[2], h[4]) > 0 for h in Hd)
        below = any(ov1(x0, x1, h[1], h[3]) > 0 and y1 - 8 <= (h[2] + h[4]) / 2 <= y1 + 2 * (y1 - y0) for h in Hd)
        inside = any(ov1(x0, x1, h[1], h[3]) > 0 and ov1(y0, y1, h[2], h[4]) > 0 for h in Hd)
        if not (right or below or inside):
            empty.append(t)
    empty = sorted(set(empty) - set(out))
    if debug:
        return tpl, out, empty, present, dict(cells=C, n_cells=len(C))
    return tpl, out, empty, present
