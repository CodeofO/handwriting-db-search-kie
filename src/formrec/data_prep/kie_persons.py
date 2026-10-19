"""AI Hub 서식 1단계: TO(태블릿) · PO(종이) 서식 페이지 라벨에서 사람 블록(이름 · 생년월일 · 주소 4~6토큰)을 찾는다.

필드 종류 라벨이 없으므로 bbox 순서와 글자 패턴으로 찾는다.
  사람     : 주민번호 쌍(앞 6자리 + 뒤 7자리(TO) 또는 6자리(PO)) → 생년월일 = 앞 6자리
  이름     : 그 앞 3토큰 안의 한글 2~4자(가장 가까운 것)
  주소     : 그 뒤 10토큰 안의 첫 시도부터 — TO = 시도 · 시·군·구 · 도로명 · 번지 · 건물명 · 호, PO = 시도 · 시·군·구 · 읍·면·동 · 읍·면·동
주소 4토큰 이상 · 왼→오 한 줄인 사람만 남긴다. 이름 하나에 주민번호 쌍이 둘이면(모호) 둘 다, 태블릿 서식에서 이름이 흔한 성씨로
시작하지 않거나 '다'로 끝나면(다른 칸 단어) 제외.
세트(subset, 이전 단계 분할 — crop 묶음 crops/kie의 파일 이름): Validation 폴더 → test, Training 페이지 md5 하위 15% → test,
  다음 10% → tune, 나머지 → train. 최종 분할(xa)은 build.split_xa가 page_id로 다시 정한다.
출력: meta/kie_persons.parquet (split, kind, page_id, img, name, birth, n_addr, i_name, i_birth, gap, a1..a6, i_a1..i_a6, folder, subset, official_test, pid)
사용: python -m formrec.data_prep.kie_persons [--limit 200] [--workers 6]
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg

KINDS = {"PO": "P.Paper/O.Form", "TO": "T.Tablet/O.Form"}
SURNAMES = set("김이박최정강조윤장임한오서신권황안송류유전홍고문양손배백허남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부가복태목형피두감음빈동온호범좌팽승간상갈단견당화창")
SIDO = {"서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
        "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"}
SGG = re.compile(r"[가-힣]+(시|군|구)$")
EMD = re.compile(r"[가-힣0-9]+(읍|면|동|리|가)$")
RD = re.compile(r"[가-힣0-9]+(로|길|가|대로|거리)$")
NUM = re.compile(r"\d{1,4}$")
BLD = re.compile(r"[가-힣]+(아파트|빌라|마을|타운|맨션|하우스|주택|오피스텔)$")
NAME = re.compile(r"^[가-힣]{2,4}$")


def pages():
    """서식(PO · TO) 페이지 목록: split(train = Training 폴더, val = Validation 폴더), kind, page_id, img, json."""
    rows = []
    for split in ("1.Training", "2.Validation"):
        for kind, sub in KINDS.items():
            jd = cfg.RAW / split / "라벨링데이터/라벨/HW-OCR/4.Validation" / sub
            idir = cfg.RAW / split / "원천데이터/HW-OCR/4.Validation" / sub
            for j in sorted(jd.glob("*.json")):
                rows.append(dict(split="train" if split.startswith("1") else "val", kind=kind, page_id=j.stem,
                                 img=str(idir / f"{j.stem}.png"), json=str(j)))
    return pd.DataFrame(rows)


def kie_subset(folder, page_id):
    if folder == "Validation":
        return "test"
    h = int(hashlib.md5(page_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < 0.15 else "tune" if h < 0.25 else "train"


def addr_at(t, s, kind="TO"):
    """t[s]가 시도일 때 이어지는 주소 토큰 수(최대 6). 패턴이 끊기면 거기서 멈춘다."""
    if kind == "PO":   # 종이 서식 = 시도 · 시·군·구 · 읍·면·동 · 읍·면·동 (4번째가 숫자면 뒤따르는 전화번호라 제외)
        pats = [lambda u: u in SIDO, SGG.match, EMD.match, EMD.match]
    else:              # 태블릿 서식 = 시도 · 시·군·구 · 도로명 · 번지 · 건물명 · 호
        pats = [lambda u: u in SIDO, SGG.match, RD.match, NUM.match, BLD.match, NUM.match]
    k = 0
    while s + k < len(t) and k < len(pats) and pats[k](t[s + k]):
        k += 1
    return k


def persons_in(b, split, kind, page_id, img):
    """페이지 bbox 목록 → 사람 레코드 목록."""
    t = [x["data"] for x in b]
    out = []
    for r in range(len(t) - 1):
        if not (re.fullmatch(r"\d{6}", t[r]) and re.fullmatch(r"\d{6,7}", t[r + 1])):
            continue
        n = next((j for j in range(r - 1, max(-1, r - 4), -1) if NAME.match(t[j])), None)
        s = next((j for j in range(r + 2, min(r + 12, len(t))) if t[j] in SIDO), None)
        if n is None or s is None:
            continue
        k = addr_at(t, s, kind)
        if k < 4:
            continue
        a = list(range(s, s + k))
        x0 = [min(b[i]["x"]) for i in a]; x1 = [max(b[i]["x"]) for i in a]
        yc = [(min(b[i]["y"]) + max(b[i]["y"])) / 2 for i in a]; h = np.mean([max(b[i]["y"]) - min(b[i]["y"]) for i in a])
        gaps = [x0[i + 1] - x1[i] for i in range(k - 1)]
        if min(gaps) < -5 or (max(yc) - min(yc)) > 0.5 * h:     # 왼→오 한 줄이 아니면 제외
            continue
        rec = dict(split=split, kind=kind, page_id=page_id, img=img, name=t[n], birth=t[r], n_addr=k, i_name=n, i_birth=r, gap=float(np.median(gaps)))
        for i in range(6):
            rec[f"a{i + 1}"] = t[s + i] if i < k else ""
            rec[f"i_a{i + 1}"] = s + i if i < k else -1
        out.append(rec)
    return out


def parse(args):
    split, kind, page_id, img, js = args
    b = json.load(open(js))["bbox"]
    return persons_in(b, split, kind, page_id, img), len(b)


def finalize(D):
    """모호한 이름 · 태블릿 비이름 제외 + folder · subset · official_test · pid."""
    amb = D.duplicated(["page_id", "i_name"], keep=False)
    bad_to = (D.kind == "TO") & (~D.name.str[0].isin(SURNAMES) | D.name.str.endswith("다"))
    print(f"제외: 모호한 이름 {amb.sum()} · 태블릿 비이름 {bad_to.sum()}")
    D = D[~amb & ~bad_to].reset_index(drop=True)
    D["folder"] = D.split.map({"train": "Training", "val": "Validation"})
    D["subset"] = [kie_subset(f, i) for f, i in zip(D.folder, D.page_id)]
    D["official_test"] = D.folder == "Validation"
    D["pid"] = D.page_id + "_" + D.i_name.astype(str) + "_" + D.i_birth.astype(str)
    return D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="시험용: (split, kind)마다 앞 N페이지 → kie_persons_limit{N}.parquet")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    P = pages()
    if a.limit:
        P = P.groupby(["split", "kind"]).head(a.limit)
    rows, ntok = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for o, n in ex.map(parse, zip(P.split, P.kind, P.page_id, P.img, P.json), chunksize=32):
            rows += o; ntok += n
    D = finalize(pd.DataFrame(rows))
    out = cfg.META / ("kie_persons" + (f"_limit{a.limit}" if a.limit else "") + ".parquet")
    cfg.atomic_write(out, lambda p: D.to_parquet(p, index=False))
    print(f"pages {len(P):,} tokens {ntok:,} persons {len(D):,} pages_with_person {D.page_id.nunique():,}")
    print(D.groupby(["kind", "subset"]).size().unstack().to_string())
    print("DONE kie_persons", out)


if __name__ == "__main__":
    main()
