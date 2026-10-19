"""실제 한국어 이름 목록(meta/names_v1.parquet) — 서식 DB의 무작위 이름 방해 항목(kie_db)이 여기서 뽑는다.

정의: AI Hub 053 PR(종이 · 자유) 페이지 순서는 [번호, 시도, 시군구, 읍면동, 신조어, 약어, 이모티콘, 한자] 뒤에 이름 약 12개.
  1) 표준 페이지 = 이름 구간 길이 11~15. 이름 구간 = 마지막 한자 항목 이후
  2) 구간 값 중 한글 2~3자만
  3) 비이름 어휘(TR 전체 + 표준 PR 페이지의 구간 밖 + 비표준 PR 페이지 전체)와 겹치는 값 제외 → 경계에서 섞인 주소 · 신조어 제거
  성씨 목록을 쓰지 않는다.
세트: cfg.subset_of (Training → train/tune 페이지 md5 10%, Validation → test) — 행 순서(세트 · 페이지 · bbox 순)가 kie_db 난수 결과를 정한다.
입력: meta/bbox.parquet (aihub_bbox)
사용: python -m formrec.data_prep.names_v1
"""
import re

import pandas as pd

from formrec.common import workspace as cfg

HANGUL23 = re.compile(r"^[가-힣]{2,3}$")
HAN = re.compile(r"[一-鿿]")
OUT = cfg.META / "names_v1.parquet"


def pr_name_section(texts):
    """PR 페이지 텍스트 순서열에서 이름 구간 인덱스 (마지막 한자 항목 이후)."""
    last_han = max((i for i, t in enumerate(texts) if HAN.search(t)), default=None)
    return [] if last_han is None else list(range(last_han + 1, len(texts)))


def build(df):
    pr = df[df.cell == "PR"].sort_values(["folder", "image_id", "order"])
    nonname = set(df[df.cell == "TR"].text)
    keep = []
    for _, g in pr.groupby(["folder", "image_id"], sort=False):
        t = g.text.tolist()
        sec = set(pr_name_section(t))
        if 11 <= len(sec) <= 15:
            nonname.update(x for i, x in enumerate(t) if i not in sec)
            keep.extend(g.index[i] for i in sec)
        else:
            nonname.update(t)
    n = df.loc[keep]
    n = n[n.text.str.match(HANGUL23) & ~n.text.isin(nonname)].copy()
    n["subset"] = [cfg.subset_of(f, i) for f, i in zip(n.folder, n.image_id)]
    return n.sort_values(["subset", "image_id", "order"]).reset_index(drop=True)


def main():
    n = build(pd.read_parquet(cfg.BBOX))
    cfg.atomic_write(OUT, lambda p: n.to_parquet(p, index=False))
    print(n.groupby("subset").agg(boxes=("text", "size"), unique=("text", "nunique"), pages=("image_id", "nunique")))
    print("DONE names_v1", len(n), "고유", n.text.nunique())


if __name__ == "__main__":
    main()
