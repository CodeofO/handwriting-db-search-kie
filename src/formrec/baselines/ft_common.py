"""비교군 FT 데이터 공통: 분할표 행 읽기 · jsonl 쓰기 · 기준 설정(configs/) 경로 · 명령줄 인자(원본 x_ft_data와 같은 이름 · 기본값).

이미지 참조 = "blob:{src}/{row}" (HW_DATASET = ds → crops/{ds}/{src}.blob). 정답 = 그 이미지에 쓰인 문자열(item.text).
세트: train = 학습, val = 에폭마다 val loss(에폭 선정), tune · test = 평가 스크립트가 읽는다.
"""
import argparse
import json

import pandas as pd

from formrec.common import workspace as cfg

CONFIGS = cfg.REPO / "configs"
PPOCR_BASE = CONFIGS / "ppocr/korean_v5_ft_base.yml"          # 원본 작업 공간 ft/kie/ppocr/korean_v5_ft.yml
PVL_BASE = CONFIGS / "pvl/pvl16_full_base.yaml"               # 원본 작업 공간 ft/kie/pvl/pvl16_full.yaml
PVL_REGISTER = CONFIGS / "pvl/paddleocr_vl_v15_template.py"   # PaddleFormers 사용자 템플릿 등록 파일
GPUS = 4


def items(sub, ds=None):
    D = pd.read_parquet(cfg.META / f"x/{ds or cfg.DATASET}_items.parquet")
    D = D[D.subset == sub].reset_index(drop=True)
    D["ref"] = [f"blob:{s}/{r}" for s, r in zip(D.src, D.row)]
    return D


def write_jsonl(path, rows):
    def w(t):
        with open(t, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfg.atomic_write(path, w)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ppocr", "vlm"])
    ap.add_argument("--epochs", type=int, default=20, help="PP-OCRv5 에폭")
    ap.add_argument("--max_len", type=int, default=0, help="PP-OCRv5 최대 글자 수 (0 = 기준 설정 40)")
    ap.add_argument("--width", type=int, default=960, help="PP-OCRv5 입력 폭(기준 960)")
    ap.add_argument("--save_every", type=int, default=1, help="PP-OCRv5 체크포인트 간격(에폭)")
    ap.add_argument("--bs", type=int, default=0, help="PP-OCRv5 GPU당 배치(0 = 기준 48). 학습률은 배치에 비례(공식 규칙)")
    ap.add_argument("--save_splits", type=int, default=0, help="VLM: 0 = 에폭마다 저장, N = 전체를 N등분한 스텝마다 저장 · val loss (큰 데이터 1회 순회)")
    ap.add_argument("--vlm_samples", type=int, default=308247, help="VLM 학습 샘플 예산 (한국어 3에폭 = 102,749 × 3)")
    ap.add_argument("--vlm_epochs", type=int, default=0, help="0 = 예산으로 자동")
    ap.add_argument("--val_max", type=int, default=20000, help="val loss용 표본 상한(큰 데이터)")
    return ap


def main(argv=None):
    """원본 x_ft_data.py와 같은 명령줄: {ppocr | vlm} [인자]."""
    a = parser().parse_args(argv)
    if a.cmd == "ppocr":
        from formrec.baselines.ppocr.ft_data import build
    else:
        from formrec.baselines.vlm.ft_data import build
    build(a)
