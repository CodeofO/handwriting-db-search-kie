"""작업 공간(데이터 · 캐시 · 결과 · 가중치) 경로와 데이터셋 설정 — 모든 모듈이 경로를 여기서만 가져온다.

작업 공간 루트: 환경변수 FORMREC_WORKSPACE (기본: 저장소 루트의 workspace/). 코드와 분리된 같은 상대 구조를 쓴다:
  data/   원천 데이터(AI Hub 053 · HANA · IAM, 직접 내려받음 — 저장소에 포함하지 않음)
  meta/   분할표 · DB · 질의(meta/x/{ds}_items.parquet 등)
  crops/  crop PNG 묶음(blob)      cache/  학습용 이미지 배열(npy)
  ft/     프레임워크 학습 데이터     runs/   학습 산출물        results/  평가 결과        logs/  실행 기록
데이터셋: 환경변수 HW_DATASET
  xa = AI Hub 053 한국어 서식(사람 블록, 칼럼 = 이미지 여러 장), xh = HANA 덴마크어 이름(한 이미지 안 여러 칼럼),
  xi = IAM 영어 문장 줄, xc = xa 칼럼을 한 줄로 이어 붙인 이미지, xk = 손글씨 KIE(서식 페이지)
  kie = xa 이전 단계의 AI Hub 필드 사다리 원본(crops/kie · cache/kie · meta/kie_db_*) — xa 분할표가 이 산출물을 참조한다.
세트: train / val / tune / test (xa·xh·xi·xc·xk). test = 공식 평가 폴더, 한 번만 본다.
"""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("FORMREC_WORKSPACE", REPO / "workspace")).resolve()
RAW = ROOT / "data/aihub053/01.데이터"
FOLDERS = {"Training": "1.Training", "Validation": "2.Validation"}
CELLS = {"PO": "P.Paper/O.Form", "PR": "P.Paper/R.Free", "TO": "T.Tablet/O.Form", "TR": "T.Tablet/R.Free"}
TUNE_FRAC = 0.10

XDS = ("xa", "xh", "xi", "xc", "xk")
DATASETS = ("kie", "hns", "hnf") + XDS
DATASET = os.environ.get("HW_DATASET", "xa")
assert DATASET in DATASETS, f"HW_DATASET={DATASET} (가능: {DATASETS})"
SUBSETS = ("train", "val", "tune", "test") if DATASET in XDS else ("train", "tune", "test")
LANG = "ko" if DATASET in ("kie", "xa", "xc", "xk") else "en"   # HANA(덴마크)도 라틴 문자

META = ROOT / "meta"
BBOX = META / "bbox.parquet"
CROPS = ROOT / f"crops/{DATASET}"              # {subset}.blob + {subset}.blobidx.npy
CACHE = ROOT / ("cache/hana" if DATASET in ("hns", "hnf") else f"cache/{DATASET}")
RESULTS = ROOT / f"results/{DATASET}"          # scores/ g1/ latency/ ...
RUNS = ROOT / f"runs/{DATASET}"
FT = ROOT / f"ft/{DATASET}"                    # 프레임워크 학습 데이터 (ppocr / pvl / ovis)
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
THIRD_PARTY = ROOT / "third_party"             # PaddleOCR · PaddleFormers 원본 사본(scripts/setup_third_party.sh)


def ensure_dirs():
    """쓰기 전에 부르는 디렉터리 생성(가져오기만 할 때는 만들지 않는다)."""
    for d in (META, CROPS, CACHE, RESULTS, RUNS, FT, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def label_dir(folder, cell):
    return RAW / FOLDERS[folder] / "라벨링데이터/라벨/HW-OCR/4.Validation" / CELLS[cell]


def image_path(folder, cell, image_id):
    return RAW / FOLDERS[folder] / "원천데이터/HW-OCR/4.Validation" / CELLS[cell] / f"{image_id}.png"


def subset_of(folder, image_id):
    """Training 폴더 → train/tune(페이지 단위 md5), Validation 폴더 → test."""
    if folder == "Validation":
        return "test"
    h = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)
    return "tune" if h / 0xFFFFFFFF < TUNE_FRAC else "train"


def cache_file(subset, size="96x288"):
    return CACHE / f"{subset}_{size}.npy"


def cache_index(subset):
    return CACHE / f"{subset}_index.parquet"


def queries(subset):
    """평가 질의 목록의 단일 정본. xa/xh/xi/xc/xk = 분할표 meta/x/{ds}_items.parquet의 그 세트 행(이미지 = blob:{src}/{row}),
    그 밖 = 캐시 인덱스(image_id, order 정렬). 모든 방법이 이 순서를 쓴다."""
    import pandas as pd
    if DATASET in XDS:
        q = pd.read_parquet(META / f"x/{DATASET}_items.parquet")
        q = q[q.subset == subset].reset_index(drop=True)
        return q.assign(image_id=q.item_id, order=0)
    q = pd.read_parquet(cache_index(subset))
    if DATASET in ("hns", "hnf"):
        q["text"] = q.name.map(hana_text)
    return q


def hana_text(name):
    """HANA 라벨(소문자 '이름 … 성') → 과제 문자열. hns: 성, hnf: '성 이름 중간이름'(이미지 순서)."""
    w = str(name).split()
    return w[-1] if DATASET == "hns" else " ".join([w[-1]] + w[:-1])


def atomic_write(path, write_fn):
    """임시 이름에 쓰고 fsync 후 rename (중간에 죽어도 반쯤 쓴 파일이 남지 않게)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
