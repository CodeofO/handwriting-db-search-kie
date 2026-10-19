"""crop 묶음(blob) → 고정 크기 uint8 캐시 (제안 모델 학습 · 평가 입력).

전처리(load): 그레이스케일, 높이 H로 비율 유지 축소(폭이 W보다 크면 폭 기준 축소, BICUBIC), 왼쪽 정렬 · 세로 가운데, 나머지 흰색(255).
build.cache(xh · xi · xc)도 같은 load를 쓴다.
AI Hub: crops/kie/{subset}.blob → cache/kie/{subset}_{H}x{W}.npy (행 = cache/kie/{subset}_index.parquet = blob 행 순서).
  최종 실험 입력은 64×1280 (--H 64 --W 1280). memmap에 1만 행마다 flush, 끝나면 fsync 후 이름 바꿈.
사용: python -m formrec.data_prep.kie_cache --H 64 --W 1280   (IO 작업, 프로세스 8개)
"""
import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from formrec.common import workspace as cfg

SUBSETS = ("train", "tune", "test")


def load(args):
    b, H, W = args
    im = Image.open(io.BytesIO(b)).convert("L")
    w, h = im.size
    s = min(H / h, W / w)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    out = np.full((H, W), 255, np.uint8)
    y0 = (H - im.height) // 2
    out[y0:y0 + im.height, :im.width] = np.asarray(im)
    return out


def write_cache(bpath, ipath, out, H, W, workers=8, chunksize=64, flush_every=10000):
    """blob 전체 → (n, H, W) uint8 npy (임시 이름 → fsync → 이름 바꿈)."""
    off = np.load(ipath); n = len(off) - 1
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.name + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, H, W))
    f = open(bpath, "rb")

    def gen():
        for k in range(n):
            f.seek(int(off[k])); yield f.read(int(off[k + 1] - off[k])), H, W
    with ProcessPoolExecutor(workers) as ex:
        for k, arr in enumerate(ex.map(load, gen(), chunksize=chunksize)):
            mm[k] = arr
            if (k + 1) % flush_every == 0:
                mm.flush(); print(f"{out.name}: {k + 1}/{n}", flush=True)
    mm.flush(); del mm
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    for s in SUBSETS:
        out = cfg.ROOT / f"cache/kie/{s}_{a.H}x{a.W}.npy"
        n = write_cache(cfg.ROOT / f"crops/kie/{s}.blob", cfg.ROOT / f"crops/kie/{s}.blobidx.npy", out, a.H, a.W, a.workers)
        print(f"{s}: {n} → {out.name}", flush=True)
    print("DONE kie_cache")


if __name__ == "__main__":
    main()
