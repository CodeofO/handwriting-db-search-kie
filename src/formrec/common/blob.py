"""crop PNG 바이트를 세트당 파일 하나(blob)로 이어 붙인 저장소 — 프레임워크 학습(PaddleOCR, PaddleFormers)이
이미지를 '경로'로 읽기 때문에 필요하다. 작은 파일 20만 개를 만들지 않으려는 것(서버 다운 전력).

파일: crops/{ds}/{subset}.blob (PNG 이어 붙임), {subset}.blobidx.npy (오프셋 N+1, int64). 행 순서 = cfg.queries(subset).
참조 문자열: "blob:{subset}/{row}" — 프레임워크의 이미지 읽기 함수를 패치해 이 문자열을 바이트로 바꾼다.
만들기: HW_DATASET=<ds> python -m formrec.common.blob   (IO 작업: run_job.sh --io)
"""
import os

import numpy as np

from formrec.common import workspace as cfg

PREFIX = "blob:"
_open = {}


def paths(subset):
    return cfg.CROPS / f"{subset}.blob", cfg.CROPS / f"{subset}.blobidx.npy"


def ref(subset, row):
    return f"{PREFIX}{subset}/{row}"


def is_ref(s):
    return isinstance(s, str) and s.startswith(PREFIX)


def read(s):
    """참조 문자열 → PNG 바이트. 프로세스(데이터로더 워커)마다 파일을 한 번 연다."""
    subset, row = s[len(PREFIX):].split("/")
    row = int(row)
    pid = os.getpid()
    if (pid, subset) not in _open:
        b, i = paths(subset)
        _open[(pid, subset)] = (open(b, "rb"), np.load(i))
    f, off = _open[(pid, subset)]
    f.seek(int(off[row]))
    return f.read(int(off[row + 1] - off[row]))


def build():
    import pandas as pd
    cfg.ensure_dirs()
    for s in cfg.SUBSETS:
        b, i = paths(s)
        if b.exists() and i.exists():
            print("skip", s); continue
        q = cfg.queries(s)[["image_id", "order"]]
        sh = pd.concat([pd.read_parquet(p, columns=["image_id", "order", "png"]) for p in sorted(cfg.CROPS.glob(f"{s}-*.parquet"))])
        sh = q.merge(sh, on=["image_id", "order"], how="left")
        assert sh.png.notna().all() and len(sh) == len(q)
        off = np.zeros(len(sh) + 1, np.int64)
        tmp = b.with_name(b.name + ".tmp")
        with open(tmp, "wb") as f:
            for k, x in enumerate(sh.png):
                f.write(x); off[k + 1] = off[k] + len(x)
                if (k + 1) % 20000 == 0:
                    f.flush(); os.fsync(f.fileno())
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, b)
        cfg.atomic_write(i, lambda t: np.save(open(t, "wb"), off))
        print(s, len(sh), f"{off[-1] / 2**20:.0f} MB", flush=True)
    print("DONE blob")


if __name__ == "__main__":
    build()
