"""일반 KIE 원본 대조: 원본 cache/xkg/ocr_*.parquet 표본 페이지에서 새 cells.pairs_cell vs 원본 xkg_cells.pairs_cell (paddle 환경 — cv2 필요).
프로세스 ≤ 8. 쓰기 없음."""
import glob
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

N_PAGES = 200


def test_pairs_cell_sample(legacy_work, workspace_root):
    pytest.importorskip("cv2")
    files = sorted(glob.glob(f"{workspace_root}/cache/xkg/ocr_*.parquet"))
    if not files:
        pytest.skip("cache/xkg 없음")
    sys.path.insert(0, legacy_work)
    import xkg_cells as old

    from formrec.kie_general import cells as new
    O = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sample(N_PAGES, random_state=0)
    args = list(zip(O.img, O.boxes))
    with ProcessPoolExecutor(8) as ex:
        a = list(ex.map(old.pairs_cell, args, chunksize=4))
        b = list(ex.map(new.pairs_cell, args, chunksize=4))
    assert a == b
    assert sum(len(x[1]) for x in a) > 0
