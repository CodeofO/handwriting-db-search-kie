"""데이터 준비 체인: 원천(AI Hub 053 · HANA · IAM) → 분할표 meta/x/{xa,xh,xi,xc}_items.parquet · crop 묶음(blob) · 캐시(npy) · 검색 DB · 어휘.

순서(docs/data.md):
  AI Hub 053 : aihub_bbox → names_v1 → kie_persons → kie_crops → kie_cache(--H 64 --W 1280) → kie_db(+ --levels F3 --N 2000000 --suffix _2M
               --no_levels_file) → kie_sub_db
  HANA       : hana
  분할 · 묶음 : build split → build links → build iam_blob → build hana_blob → build cache --ds xi / xh → build xc_blob → build cache --ds xc
  검색 · 어휘 : search_db → vocab
"""
