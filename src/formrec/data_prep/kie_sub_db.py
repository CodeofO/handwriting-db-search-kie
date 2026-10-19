"""서브 실험 DB 규모: F3 DB 2만(meta/kie_db_F3_20k.parquet) — 기본 20만 DB에서 부분 추출. 200만은 kie_db --levels F3 --N 2000000 --suffix _2M.

실제 F3 값이 약 2.5만이라 20만 DB를 그대로 줄일 수 없다 → 이전 단계 tune · test 정답은 모두 넣고, 나머지를 기본 DB에서
실제/합성 비율을 유지해 무작위 추출(시드 0). 최종 분할(xa)의 질의 정답 보충은 평가 쪽에서 한다.
사용: python -m formrec.data_prep.kie_sub_db
"""
import pandas as pd

from formrec.common import workspace as cfg

N = 20_000


def build(D, L, n=N):
    need = set(L[(L.level == "F3") & L.subset.isin(["tune", "test"])].text)
    keep = D.text.isin(need)
    rest = D[~keep]
    k = n - int(keep.sum())
    fr = rest.real.mean()
    pick = pd.concat([rest[rest.real].sample(int(k * fr), random_state=0), rest[~rest.real].sample(k - int(k * fr), random_state=0)])
    return pd.concat([D[keep], pick]).sort_values("text").reset_index(drop=True), int(keep.sum())


def main():
    out, nk = build(pd.read_parquet(cfg.META / "kie_db_F3.parquet"), pd.read_parquet(cfg.META / "kie_levels.parquet"))
    cfg.atomic_write(cfg.META / "kie_db_F3_20k.parquet", lambda t: out.to_parquet(t, index=False))
    print(f"F3@20k: {len(out):,} (tune·test 정답 {nk:,}, 실제 {int(out.real.sum()):,}, 합성 {int((~out.real).sum()):,})")
    print("DONE kie_sub_db")


if __name__ == "__main__":
    main()
