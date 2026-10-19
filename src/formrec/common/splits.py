"""묶음 단위 분할 해시 — 페이지 · 필기자 · 이미지 같은 묶음 키를 시드 고정 md5로 train / val / tune에 나눈다(데이터 준비 · KIE 공용)."""
import hashlib

SALT = "x1:"


def h01(key):
    return int(hashlib.md5((SALT + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def case2(key):
    """원본에 train/test만 있는 데이터(AI Hub 공식 Training/Validation · HANA): train을 0.8 / 0.1 / 0.1로 train / val / tune."""
    h = h01(key)
    return "val" if h < 0.1 else "tune" if h < 0.2 else "train"
