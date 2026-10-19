"""공통 픽스처. 원본 대조 테스트(tests/legacy)는 LEGACY_WORK · FORMREC_WORKSPACE 가 있을 때만 돈다."""
import os

import pytest


@pytest.fixture(scope="session")
def legacy_work():
    p = os.environ.get("LEGACY_WORK")
    if not p or not os.path.isdir(p):
        pytest.skip("LEGACY_WORK(원본 코드 경로)가 없어 원본 대조 테스트를 건너뜀")
    return p


@pytest.fixture(scope="session")
def workspace_root():
    p = os.environ.get("FORMREC_WORKSPACE")
    if not p or not os.path.isdir(p):
        pytest.skip("FORMREC_WORKSPACE(산출물이 있는 작업 공간)가 없어 건너뜀")
    return p
