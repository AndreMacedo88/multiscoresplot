"""Shared fixtures for multiscoresplot tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture()
def scdata_path() -> Path:
    """Return path to the test .h5ad file, skipping if the symlink target is missing."""
    path = DATA_DIR / "scdata.h5ad"
    if not path.exists():
        pytest.skip("test data not available (symlink target missing)")
    return path
