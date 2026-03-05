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


@pytest.fixture()
def adata(scdata_path):
    """Load the test AnnData object, skipping if unavailable."""
    import scanpy as sc

    return sc.read_h5ad(scdata_path)


@pytest.fixture()
def marker_genes() -> dict[str, list[str]]:
    """Return the collapsed SVZ lineage marker gene sets."""
    return {
        "qNSCs": ["Aqp4", "Cxcl14", "Cpe", "S100a6"],
        "aNSCs": ["Btg2", "Egr1", "Egfr", "Ascl1", "Top2a"],
        "TAP": ["Mki67", "Ube2c"],
        "NB": ["Dcx", "Stmn2"],
    }
