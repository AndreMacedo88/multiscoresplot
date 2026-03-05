"""Tests for multiscoresplot._scoring."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from multiscoresplot import score_gene_sets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_adata(n_cells: int = 50, n_genes: int = 20, seed: int = 42) -> anndata.AnnData:
    """Create a small synthetic AnnData with random count data."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=[f"Cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    return anndata.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Unit tests (synthetic data)
# ---------------------------------------------------------------------------


class TestScoreGeneSetsUnit:
    """Unit tests using small synthetic AnnData objects."""

    def test_returns_dataframe(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = score_gene_sets(adata, gene_sets)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["score-A", "score-B"]
        assert len(result) == adata.n_obs

    def test_score_columns_in_adata_obs_inplace(self):
        adata = _make_synthetic_adata()
        gene_sets = {"X": ["Gene0", "Gene1"]}
        score_gene_sets(adata, gene_sets, inplace=True)

        assert "score-X" in adata.obs.columns

    def test_score_not_inplace(self):
        adata = _make_synthetic_adata()
        gene_sets = {"X": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, inplace=False)

        assert "score-X" not in adata.obs.columns
        assert "score-X" in result.columns

    def test_scores_bounded_0_1(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        result = score_gene_sets(adata, gene_sets)

        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_empty_gene_sets_raises(self):
        adata = _make_synthetic_adata()
        with pytest.raises(ValueError, match="non-empty dict"):
            score_gene_sets(adata, {})

    def test_empty_gene_list_raises(self):
        adata = _make_synthetic_adata()
        with pytest.raises(ValueError, match="non-empty list"):
            score_gene_sets(adata, {"A": []})

    def test_missing_genes_handled(self):
        """Gene set with genes not in adata should not crash (imputed as missing)."""
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "FAKE_GENE_1", "FAKE_GENE_2"]}
        result = score_gene_sets(adata, gene_sets)

        assert isinstance(result, pd.DataFrame)
        assert "score-A" in result.columns

    def test_no_ucell_columns_left(self):
        """After scoring, no intermediate _UCell columns should remain in adata.obs."""
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        score_gene_sets(adata, gene_sets, inplace=True)

        ucell_cols = [c for c in adata.obs.columns if c.endswith("_UCell")]
        assert ucell_cols == []


# ---------------------------------------------------------------------------
# Integration tests (real data — skipped if unavailable)
# ---------------------------------------------------------------------------


class TestScoreGeneSetsIntegration:
    """Integration tests using real scRNA-seq data."""

    def test_score_real_data(self, adata, marker_genes):
        result = score_gene_sets(adata, marker_genes)

        assert result.shape == (15615, 4)
        assert list(result.columns) == [
            "score-qNSCs",
            "score-aNSCs",
            "score-TAP",
            "score-NB",
        ]
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_known_cell_type_scores_higher(self, adata, marker_genes):
        """NB cells should have higher median score-NB than qNSC cells."""
        score_gene_sets(adata, marker_genes, inplace=True)

        nb_mask = adata.obs["celltype"] == "NB"
        qnsc_mask = adata.obs["celltype"].isin(["qNSC1", "qNSC2"])

        median_nb_in_nb = adata.obs.loc[nb_mask, "score-NB"].median()
        median_nb_in_qnsc = adata.obs.loc[qnsc_mask, "score-NB"].median()

        assert median_nb_in_nb > median_nb_in_qnsc
