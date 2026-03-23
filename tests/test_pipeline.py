"""Tests for multiscoresplot._pipeline (plot_scores convenience function)."""

from __future__ import annotations

import anndata
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from multiscoresplot import plot_scores
from multiscoresplot._colorspace import RGBResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_adata(n_cells: int = 50, n_genes: int = 20, seed: int = 42) -> anndata.AnnData:
    """Create a small synthetic AnnData with random count data + UMAP."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=[f"Cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.standard_normal((n_cells, 2)).astype(np.float32)
    return adata


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPlotScores:
    """Tests for the plot_scores convenience function."""

    def test_returns_tuple_of_three(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = plot_scores(adata, gene_sets, show=False)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_scores_dataframe_correct_columns(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        scores, _rgb, _ax = plot_scores(adata, gene_sets, show=False)
        assert isinstance(scores, pd.DataFrame)
        assert list(scores.columns) == ["score-A", "score-B"]

    def test_rgb_result_type(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        _scores, rgb, _ax = plot_scores(adata, gene_sets, show=False)
        assert isinstance(rgb, RGBResult)

    def test_auto_method_blend_for_2_sets(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        _scores, rgb, _ax = plot_scores(adata, gene_sets, show=False)
        assert rgb.method == "direct"

    def test_auto_method_blend_for_3_sets(self):
        adata = _make_synthetic_adata()
        gene_sets = {
            "A": ["Gene0", "Gene1"],
            "B": ["Gene2", "Gene3"],
            "C": ["Gene4", "Gene5"],
        }
        _scores, rgb, _ax = plot_scores(adata, gene_sets, show=False)
        assert rgb.method == "direct"

    def test_auto_method_pca_for_4_sets(self):
        adata = _make_synthetic_adata()
        gene_sets = {
            "A": ["Gene0", "Gene1"],
            "B": ["Gene2", "Gene3"],
            "C": ["Gene4", "Gene5"],
            "D": ["Gene6", "Gene7"],
        }
        _scores, rgb, _ax = plot_scores(adata, gene_sets, show=False)
        assert rgb.method == "pca"

    def test_explicit_method_pca_for_2_sets(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        _scores, rgb, _ax = plot_scores(adata, gene_sets, method="pca", show=False)
        assert rgb.method == "pca"

    def test_explicit_method_blend_string(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        _scores, rgb, _ax = plot_scores(adata, gene_sets, method="blend", show=False)
        assert rgb.method == "direct"

    def test_show_false_returns_axes(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        _scores, _rgb, ax = plot_scores(adata, gene_sets, show=False)
        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)

    def test_callable_method_accepted(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}

        def my_reducer(X, n_components, **kwargs):
            return np.full((X.shape[0], 3), 0.5)

        _scores, rgb, _ax = plot_scores(adata, gene_sets, method=my_reducer, show=False)
        assert rgb.method == "my_reducer"

    def test_inplace_stores_in_adata(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        plot_scores(adata, gene_sets, inplace=True, show=False)
        assert "score-A" in adata.obs.columns
        assert "score-B" in adata.obs.columns

    def test_clip_and_normalize_forwarded(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        scores, _rgb, _ax = plot_scores(adata, gene_sets, clip_pct=99, normalize=True, show=False)
        for col in scores.columns:
            assert scores[col].min() == pytest.approx(0.0)
            assert scores[col].max() == pytest.approx(1.0)

    def test_custom_prefix_forwarded(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        scores, rgb, _ax = plot_scores(adata, gene_sets, prefix="msp-", show=False)
        assert "msp-A" in scores.columns
        assert "msp-B" in scores.columns
        assert rgb.prefix == "msp-"
        assert rgb.gene_set_names == ["A", "B"]

    def test_custom_suffix_forwarded(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        scores, rgb, _ax = plot_scores(adata, gene_sets, suffix="_v2", show=False)
        assert "score-A_v2" in scores.columns
        assert "score-B_v2" in scores.columns
        assert rgb.suffix == "_v2"
        assert rgb.gene_set_names == ["A", "B"]


# ---------------------------------------------------------------------------
# Integration tests (real data — skipped if unavailable)
# ---------------------------------------------------------------------------


class TestPlotScoresIntegration:
    """End-to-end test with real scRNA-seq data."""

    def test_full_pipeline_real_data(self, adata, marker_genes):
        scores, rgb, _ax = plot_scores(
            adata,
            marker_genes,
            basis="X_umap",
            show=False,
        )
        assert scores.shape == (adata.n_obs, 4)
        assert isinstance(rgb, RGBResult)
        # 4 gene sets -> auto pca
        assert rgb.method == "pca"
