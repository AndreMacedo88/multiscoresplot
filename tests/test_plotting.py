"""Tests for multiscoresplot._plotting (pipeline step 4)."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from multiscoresplot._colorspace import RGBResult, blend_to_rgb, reduce_to_rgb
from multiscoresplot._plotting import plot_embedding
from multiscoresplot._scoring import SCORE_PREFIX


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _random_coords(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2))


def _random_rgb(n: int = 100, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 3))


def _make_scores(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a score DataFrame with ``score-`` prefixed columns."""
    return pd.DataFrame({f"{SCORE_PREFIX}{k}": v for k, v in values.items()})


# ===========================================================================
# TestPlotEmbeddingUnit
# ===========================================================================


class TestPlotEmbeddingUnit:
    """Unit tests for plot_embedding."""

    def test_returns_axes_when_show_false(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, legend=False, show=False)
        assert result is not None

    def test_returns_none_when_show_true(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, legend=False, show=True)
        assert result is None

    def test_accepts_raw_ndarray(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        ax = plot_embedding(coords, rgb, legend=False, show=False)
        assert ax is not None

    def test_accepts_anndata_full_key(self) -> None:
        """basis= accepts the full obsm key name."""
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        ax = plot_embedding(adata, rgb, basis="X_umap", legend=False, show=False)
        assert ax is not None

    def test_accepts_anndata_custom_obsm_key(self) -> None:
        """basis= works with arbitrary obsm key names."""
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["umap_consensus"] = _random_coords(n)
        rgb = _random_rgb(n)
        ax = plot_embedding(adata, rgb, basis="umap_consensus", legend=False, show=False)
        assert ax is not None

    def test_old_basis_shorthand_deprecation(self) -> None:
        """Passing the old short basis name emits DeprecationWarning."""
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_embedding(adata, rgb, basis="umap", legend=False, show=False)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "full obsm key" in str(dep_warnings[0].message)
        assert ax is not None

    def test_mismatched_rgb_rows_raises(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(30)  # wrong count
        with pytest.raises(ValueError, match="rows"):
            plot_embedding(coords, rgb, legend=False, show=False)

    def test_invalid_rgb_columns_raises(self) -> None:
        coords = _random_coords(50)
        rgb = np.random.default_rng(0).random((50, 2))  # 2 cols instead of 3
        with pytest.raises(ValueError, match="rgb must be"):
            plot_embedding(coords, rgb, legend=False, show=False)

    def test_anndata_without_basis_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with pytest.raises(ValueError, match="basis must be provided"):
            plot_embedding(adata, rgb, legend=False, show=False)

    def test_missing_obsm_key_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        rgb = _random_rgb(n)
        with pytest.raises(KeyError, match=r"not found in adata\.obsm"):
            plot_embedding(adata, rgb, basis="X_umap", legend=False, show=False)

    def test_custom_point_size(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        ax = plot_embedding(coords, rgb, point_size=10.0, legend=False, show=False)
        assert ax is not None
        sizes = ax.collections[0].get_sizes()
        assert np.all(sizes == 10.0)

    def test_custom_ax_reused(self) -> None:
        _, my_ax = plt.subplots()
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, ax=my_ax, legend=False, show=False)
        assert result is my_ax

    def test_default_point_size(self) -> None:
        n = 200
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        ax = plot_embedding(coords, rgb, legend=False, show=False)
        sizes = ax.collections[0].get_sizes()
        expected = 120_000 / n
        assert np.allclose(sizes, expected)

    def test_custom_dpi(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(coords, rgb, legend=False, dpi=150, show=False)
        assert ax.figure.dpi == 150

    def test_legend_true_plain_ndarray_no_method_raises(self) -> None:
        """legend=True with plain ndarray and no method should raise."""
        coords = _random_coords()
        rgb = _random_rgb()
        with pytest.raises(ValueError, match="method"):
            plot_embedding(coords, rgb, legend=True, show=False)


# ===========================================================================
# TestPlotEmbeddingWithRGBResult
# ===========================================================================


class TestPlotEmbeddingWithRGBResult:
    """Tests for RGBResult metadata propagation to plot_embedding."""

    def _make_blend_result(self, n: int = 50) -> tuple[np.ndarray, RGBResult]:
        rng = np.random.default_rng(42)
        scores = _make_scores({"A": rng.random(n).tolist(), "B": rng.random(n).tolist()})
        return _random_coords(n), blend_to_rgb(scores)

    def _make_reduce_result(self, n: int = 50) -> tuple[np.ndarray, RGBResult]:
        rng = np.random.default_rng(42)
        scores = _make_scores({f"s{i}": rng.random(n).tolist() for i in range(4)})
        return _random_coords(n), reduce_to_rgb(scores, method="pca")

    def test_rgb_result_auto_method_direct(self) -> None:
        coords, result = self._make_blend_result()
        ax = plot_embedding(coords, result, show=False)
        # Legend should be created (inset axes)
        assert len(ax.child_axes) > 0

    def test_rgb_result_auto_method_pca(self) -> None:
        coords, result = self._make_reduce_result()
        ax = plot_embedding(coords, result, show=False)
        assert len(ax.child_axes) > 0

    def test_rgb_result_auto_gene_set_names(self) -> None:
        coords, result = self._make_blend_result()
        assert result.gene_set_names == ["A", "B"]
        ax = plot_embedding(coords, result, show=False)
        assert ax is not None

    def test_explicit_method_overrides_rgb_result(self) -> None:
        coords, result = self._make_reduce_result()
        # Override method from pca to nmf
        ax = plot_embedding(coords, result, method="nmf", show=False)
        assert ax is not None

    def test_explicit_gene_set_names_overrides_rgb_result(self) -> None:
        coords, result = self._make_blend_result()
        ax = plot_embedding(coords, result, gene_set_names=["Custom1", "Custom2"], show=False)
        assert ax is not None

    def test_legend_false_skips_even_with_rgb_result(self) -> None:
        coords, result = self._make_reduce_result()
        ax = plot_embedding(coords, result, legend=False, show=False)
        fig = ax.figure
        assert len(fig.axes) == 1


# ===========================================================================
# TestPlotEmbeddingWithLegend
# ===========================================================================


class TestPlotEmbeddingWithLegend:
    """Tests for legend integration in plot_embedding."""

    def test_legend_inset_creates_extra_axes(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="direct",
            gene_set_names=["A", "B"],
            legend=True,
            legend_style="inset",
            show=False,
        )
        # Inset axes are tracked as child axes of the parent
        assert len(ax.child_axes) > 0

    def test_legend_side_creates_extra_axes(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="direct",
            gene_set_names=["A", "B"],
            legend=True,
            legend_style="side",
            show=False,
        )
        fig = ax.figure
        assert len(fig.axes) > 1

    def test_legend_false_single_axes(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(coords, rgb, legend=False, show=False)
        fig = ax.figure
        assert len(fig.axes) == 1

    def test_nmf_method_creates_legend(self) -> None:
        """method='nmf' should create a reduction-style legend."""
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="nmf",
            legend=True,
            legend_style="inset",
            show=False,
        )
        assert len(ax.child_axes) > 0

    def test_ica_method_creates_legend(self) -> None:
        """method='ica' should create a reduction-style legend."""
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="ica",
            legend=True,
            legend_style="inset",
            show=False,
        )
        assert len(ax.child_axes) > 0

    def test_legend_size_param(self) -> None:
        """legend_size controls the legend inset size."""
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="pca",
            legend=True,
            legend_size=0.5,
            show=False,
        )
        assert len(ax.child_axes) > 0

    def test_legend_resolution_param(self) -> None:
        """legend_resolution is forwarded to render_legend."""
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(
            coords,
            rgb,
            method="pca",
            legend=True,
            legend_resolution=64,
            show=False,
        )
        assert len(ax.child_axes) > 0


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestPlotEmbeddingIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_full_pipeline_direct(self, adata, marker_genes) -> None:
        from multiscoresplot import blend_to_rgb, plot_embedding, score_gene_sets

        # Use only 3 gene sets — blend_to_rgb supports at most 3.
        three_sets = dict(list(marker_genes.items())[:3])
        scores = score_gene_sets(adata, three_sets, inplace=False)
        rgb = blend_to_rgb(scores)
        ax = plot_embedding(
            adata,
            rgb,
            basis="X_umap",
            show=False,
        )
        assert ax is not None
        assert len(ax.collections) > 0

    def test_full_pipeline_pca(self, adata, marker_genes) -> None:
        from multiscoresplot import plot_embedding, reduce_to_rgb, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="pca")
        ax = plot_embedding(
            adata,
            rgb,
            basis="X_umap",
            show=False,
        )
        assert ax is not None

    def test_full_pipeline_nmf(self, adata, marker_genes) -> None:
        from multiscoresplot import plot_embedding, reduce_to_rgb, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="nmf")
        ax = plot_embedding(
            adata,
            rgb,
            basis="X_umap",
            show=False,
        )
        assert ax is not None

    def test_non_umap_basis(self, adata, marker_genes) -> None:
        from multiscoresplot import blend_to_rgb, plot_embedding, score_gene_sets

        # Use only 3 gene sets — blend_to_rgb supports at most 3.
        three_sets = dict(list(marker_genes.items())[:3])
        scores = score_gene_sets(adata, three_sets, inplace=False)
        rgb = blend_to_rgb(scores)
        ax = plot_embedding(
            adata,
            rgb,
            basis="X_scanorama",
            show=False,
        )
        assert ax is not None
