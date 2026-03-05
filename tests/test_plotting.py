"""Tests for multiscoresplot._plotting (pipeline step 4)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from multiscoresplot._plotting import plot_embedding


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


# ===========================================================================
# TestPlotEmbeddingUnit
# ===========================================================================


class TestPlotEmbeddingUnit:
    """Unit tests for plot_embedding."""

    def test_returns_axes_when_show_false(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, show=False)
        assert result is not None

    def test_returns_none_when_show_true(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, show=True)
        assert result is None

    def test_accepts_raw_ndarray(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        ax = plot_embedding(coords, rgb, show=False)
        assert ax is not None

    def test_accepts_anndata(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        ax = plot_embedding(adata, rgb, basis="umap", show=False)
        assert ax is not None

    def test_mismatched_rgb_rows_raises(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(30)  # wrong count
        with pytest.raises(ValueError, match="rows"):
            plot_embedding(coords, rgb, show=False)

    def test_invalid_rgb_columns_raises(self) -> None:
        coords = _random_coords(50)
        rgb = np.random.default_rng(0).random((50, 2))  # 2 cols instead of 3
        with pytest.raises(ValueError, match="rgb must be"):
            plot_embedding(coords, rgb, show=False)

    def test_anndata_without_basis_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with pytest.raises(ValueError, match="basis must be provided"):
            plot_embedding(adata, rgb, show=False)

    def test_missing_obsm_key_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        rgb = _random_rgb(n)
        with pytest.raises(KeyError, match="X_umap"):
            plot_embedding(adata, rgb, basis="umap", show=False)

    def test_custom_point_size(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        ax = plot_embedding(coords, rgb, point_size=10.0, show=False)
        assert ax is not None
        sizes = ax.collections[0].get_sizes()
        assert np.all(sizes == 10.0)

    def test_custom_ax_reused(self) -> None:
        _, my_ax = plt.subplots()
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding(coords, rgb, ax=my_ax, show=False)
        assert result is my_ax

    def test_default_point_size(self) -> None:
        n = 200
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        ax = plot_embedding(coords, rgb, show=False)
        sizes = ax.collections[0].get_sizes()
        expected = 120_000 / n
        assert np.allclose(sizes, expected)


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

    def test_legend_true_method_none_no_error(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        ax = plot_embedding(coords, rgb, legend=True, method=None, show=False)
        fig = ax.figure
        # No legend added since method is None
        assert len(fig.axes) == 1


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestPlotEmbeddingIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_full_pipeline_direct(self, adata, marker_genes) -> None:
        from multiscoresplot import plot_embedding, project_direct, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = project_direct(scores)
        ax = plot_embedding(
            adata,
            rgb,
            basis="umap",
            method="direct",
            gene_set_names=list(marker_genes.keys()),
            show=False,
        )
        assert ax is not None
        assert len(ax.collections) > 0

    def test_full_pipeline_pca(self, adata, marker_genes) -> None:
        from multiscoresplot import plot_embedding, project_pca, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = project_pca(scores)
        ax = plot_embedding(
            adata,
            rgb,
            basis="umap",
            method="pca",
            show=False,
        )
        assert ax is not None

    def test_non_umap_basis(self, adata, marker_genes) -> None:
        from multiscoresplot import plot_embedding, project_direct, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = project_direct(scores)
        ax = plot_embedding(
            adata,
            rgb,
            basis="scanorama",
            method="direct",
            gene_set_names=list(marker_genes.keys()),
            show=False,
        )
        assert ax is not None
