"""Tests for multiscoresplot._interactive (interactive Plotly plotting)."""

from __future__ import annotations

import base64
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")

from multiscoresplot._interactive import (  # noqa: E402
    _render_legend_to_base64,
    plot_embedding_interactive,
)


def _random_coords(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2))


def _random_rgb(n: int = 100, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 3))


def _make_scores(n: int = 100, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "score-A": rng.random(n),
            "score-B": rng.random(n),
            "score-C": rng.random(n),
        }
    )


# ===========================================================================
# Unit tests
# ===========================================================================


class TestPlotEmbeddingInteractiveUnit:
    """Unit tests for plot_embedding_interactive."""

    def test_returns_none_when_show_true(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_embedding_interactive(coords, rgb, show=True)
        assert result is None

    def test_returns_figure_when_show_false(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding_interactive(coords, rgb, show=False)
        assert result is not None
        assert hasattr(result, "data")

    def test_accepts_raw_ndarray(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        fig = plot_embedding_interactive(coords, rgb, show=False)
        assert fig is not None

    def test_accepts_anndata(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(adata, rgb, basis="umap", show=False)
        assert fig is not None

    def test_mismatched_rgb_rows_raises(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(30)
        with pytest.raises(ValueError, match="rows"):
            plot_embedding_interactive(coords, rgb, show=False)

    def test_custom_point_size_alpha_title(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        fig = plot_embedding_interactive(
            coords, rgb, point_size=5, alpha=0.5, title="Test", show=False
        )
        assert fig.layout.title.text == "Test"
        assert fig.data[0].marker.size == 5

    def test_anndata_without_basis_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with pytest.raises(ValueError, match="basis must be provided"):
            plot_embedding_interactive(adata, rgb, show=False)

    def test_custom_width_height(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, width=800, height=600, show=False)
        assert fig.layout.width == 800
        assert fig.layout.height == 600


# ===========================================================================
# Hover info tests
# ===========================================================================


class TestHoverInfo:
    """Tests for hover text assembly."""

    def test_explicit_scores_prefix_stripped(self) -> None:
        n = 10
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        scores = _make_scores(n)
        fig = plot_embedding_interactive(coords, rgb, scores=scores, show=False)
        hover = fig.data[0].hovertext[0]
        assert "A:" in hover
        assert "B:" in hover
        assert "score-" not in hover

    def test_auto_extraction_from_adata_obs(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        rng = np.random.default_rng(42)
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        adata.obs["score-X"] = rng.random(n)
        adata.obs["score-Y"] = rng.random(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(adata, rgb, basis="umap", show=False)
        hover = fig.data[0].hovertext[0]
        assert "X:" in hover
        assert "Y:" in hover

    def test_rgb_labeled_by_component_names_with_method(self) -> None:
        n = 10
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(coords, rgb, method="pca", show=False)
        hover = fig.data[0].hovertext[0]
        assert "PC1:" in hover
        assert "PC2:" in hover
        assert "PC3:" in hover

    def test_rgb_labeled_rgb_when_method_none(self) -> None:
        n = 10
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(coords, rgb, method=None, show=False)
        hover = fig.data[0].hovertext[0]
        assert "R:" in hover
        assert "G:" in hover
        assert "B:" in hover

    def test_hover_columns_numeric_and_categorical(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        adata.obs["n_counts"] = np.arange(n, dtype=float)
        adata.obs["cell_type"] = pd.Categorical(["TypeA"] * 10 + ["TypeB"] * 10)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(
            adata, rgb, basis="umap", hover_columns=["n_counts", "cell_type"], show=False
        )
        hover = fig.data[0].hovertext[0]
        assert "n_counts:" in hover
        assert "cell_type:" in hover

    def test_hover_columns_missing_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 10
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with pytest.raises(KeyError, match="not_a_column"):
            plot_embedding_interactive(
                adata, rgb, basis="umap", hover_columns=["not_a_column"], show=False
            )

    def test_hover_columns_with_raw_coords_raises(self) -> None:
        coords = _random_coords(10)
        rgb = _random_rgb(10)
        with pytest.raises(ValueError, match="hover_columns requires an AnnData"):
            plot_embedding_interactive(coords, rgb, hover_columns=["something"], show=False)

    def test_gene_set_names_override_labels(self) -> None:
        n = 10
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        scores = _make_scores(n)
        fig = plot_embedding_interactive(
            coords, rgb, scores=scores, gene_set_names=["Alpha", "Beta", "Gamma"], show=False
        )
        hover = fig.data[0].hovertext[0]
        assert "Alpha:" in hover
        assert "Beta:" in hover
        assert "Gamma:" in hover


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestPlotEmbeddingInteractiveIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_full_pipeline(self, adata, marker_genes) -> None:
        from multiscoresplot import reduce_to_rgb, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="nmf")
        fig = plot_embedding_interactive(
            adata,
            rgb,
            basis="umap",
            scores=scores,
            method="nmf",
            gene_set_names=list(marker_genes.keys()),
            show=False,
        )
        assert fig is not None
        assert len(fig.data[0].hovertext) == adata.n_obs

    def test_with_hover_columns(self, adata, marker_genes) -> None:
        from multiscoresplot import reduce_to_rgb, score_gene_sets

        scores = score_gene_sets(adata, marker_genes, inplace=True)
        rgb = reduce_to_rgb(scores, method="pca")
        # Pick an existing obs column for hover
        obs_col = adata.obs.columns[0]
        fig = plot_embedding_interactive(
            adata,
            rgb,
            basis="umap",
            method="pca",
            hover_columns=[obs_col],
            show=False,
        )
        assert fig is not None
        assert obs_col in fig.data[0].hovertext[0]


# ===========================================================================
# Legend tests
# ===========================================================================


class TestInteractiveLegend:
    """Tests for the colormap legend overlay in interactive plots."""

    def test_legend_adds_layout_image(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method="pca", show=False)
        assert len(fig.layout.images) == 1

    def test_legend_false_no_image(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method="pca", legend=False, show=False)
        assert not fig.layout.images

    def test_legend_method_none_no_image(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method=None, show=False)
        assert not fig.layout.images

    def test_legend_direct_2set(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(
            coords, rgb, method="direct", gene_set_names=["A", "B"], show=False
        )
        assert len(fig.layout.images) == 1

    def test_legend_direct_3set(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(
            coords, rgb, method="direct", gene_set_names=["A", "B", "C"], show=False
        )
        assert len(fig.layout.images) == 1

    @pytest.mark.parametrize("loc", ["lower right", "lower left", "upper right", "upper left"])
    def test_legend_loc_positions(self, loc: str) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method="pca", legend_loc=loc, show=False)
        img = fig.layout.images[0]
        # Just check the image was placed (exact values tested implicitly by the dict)
        assert img.xref == "paper"
        assert img.yref == "paper"

    def test_legend_custom_size(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method="pca", legend_size=0.5, show=False)
        assert fig.layout.images[0].sizex == 0.5
        assert fig.layout.images[0].sizey == 0.5

    def test_legend_direct_no_names_skips(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(
            coords, rgb, method="direct", gene_set_names=None, show=False
        )
        assert not fig.layout.images

    def test_render_legend_to_base64_valid_png(self) -> None:
        uri = _render_legend_to_base64("pca")
        assert uri.startswith("data:image/png;base64,")
        # Decode and check PNG magic bytes
        b64_data = uri.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        assert raw[:4] == b"\x89PNG"
