"""Tests for multiscoresplot._interactive (interactive Plotly plotting)."""

from __future__ import annotations

import base64
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")

from multiscoresplot._colorspace import blend_to_rgb, reduce_to_rgb  # noqa: E402
from multiscoresplot._interactive import (  # noqa: E402
    _render_legend_to_base64,
    plot_embedding_interactive,
)
from multiscoresplot._scoring import SCORE_PREFIX  # noqa: E402


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


def _make_scores_dict(n: int = 100, seed: int = 2) -> pd.DataFrame:
    """Build via _make_scores helper from _colorspace tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"{SCORE_PREFIX}{k}": rng.random(n).tolist() for k in ["A", "B"]})


# ===========================================================================
# Unit tests
# ===========================================================================


class TestPlotEmbeddingInteractiveUnit:
    """Unit tests for plot_embedding_interactive."""

    def test_returns_none_when_show_true(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_embedding_interactive(coords, rgb, legend=False, show=True)
        assert result is None

    def test_returns_figure_when_show_false(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        result = plot_embedding_interactive(coords, rgb, legend=False, show=False)
        assert result is not None
        assert hasattr(result, "data")

    def test_accepts_raw_ndarray(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        fig = plot_embedding_interactive(coords, rgb, legend=False, show=False)
        assert fig is not None

    def test_accepts_anndata_full_key(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(adata, rgb, basis="X_umap", legend=False, show=False)
        assert fig is not None

    def test_accepts_anndata_custom_obsm_key(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["umap_consensus"] = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(
            adata, rgb, basis="umap_consensus", legend=False, show=False
        )
        assert fig is not None

    def test_old_basis_shorthand_deprecation(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 30
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_embedding_interactive(adata, rgb, basis="umap", legend=False, show=False)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "full obsm key" in str(dep_warnings[0].message)
        assert fig is not None

    def test_mismatched_rgb_rows_raises(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(30)
        with pytest.raises(ValueError, match="rows"):
            plot_embedding_interactive(coords, rgb, legend=False, show=False)

    def test_custom_point_size_alpha_title(self) -> None:
        coords = _random_coords(50)
        rgb = _random_rgb(50)
        fig = plot_embedding_interactive(
            coords,
            rgb,
            point_size=5,
            alpha=0.5,
            title="Test",
            legend=False,
            show=False,
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
            plot_embedding_interactive(adata, rgb, legend=False, show=False)

    def test_figsize_and_dpi(self) -> None:
        """figsize * dpi gives pixel dimensions."""
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(
            coords, rgb, figsize=(8.0, 6.0), dpi=100, legend=False, show=False
        )
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_figsize_dpi_custom(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(
            coords, rgb, figsize=(5.0, 5.0), dpi=150, legend=False, show=False
        )
        assert fig.layout.width == 750
        assert fig.layout.height == 750

    def test_legend_true_plain_ndarray_no_method_raises(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        with pytest.raises(ValueError, match="method"):
            plot_embedding_interactive(coords, rgb, legend=True, show=False)


# ===========================================================================
# RGBResult tests
# ===========================================================================


class TestInteractiveRGBResult:
    """Tests for RGBResult metadata propagation to interactive plot."""

    def test_rgb_result_auto_method(self) -> None:
        n = 50
        coords = _random_coords(n)
        scores = _make_scores_dict(n)
        result = reduce_to_rgb(scores, method="pca")
        fig = plot_embedding_interactive(coords, result, show=False)
        assert fig is not None
        # Legend should be present
        assert len(fig.layout.images) == 1

    def test_rgb_result_auto_gene_set_names_in_hover(self) -> None:
        n = 10
        coords = _random_coords(n)
        scores = _make_scores_dict(n)
        result = reduce_to_rgb(scores, method="pca")
        fig = plot_embedding_interactive(coords, result, scores=scores, show=False)
        hover = fig.data[0].hovertext[0]
        # gene_set_names from RGBResult should be used
        assert "A:" in hover
        assert "B:" in hover

    def test_rgb_result_blend_auto_method_direct(self) -> None:
        n = 50
        coords = _random_coords(n)
        scores = _make_scores_dict(n)
        result = blend_to_rgb(scores)
        fig = plot_embedding_interactive(coords, result, show=False)
        # Direct legend with gene_set_names
        assert len(fig.layout.images) == 1

    def test_explicit_method_overrides_rgb_result(self) -> None:
        n = 50
        coords = _random_coords(n)
        scores = _make_scores_dict(n)
        result = reduce_to_rgb(scores, method="pca")
        fig = plot_embedding_interactive(coords, result, method="nmf", show=False)
        assert fig is not None
        hover = fig.data[0].hovertext[0]
        assert "NMF1:" in hover  # overridden method used for channel labels


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
        fig = plot_embedding_interactive(coords, rgb, scores=scores, legend=False, show=False)
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
        fig = plot_embedding_interactive(adata, rgb, basis="X_umap", legend=False, show=False)
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
        fig = plot_embedding_interactive(coords, rgb, method=None, legend=False, show=False)
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
            adata,
            rgb,
            basis="X_umap",
            legend=False,
            hover_columns=["n_counts", "cell_type"],
            show=False,
        )
        hover = fig.data[0].hovertext[0]
        assert "n_counts:" in hover
        assert "cell_type:" in hover

    def test_hover_columns_missing_obs_and_var_raises(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 10
        adata = anndata.AnnData(np.zeros((n, 5)))
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        with pytest.raises(KeyError, match="not found"):
            plot_embedding_interactive(
                adata,
                rgb,
                basis="X_umap",
                legend=False,
                hover_columns=["not_a_column"],
                show=False,
            )

    def test_hover_columns_with_raw_coords_raises(self) -> None:
        coords = _random_coords(10)
        rgb = _random_rgb(10)
        with pytest.raises(ValueError, match="hover_columns requires an AnnData"):
            plot_embedding_interactive(
                coords, rgb, legend=False, hover_columns=["something"], show=False
            )

    def test_gene_set_names_override_labels(self) -> None:
        n = 10
        coords = _random_coords(n)
        rgb = _random_rgb(n)
        scores = _make_scores(n)
        fig = plot_embedding_interactive(
            coords,
            rgb,
            scores=scores,
            legend=False,
            gene_set_names=["Alpha", "Beta", "Gamma"],
            show=False,
        )
        hover = fig.data[0].hovertext[0]
        assert "Alpha:" in hover
        assert "Beta:" in hover
        assert "Gamma:" in hover


# ===========================================================================
# Hover gene expression from adata.var (TODO 9)
# ===========================================================================


class TestHoverGeneExpression:
    """Tests for hover_columns falling back to gene expression."""

    def test_hover_gene_from_var_dense(self) -> None:
        anndata = pytest.importorskip("anndata")
        n = 20
        rng = np.random.default_rng(99)
        X = rng.random((n, 5))
        adata = anndata.AnnData(X)
        adata.var_names = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(
            adata,
            rgb,
            basis="X_umap",
            legend=False,
            hover_columns=["GeneA"],
            show=False,
        )
        hover = fig.data[0].hovertext[0]
        assert "GeneA:" in hover

    def test_hover_gene_from_var_sparse(self) -> None:
        anndata = pytest.importorskip("anndata")
        import scipy.sparse as sp

        n = 20
        rng = np.random.default_rng(99)
        X = sp.csr_matrix(rng.random((n, 5)))
        adata = anndata.AnnData(X)
        adata.var_names = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
        adata.obsm["X_umap"] = _random_coords(n)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(
            adata,
            rgb,
            basis="X_umap",
            legend=False,
            hover_columns=["GeneB"],
            show=False,
        )
        hover = fig.data[0].hovertext[0]
        assert "GeneB:" in hover

    def test_hover_mixed_obs_and_gene(self) -> None:
        """Mix of obs column and gene name in hover_columns."""
        anndata = pytest.importorskip("anndata")
        n = 20
        rng = np.random.default_rng(99)
        X = rng.random((n, 3))
        adata = anndata.AnnData(X)
        adata.var_names = ["GeneA", "GeneB", "GeneC"]
        adata.obsm["X_umap"] = _random_coords(n)
        adata.obs["n_counts"] = np.arange(n, dtype=float)
        rgb = _random_rgb(n)
        fig = plot_embedding_interactive(
            adata,
            rgb,
            basis="X_umap",
            legend=False,
            hover_columns=["n_counts", "GeneA"],
            show=False,
        )
        hover = fig.data[0].hovertext[0]
        assert "n_counts:" in hover
        assert "GeneA:" in hover


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
            basis="X_umap",
            scores=scores,
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
            basis="X_umap",
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

    def test_legend_method_none_no_legend_raises(self) -> None:
        """method=None + legend=True should raise ValueError."""
        coords = _random_coords()
        rgb = _random_rgb()
        with pytest.raises(ValueError, match="method"):
            plot_embedding_interactive(coords, rgb, method=None, show=False)

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
        assert img.xref == "paper"
        assert img.yref == "paper"

    def test_legend_custom_size(self) -> None:
        coords = _random_coords()
        rgb = _random_rgb()
        fig = plot_embedding_interactive(coords, rgb, method="pca", legend_size=0.5, show=False)
        assert fig.layout.images[0].sizex == 0.5
        assert fig.layout.images[0].sizey == 0.5

    def test_legend_direct_no_names_raises(self) -> None:
        """method='direct' without gene_set_names should raise."""
        coords = _random_coords()
        rgb = _random_rgb()
        with pytest.raises(ValueError, match="gene_set_names"):
            plot_embedding_interactive(
                coords, rgb, method="direct", gene_set_names=None, show=False
            )

    def test_legend_kwargs_forwarded(self) -> None:
        """legend_kwargs should be forwarded without error."""
        coords = _random_coords()
        rgb = _random_rgb()
        # Pass a non-conflicting kwarg (resolution is filtered, use n_sets)
        fig = plot_embedding_interactive(
            coords,
            rgb,
            method="pca",
            legend_kwargs={"n_sets": 3},
            show=False,
        )
        assert len(fig.layout.images) == 1

    def test_render_legend_to_base64_valid_png(self) -> None:
        uri = _render_legend_to_base64("pca")
        assert uri.startswith("data:image/png;base64,")
        # Decode and check PNG magic bytes
        b64_data = uri.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        assert raw[:4] == b"\x89PNG"
