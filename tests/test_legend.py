"""Tests for multiscoresplot._legend (pipeline step 5)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from multiscoresplot._colorspace import DEFAULT_COLORS_2, DEFAULT_COLORS_3
from multiscoresplot._legend import render_legend


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ===========================================================================
# TestRenderLegendDirect
# ===========================================================================


class TestRenderLegendDirect:
    """Tests for direct-mode legends."""

    def test_2set_square_renders(self) -> None:
        _, ax = plt.subplots()
        result = render_legend(ax, "direct", n_sets=2, gene_set_names=["A", "B"])
        assert result is ax
        assert len(ax.images) > 0

    def test_3set_triangle_renders(self) -> None:
        _, ax = plt.subplots()
        result = render_legend(ax, "direct", n_sets=3, gene_set_names=["A", "B", "C"])
        assert result is ax
        assert len(ax.images) > 0

    def test_4set_raises(self) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="2-3 gene sets"):
            render_legend(ax, "direct", n_sets=4, gene_set_names=["A", "B", "C", "D"])

    def test_missing_n_sets_and_names_raises(self) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="n_sets or gene_set_names"):
            render_legend(ax, "direct")

    def test_invalid_n_sets_raises(self) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="2-3 gene sets"):
            render_legend(ax, "direct", n_sets=5)

    def test_custom_colors_accepted(self) -> None:
        _, ax = plt.subplots()
        custom = [(1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
        result = render_legend(ax, "direct", n_sets=2, gene_set_names=["X", "Y"], colors=custom)
        assert result is ax
        assert len(ax.images) > 0

    def test_2set_corner_pixel_values(self) -> None:
        """Verify the 4 corners of a 2-set square legend."""
        _, ax = plt.subplots()
        render_legend(ax, "direct", n_sets=2, gene_set_names=["A", "B"], resolution=64)
        img_data = ax.images[0].get_array()
        # img_data shape: (64, 64, 3), origin="lower"
        # (0,0) = bottom-left = s0=0, s1=0 → white
        np.testing.assert_allclose(img_data[0, 0], [1.0, 1.0, 1.0], atol=0.02)
        # (1,0) = bottom-right = s0=1, s1=0 → blue (default colour 0)
        np.testing.assert_allclose(img_data[0, -1], list(DEFAULT_COLORS_2[0]), atol=0.05)
        # (0,1) = top-left = s0=0, s1=1 → red (default colour 1)
        np.testing.assert_allclose(img_data[-1, 0], list(DEFAULT_COLORS_2[1]), atol=0.05)
        # (1,1) = top-right = s0=1, s1=1 → black
        np.testing.assert_allclose(img_data[-1, -1], [0.0, 0.0, 0.0], atol=0.05)


# ===========================================================================
# TestRenderLegendPCA
# ===========================================================================


class TestRenderLegendPCA:
    """Tests for PCA-mode legends (backward compatibility)."""

    def test_pca_triangle_renders(self) -> None:
        _, ax = plt.subplots()
        result = render_legend(ax, "pca")
        assert result is ax
        assert len(ax.images) > 0

    def test_pca_n_sets_ignored(self) -> None:
        """n_sets is irrelevant for PCA — should not error."""
        _, ax = plt.subplots()
        result = render_legend(ax, "pca", n_sets=10)
        assert result is ax

    def test_pca_triangle_vertex_colors(self) -> None:
        """Triangle vertices should be approximately red, green, blue."""
        _, ax = plt.subplots()
        render_legend(ax, "pca", resolution=128)
        img_data = ax.images[0].get_array()
        # Image shape is (height, width, 4) RGBA
        height, width = img_data.shape[:2]

        # v0 (top-center): red channel dominant — check a pixel near top center
        top_pixel = img_data[2, width // 2, :3]
        assert top_pixel[0] > 0.5, f"Top vertex R channel too low: {top_pixel}"

        # v1 (bottom-left): green channel dominant
        bl_pixel = img_data[height - 3, 2, :3]
        assert bl_pixel[1] > 0.5, f"Bottom-left vertex G channel too low: {bl_pixel}"

        # v2 (bottom-right): blue channel dominant
        br_pixel = img_data[height - 3, width - 3, :3]
        assert br_pixel[2] > 0.5, f"Bottom-right vertex B channel too low: {br_pixel}"


# ===========================================================================
# TestRenderLegendReduce
# ===========================================================================


class TestRenderLegendReduce:
    """Tests for the generalized reduction legend mode."""

    def test_reduce_renders_with_custom_labels(self) -> None:
        _, ax = plt.subplots()
        result = render_legend(ax, "reduce", component_labels=["NMF1", "NMF2", "NMF3"])
        assert result is ax
        assert len(ax.images) > 0

    def test_reduce_default_labels(self) -> None:
        """method='reduce' without component_labels uses C1/C2/C3."""
        _, ax = plt.subplots()
        result = render_legend(ax, "reduce")
        assert result is ax
        # Verify labels by checking text objects on axes
        texts = [t.get_text() for t in ax.texts]
        assert "C1" in texts
        assert "C2" in texts
        assert "C3" in texts

    def test_pca_backward_compat_labels(self) -> None:
        """method='pca' without component_labels uses PC1/PC2/PC3."""
        _, ax = plt.subplots()
        render_legend(ax, "pca")
        texts = [t.get_text() for t in ax.texts]
        assert "PC1" in texts
        assert "PC2" in texts
        assert "PC3" in texts

    def test_nmf_method_renders(self) -> None:
        """Any non-'direct' method string is treated as reduction."""
        _, ax = plt.subplots()
        result = render_legend(ax, "nmf", component_labels=["NMF1", "NMF2", "NMF3"])
        assert result is ax
        assert len(ax.images) > 0

    def test_ica_method_renders(self) -> None:
        _, ax = plt.subplots()
        result = render_legend(ax, "ica", component_labels=["IC1", "IC2", "IC3"])
        assert result is ax
        assert len(ax.images) > 0


# ===========================================================================
# TestLegendPixelValues
# ===========================================================================


class TestLegendPixelValues:
    """Tests for correct pixel colour output."""

    def test_3set_triangle_vertices_match_base_colors(self) -> None:
        """Triangle vertex pixels should approximate the default base colours."""
        _, ax = plt.subplots()
        render_legend(
            ax,
            "direct",
            n_sets=3,
            gene_set_names=["R", "G", "B"],
            resolution=128,
        )
        img_data = ax.images[0].get_array()
        height, width = img_data.shape[:2]

        # v0 (top-center): gene set 0 score = 1, others = 0 → base colour 0
        top = img_data[2, width // 2, :3]
        expected_top = np.array(DEFAULT_COLORS_3[0])
        np.testing.assert_allclose(top, expected_top, atol=0.15)

        # v1 (bottom-left): gene set 1 score = 1, others = 0 → base colour 1
        bl = img_data[height - 3, 2, :3]
        expected_bl = np.array(DEFAULT_COLORS_3[1])
        np.testing.assert_allclose(bl, expected_bl, atol=0.15)

        # v2 (bottom-right): gene set 2 score = 1, others = 0 → base colour 2
        br = img_data[height - 3, width - 3, :3]
        expected_br = np.array(DEFAULT_COLORS_3[2])
        np.testing.assert_allclose(br, expected_br, atol=0.15)

    def test_2set_square_corners_multiplicative(self) -> None:
        """2-set square corners should match multiplicative blend output."""
        _, ax = plt.subplots()
        colors = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
        render_legend(
            ax,
            "direct",
            n_sets=2,
            gene_set_names=["A", "B"],
            colors=colors,
            resolution=64,
        )
        img_data = ax.images[0].get_array()
        # Mid-point of x-axis, y=0 → s0=0.5, s1=0 → (1-0.5*(1-0), 1, 1-0.5*(1-1)) = (0.5, 1, 1)
        # Actually: colour0=(0,0,1), s0=0.5 → grad = 1 - 0.5*(1-(0,0,1)) = (0.5, 0.5, 1.0)
        # colour1=(1,0,0), s1=0 → grad = (1, 1, 1)
        # product = (0.5, 0.5, 1.0)
        mid_bottom = img_data[0, 32, :]
        np.testing.assert_allclose(mid_bottom, [0.5, 0.5, 1.0], atol=0.1)
