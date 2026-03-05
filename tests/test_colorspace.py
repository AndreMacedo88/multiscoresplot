"""Tests for multiscoresplot._colorspace (pipeline steps 2-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multiscoresplot._colorspace import project_direct, project_pca
from multiscoresplot._scoring import SCORE_PREFIX, score_gene_sets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a score DataFrame with ``score-`` prefixed columns."""
    return pd.DataFrame({f"{SCORE_PREFIX}{k}": v for k, v in values.items()})


# ===========================================================================
# TestProjectDirect
# ===========================================================================


class TestProjectDirect:
    """Unit tests for the direct multiplicative-blend projection."""

    # ---- shape & bounds ---------------------------------------------------

    def test_shape_2_gene_sets(self) -> None:
        df = _make_scores({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]})
        rgb = project_direct(df)
        assert rgb.shape == (3, 3)

    def test_shape_3_gene_sets(self) -> None:
        df = _make_scores({"A": [0.2], "B": [0.4], "C": [0.6]})
        rgb = project_direct(df)
        assert rgb.shape == (1, 3)

    def test_shape_4_gene_sets(self) -> None:
        df = _make_scores({"A": [0.1], "B": [0.2], "C": [0.3], "D": [0.4]})
        rgb = project_direct(df)
        assert rgb.shape == (1, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(42)
        vals = {f"s{i}": rng.random(50).tolist() for i in range(3)}
        df = _make_scores(vals)
        rgb = project_direct(df)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    # ---- corner cases: all-zero and all-one scores ------------------------

    def test_all_zero_scores_give_white(self) -> None:
        df = _make_scores({"A": [0.0] * 5, "B": [0.0] * 5})
        rgb = project_direct(df)
        np.testing.assert_allclose(rgb, 1.0)

    def test_all_one_2_sets_give_black(self) -> None:
        df = _make_scores({"A": [1.0], "B": [1.0]})
        rgb = project_direct(df)
        # blue * red multiplicative → (0,0,1)*(1,0,0) element-wise product
        # gradient_blue = 1 - 1*(1 - (0,0,1)) = (0,0,1)
        # gradient_red  = 1 - 1*(1 - (1,0,0)) = (1,0,0)
        # product = (0,0,0)
        np.testing.assert_allclose(rgb, 0.0, atol=1e-12)

    def test_all_one_3_sets_give_black(self) -> None:
        df = _make_scores({"A": [1.0], "B": [1.0], "C": [1.0]})
        rgb = project_direct(df)
        np.testing.assert_allclose(rgb, 0.0, atol=1e-12)

    # ---- single gene set active → its base colour -------------------------

    def test_single_gene_set_active_2(self) -> None:
        """Only gene set 0 (blue) active → should be blue."""
        df = _make_scores({"A": [1.0], "B": [0.0]})
        rgb = project_direct(df)
        np.testing.assert_allclose(rgb, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_single_gene_set_active_3(self) -> None:
        """Only gene set 1 (green) active → should be green."""
        df = _make_scores({"A": [0.0], "B": [1.0], "C": [0.0]})
        rgb = project_direct(df)
        np.testing.assert_allclose(rgb, [[0.0, 1.0, 0.0]], atol=1e-12)

    # ---- custom colours ----------------------------------------------------

    def test_custom_colors_respected(self) -> None:
        custom = [(1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
        df = _make_scores({"A": [1.0], "B": [0.0]})
        rgb = project_direct(df, colors=custom)
        # Only A active with colour (1,1,0): gradient = (1,1,0), product = (1,1,0)
        np.testing.assert_allclose(rgb, [[1.0, 1.0, 0.0]], atol=1e-12)

    # ---- error cases -------------------------------------------------------

    def test_more_than_4_sets_raises(self) -> None:
        df = _make_scores({f"s{i}": [0.5] for i in range(5)})
        with pytest.raises(ValueError, match="project_pca"):
            project_direct(df)

    def test_fewer_than_2_sets_raises(self) -> None:
        df = _make_scores({"only": [0.5]})
        with pytest.raises(ValueError, match="At least 2"):
            project_direct(df)

    def test_wrong_color_count_raises(self) -> None:
        df = _make_scores({"A": [0.5], "B": [0.5]})
        with pytest.raises(ValueError, match="Expected 2 colors"):
            project_direct(df, colors=[(1, 0, 0)])

    def test_no_score_columns_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="No score columns"):
            project_direct(df)

    # ---- 4-gene-set brightness behaviour ----------------------------------

    def test_4_sets_brightness_pair_zero_no_dimming(self) -> None:
        """Brightness pair all zero → base hue unchanged."""
        df = _make_scores({"A": [1.0], "B": [0.0], "C": [0.0], "D": [0.0]})
        rgb = project_direct(df)
        # Hue pair: A=1(blue), B=0 → blue. Brightness pair zero → no dimming.
        np.testing.assert_allclose(rgb, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_4_sets_brightness_pair_one_alpha_one_gives_black(self) -> None:
        """Brightness pair all 1 with alpha=1 → complete darkening."""
        df = _make_scores({"A": [0.0], "B": [0.0], "C": [1.0], "D": [1.0]})
        rgb = project_direct(df, brightness_alpha=1.0)
        # Hue pair all zero → white. Brightness mean=1, alpha=1 → 1*(1-1)=0.
        np.testing.assert_allclose(rgb, 0.0, atol=1e-12)

    def test_pair_order_infer_reorders_by_variance(self) -> None:
        """``pair_order='infer'`` assigns most variable scores to hue."""
        rng = np.random.default_rng(99)
        n = 200
        # C and D have much higher variance than A and B.
        df = _make_scores(
            {
                "A": (rng.random(n) * 0.01).tolist(),
                "B": (rng.random(n) * 0.01).tolist(),
                "C": rng.random(n).tolist(),
                "D": rng.random(n).tolist(),
            }
        )
        rgb_infer = project_direct(df, pair_order="infer")
        # With "infer", C and D (high variance) become the hue pair.
        # With "columns", A and B (low variance) are the hue pair.
        rgb_columns = project_direct(df, pair_order="columns")
        # The outputs should differ because the pair assignment changed.
        assert not np.allclose(rgb_infer, rgb_columns)


# ===========================================================================
# TestProjectPCA
# ===========================================================================


class TestProjectPCA:
    """Unit tests for the PCA/SVD-based projection."""

    def test_shape_2_gene_sets(self) -> None:
        rng = np.random.default_rng(0)
        df = _make_scores({"A": rng.random(20).tolist(), "B": rng.random(20).tolist()})
        rgb = project_pca(df)
        assert rgb.shape == (20, 3)

    def test_shape_3_gene_sets(self) -> None:
        rng = np.random.default_rng(1)
        df = _make_scores({f"s{i}": rng.random(10).tolist() for i in range(3)})
        rgb = project_pca(df)
        assert rgb.shape == (10, 3)

    def test_shape_5_gene_sets(self) -> None:
        rng = np.random.default_rng(2)
        df = _make_scores({f"s{i}": rng.random(30).tolist() for i in range(5)})
        rgb = project_pca(df)
        assert rgb.shape == (30, 3)

    def test_shape_10_gene_sets(self) -> None:
        rng = np.random.default_rng(3)
        df = _make_scores({f"s{i}": rng.random(50).tolist() for i in range(10)})
        rgb = project_pca(df)
        assert rgb.shape == (50, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(4)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(5)})
        rgb = project_pca(df)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_each_pc_spans_full_range(self) -> None:
        rng = np.random.default_rng(5)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(4)})
        rgb = project_pca(df)
        for ch in range(3):
            col = rgb[:, ch]
            if not np.allclose(col, 0.0):
                assert col.min() == pytest.approx(0.0, abs=1e-12)
                assert col.max() == pytest.approx(1.0, abs=1e-12)

    def test_2_gene_sets_third_channel_zero(self) -> None:
        rng = np.random.default_rng(6)
        df = _make_scores({"A": rng.random(20).tolist(), "B": rng.random(20).tolist()})
        rgb = project_pca(df)
        np.testing.assert_allclose(rgb[:, 2], 0.0)

    def test_constant_scores_no_crash(self) -> None:
        df = _make_scores({"A": [0.5] * 10, "B": [0.5] * 10})
        rgb = project_pca(df)
        assert rgb.shape == (10, 3)
        np.testing.assert_allclose(rgb, 0.0)

    def test_fewer_than_2_sets_raises(self) -> None:
        df = _make_scores({"only": [0.5]})
        with pytest.raises(ValueError, match="At least 2"):
            project_pca(df)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        df = _make_scores({f"s{i}": rng.random(40).tolist() for i in range(4)})
        rgb1 = project_pca(df)
        rgb2 = project_pca(df)
        np.testing.assert_array_equal(rgb1, rgb2)


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestColorspaceIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_project_direct_real_data(self, adata, marker_genes) -> None:
        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = project_direct(scores)
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_project_pca_real_data(self, adata, marker_genes) -> None:
        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = project_pca(scores)
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
