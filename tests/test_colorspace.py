"""Tests for multiscoresplot._colorspace (pipeline steps 2-3)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from multiscoresplot._colorspace import (
    RGBResult,
    blend_to_rgb,
    project_direct,
    project_pca,
    reduce_to_rgb,
)
from multiscoresplot._scoring import SCORE_PREFIX, score_gene_sets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a score DataFrame with ``score-`` prefixed columns."""
    return pd.DataFrame({f"{SCORE_PREFIX}{k}": v for k, v in values.items()})


# ===========================================================================
# TestBlendToRgb (was TestProjectDirect)
# ===========================================================================


class TestBlendToRgb:
    """Unit tests for the direct multiplicative-blend projection."""

    # ---- shape & bounds ---------------------------------------------------

    def test_shape_2_gene_sets(self) -> None:
        df = _make_scores({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]})
        rgb = blend_to_rgb(df)
        assert rgb.shape == (3, 3)

    def test_shape_3_gene_sets(self) -> None:
        df = _make_scores({"A": [0.2], "B": [0.4], "C": [0.6]})
        rgb = blend_to_rgb(df)
        assert rgb.shape == (1, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(42)
        vals = {f"s{i}": rng.random(50).tolist() for i in range(3)}
        df = _make_scores(vals)
        rgb = blend_to_rgb(df)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    # ---- corner cases: all-zero and all-one scores ------------------------

    def test_all_zero_scores_give_white(self) -> None:
        df = _make_scores({"A": [0.0] * 5, "B": [0.0] * 5})
        rgb = blend_to_rgb(df)
        np.testing.assert_allclose(rgb, 1.0)

    def test_all_one_2_sets_give_black(self) -> None:
        df = _make_scores({"A": [1.0], "B": [1.0]})
        rgb = blend_to_rgb(df)
        np.testing.assert_allclose(rgb, 0.0, atol=1e-12)

    def test_all_one_3_sets_give_black(self) -> None:
        df = _make_scores({"A": [1.0], "B": [1.0], "C": [1.0]})
        rgb = blend_to_rgb(df)
        np.testing.assert_allclose(rgb, 0.0, atol=1e-12)

    # ---- single gene set active → its base colour -------------------------

    def test_single_gene_set_active_2(self) -> None:
        """Only gene set 0 (blue) active → should be blue."""
        df = _make_scores({"A": [1.0], "B": [0.0]})
        rgb = blend_to_rgb(df)
        np.testing.assert_allclose(rgb, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_single_gene_set_active_3(self) -> None:
        """Only gene set 1 (green) active → should be green."""
        df = _make_scores({"A": [0.0], "B": [1.0], "C": [0.0]})
        rgb = blend_to_rgb(df)
        np.testing.assert_allclose(rgb, [[0.0, 1.0, 0.0]], atol=1e-12)

    # ---- custom colours ----------------------------------------------------

    def test_custom_colors_respected(self) -> None:
        custom = [(1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
        df = _make_scores({"A": [1.0], "B": [0.0]})
        rgb = blend_to_rgb(df, colors=custom)
        np.testing.assert_allclose(rgb, [[1.0, 1.0, 0.0]], atol=1e-12)

    # ---- error cases -------------------------------------------------------

    def test_4_gene_sets_raises(self) -> None:
        df = _make_scores({"A": [0.1], "B": [0.2], "C": [0.3], "D": [0.4]})
        with pytest.raises(ValueError, match="reduce_to_rgb"):
            blend_to_rgb(df)

    def test_more_than_4_sets_raises(self) -> None:
        df = _make_scores({f"s{i}": [0.5] for i in range(5)})
        with pytest.raises(ValueError, match="reduce_to_rgb"):
            blend_to_rgb(df)

    def test_fewer_than_2_sets_raises(self) -> None:
        df = _make_scores({"only": [0.5]})
        with pytest.raises(ValueError, match="At least 2"):
            blend_to_rgb(df)

    def test_wrong_color_count_raises(self) -> None:
        df = _make_scores({"A": [0.5], "B": [0.5]})
        with pytest.raises(ValueError, match="Expected 2 colors"):
            blend_to_rgb(df, colors=[(1, 0, 0)])

    def test_no_score_columns_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="No score columns"):
            blend_to_rgb(df)


# ===========================================================================
# TestProjectPCA (kept for backward-compat coverage via reduce_to_rgb)
# ===========================================================================


class TestProjectPCA:
    """Unit tests for the PCA/SVD-based projection via reduce_to_rgb."""

    def test_shape_2_gene_sets(self) -> None:
        rng = np.random.default_rng(0)
        df = _make_scores({"A": rng.random(20).tolist(), "B": rng.random(20).tolist()})
        rgb = reduce_to_rgb(df, method="pca")
        assert rgb.shape == (20, 3)

    def test_shape_3_gene_sets(self) -> None:
        rng = np.random.default_rng(1)
        df = _make_scores({f"s{i}": rng.random(10).tolist() for i in range(3)})
        rgb = reduce_to_rgb(df, method="pca")
        assert rgb.shape == (10, 3)

    def test_shape_5_gene_sets(self) -> None:
        rng = np.random.default_rng(2)
        df = _make_scores({f"s{i}": rng.random(30).tolist() for i in range(5)})
        rgb = reduce_to_rgb(df, method="pca")
        assert rgb.shape == (30, 3)

    def test_shape_10_gene_sets(self) -> None:
        rng = np.random.default_rng(3)
        df = _make_scores({f"s{i}": rng.random(50).tolist() for i in range(10)})
        rgb = reduce_to_rgb(df, method="pca")
        assert rgb.shape == (50, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(4)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(5)})
        rgb = reduce_to_rgb(df, method="pca")
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_each_pc_spans_full_range(self) -> None:
        rng = np.random.default_rng(5)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(4)})
        rgb = reduce_to_rgb(df, method="pca")
        for ch in range(3):
            col = rgb[:, ch]
            if not np.allclose(col, 0.0):
                assert col.min() == pytest.approx(0.0, abs=1e-12)
                assert col.max() == pytest.approx(1.0, abs=1e-12)

    def test_2_gene_sets_third_channel_zero(self) -> None:
        rng = np.random.default_rng(6)
        df = _make_scores({"A": rng.random(20).tolist(), "B": rng.random(20).tolist()})
        rgb = reduce_to_rgb(df, method="pca")
        np.testing.assert_allclose(rgb[:, 2], 0.0)

    def test_constant_scores_no_crash(self) -> None:
        df = _make_scores({"A": [0.5] * 10, "B": [0.5] * 10})
        rgb = reduce_to_rgb(df, method="pca")
        assert rgb.shape == (10, 3)
        np.testing.assert_allclose(rgb, 0.0)

    def test_fewer_than_2_sets_raises(self) -> None:
        df = _make_scores({"only": [0.5]})
        with pytest.raises(ValueError, match="At least 2"):
            reduce_to_rgb(df, method="pca")

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        df = _make_scores({f"s{i}": rng.random(40).tolist() for i in range(4)})
        rgb1 = reduce_to_rgb(df, method="pca")
        rgb2 = reduce_to_rgb(df, method="pca")
        np.testing.assert_array_equal(rgb1, rgb2)


# ===========================================================================
# RGBResult tests
# ===========================================================================


class TestRGBResult:
    """Tests for the RGBResult metadata wrapper."""

    def test_blend_returns_rgb_result(self) -> None:
        df = _make_scores({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]})
        result = blend_to_rgb(df)
        assert isinstance(result, RGBResult)
        assert result.method == "direct"
        assert result.gene_set_names == ["A", "B"]
        assert result.colors is not None

    def test_reduce_returns_rgb_result(self) -> None:
        rng = np.random.default_rng(42)
        df = _make_scores({f"s{i}": rng.random(20).tolist() for i in range(3)})
        result = reduce_to_rgb(df, method="pca")
        assert isinstance(result, RGBResult)
        assert result.method == "pca"
        assert result.gene_set_names == ["s0", "s1", "s2"]
        assert result.colors is None

    def test_rgb_result_as_ndarray(self) -> None:
        df = _make_scores({"A": [0.5, 0.3], "B": [0.1, 0.9]})
        result = blend_to_rgb(df)
        arr = np.asarray(result)
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, result.rgb)

    def test_rgb_result_shape_property(self) -> None:
        df = _make_scores({"A": [0.5], "B": [0.1], "C": [0.9]})
        result = blend_to_rgb(df)
        assert result.shape == (1, 3)

    def test_rgb_result_indexing(self) -> None:
        df = _make_scores({"A": [0.0, 1.0], "B": [1.0, 0.0]})
        result = blend_to_rgb(df)
        row = result[0]
        assert row.shape == (3,)

    def test_rgb_result_comparison(self) -> None:
        df = _make_scores({"A": [0.5, 0.3], "B": [0.1, 0.9]})
        result = blend_to_rgb(df)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_rgb_result_gene_set_names_from_columns(self) -> None:
        df = _make_scores({"qNSCs": [0.5], "aNSCs": [0.3]})
        result = blend_to_rgb(df)
        assert result.gene_set_names == ["qNSCs", "aNSCs"]

    def test_reduce_nmf_returns_rgb_result(self) -> None:
        rng = np.random.default_rng(42)
        df = _make_scores({f"s{i}": rng.random(30).tolist() for i in range(4)})
        result = reduce_to_rgb(df, method="nmf")
        assert isinstance(result, RGBResult)
        assert result.method == "nmf"


# ===========================================================================
# Deprecation tests
# ===========================================================================


class TestDeprecationWrappers:
    """Tests that old names emit DeprecationWarning and still work."""

    def test_project_direct_deprecation_warning(self) -> None:
        df = _make_scores({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rgb = project_direct(df)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "blend_to_rgb" in str(w[0].message)
        assert rgb.shape == (3, 3)

    def test_project_pca_deprecation_warning(self) -> None:
        rng = np.random.default_rng(0)
        df = _make_scores({"A": rng.random(20).tolist(), "B": rng.random(20).tolist()})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rgb = project_pca(df)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "reduce_to_rgb" in str(w[0].message)
        assert rgb.shape == (20, 3)

    def test_project_direct_result_matches_blend(self) -> None:
        df = _make_scores({"A": [0.3, 0.7], "B": [0.8, 0.2]})
        expected = blend_to_rgb(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            actual = project_direct(df)
        np.testing.assert_array_equal(actual, expected)

    def test_project_pca_result_matches_reduce(self) -> None:
        rng = np.random.default_rng(10)
        df = _make_scores({f"s{i}": rng.random(30).tolist() for i in range(4)})
        expected = reduce_to_rgb(df, method="pca")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            actual = project_pca(df)
        np.testing.assert_array_equal(actual, expected)


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestColorspaceIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_blend_to_rgb_real_data(self, adata, marker_genes) -> None:
        # Use only 3 gene sets — blend_to_rgb supports at most 3.
        three_sets = dict(list(marker_genes.items())[:3])
        scores = score_gene_sets(adata, three_sets, inplace=False)
        rgb = blend_to_rgb(scores)
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_reduce_to_rgb_pca_real_data(self, adata, marker_genes) -> None:
        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="pca")
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
