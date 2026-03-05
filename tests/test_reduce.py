"""Tests for reduce_to_rgb (NMF, ICA, registry) in multiscoresplot._colorspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multiscoresplot._colorspace import (
    get_component_labels,
    reduce_to_rgb,
    register_reducer,
)
from multiscoresplot._scoring import SCORE_PREFIX, score_gene_sets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a score DataFrame with ``score-`` prefixed columns."""
    return pd.DataFrame({f"{SCORE_PREFIX}{k}": v for k, v in values.items()})


# ===========================================================================
# TestReduceToRgbNMF
# ===========================================================================


class TestReduceToRgbNMF:
    """Tests for reduce_to_rgb with method='nmf'."""

    def test_shape(self) -> None:
        rng = np.random.default_rng(10)
        df = _make_scores({f"s{i}": rng.random(50).tolist() for i in range(4)})
        rgb = reduce_to_rgb(df, method="nmf")
        assert rgb.shape == (50, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(11)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(5)})
        rgb = reduce_to_rgb(df, method="nmf")
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_constant_scores(self) -> None:
        df = _make_scores({"A": [0.5] * 20, "B": [0.5] * 20, "C": [0.5] * 20})
        rgb = reduce_to_rgb(df, method="nmf")
        assert rgb.shape == (20, 3)
        np.testing.assert_allclose(rgb, 0.0)

    def test_kwargs_forwarded(self) -> None:
        rng = np.random.default_rng(12)
        df = _make_scores({f"s{i}": rng.random(40).tolist() for i in range(3)})
        # Should not raise — just verifying kwargs are accepted
        rgb = reduce_to_rgb(df, method="nmf", max_iter=50)
        assert rgb.shape == (40, 3)

    def test_2_gene_sets(self) -> None:
        rng = np.random.default_rng(13)
        df = _make_scores({"A": rng.random(30).tolist(), "B": rng.random(30).tolist()})
        rgb = reduce_to_rgb(df, method="nmf")
        assert rgb.shape == (30, 3)


# ===========================================================================
# TestReduceToRgbICA
# ===========================================================================


class TestReduceToRgbICA:
    """Tests for reduce_to_rgb with method='ica'."""

    def test_shape(self) -> None:
        rng = np.random.default_rng(20)
        df = _make_scores({f"s{i}": rng.random(50).tolist() for i in range(4)})
        rgb = reduce_to_rgb(df, method="ica")
        assert rgb.shape == (50, 3)

    def test_rgb_bounded(self) -> None:
        rng = np.random.default_rng(21)
        df = _make_scores({f"s{i}": rng.random(100).tolist() for i in range(5)})
        rgb = reduce_to_rgb(df, method="ica")
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_constant_scores(self) -> None:
        df = _make_scores({"A": [0.5] * 20, "B": [0.5] * 20, "C": [0.5] * 20})
        rgb = reduce_to_rgb(df, method="ica")
        assert rgb.shape == (20, 3)
        np.testing.assert_allclose(rgb, 0.0)

    def test_kwargs_forwarded(self) -> None:
        rng = np.random.default_rng(22)
        df = _make_scores({f"s{i}": rng.random(40).tolist() for i in range(3)})
        rgb = reduce_to_rgb(df, method="ica", max_iter=50)
        assert rgb.shape == (40, 3)

    def test_2_gene_sets(self) -> None:
        rng = np.random.default_rng(23)
        df = _make_scores({"A": rng.random(30).tolist(), "B": rng.random(30).tolist()})
        rgb = reduce_to_rgb(df, method="ica")
        assert rgb.shape == (30, 3)


# ===========================================================================
# TestReduceToRgbErrors
# ===========================================================================


class TestReduceToRgbErrors:
    """Error-handling tests for reduce_to_rgb."""

    def test_unknown_method_raises(self) -> None:
        df = _make_scores({"A": [0.5], "B": [0.5]})
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce_to_rgb(df, method="bogus")

    def test_fewer_than_2_sets_raises(self) -> None:
        df = _make_scores({"only": [0.5]})
        with pytest.raises(ValueError, match="At least 2"):
            reduce_to_rgb(df, method="pca")

    def test_no_score_columns_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="No score columns"):
            reduce_to_rgb(df, method="pca")


# ===========================================================================
# TestRegisterReducer
# ===========================================================================


class TestRegisterReducer:
    """Tests for the register_reducer extensibility mechanism."""

    def test_custom_reducer(self) -> None:
        def _dummy_reducer(X, n_components, **kwargs):
            return np.full((X.shape[0], 3), 0.5)

        register_reducer("dummy_test", _dummy_reducer, component_prefix="DT")
        df = _make_scores({"A": [0.1, 0.2], "B": [0.3, 0.4]})
        rgb = reduce_to_rgb(df, method="dummy_test")
        np.testing.assert_allclose(rgb, 0.5)

    def test_custom_component_labels(self) -> None:
        register_reducer(
            "labeled_test", lambda X, n, **kw: np.zeros((X.shape[0], 3)), component_prefix="LT"
        )
        labels = get_component_labels("labeled_test")
        assert labels == ["LT1", "LT2", "LT3"]

    def test_default_component_labels(self) -> None:
        """Unregistered method falls back to C1/C2/C3."""
        labels = get_component_labels("nonexistent_method")
        assert labels == ["C1", "C2", "C3"]

    def test_builtin_pca_labels(self) -> None:
        labels = get_component_labels("pca")
        assert labels == ["PC1", "PC2", "PC3"]

    def test_builtin_nmf_labels(self) -> None:
        labels = get_component_labels("nmf")
        assert labels == ["NMF1", "NMF2", "NMF3"]

    def test_builtin_ica_labels(self) -> None:
        labels = get_component_labels("ica")
        assert labels == ["IC1", "IC2", "IC3"]


# ===========================================================================
# Integration tests (require test data)
# ===========================================================================


class TestReduceIntegration:
    """End-to-end tests with real scRNA-seq data."""

    def test_nmf_real_data(self, adata, marker_genes) -> None:
        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="nmf")
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_ica_real_data(self, adata, marker_genes) -> None:
        scores = score_gene_sets(adata, marker_genes, inplace=False)
        rgb = reduce_to_rgb(scores, method="ica")
        assert rgb.shape == (adata.n_obs, 3)
        assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
