"""Tests for multiscoresplot._scoring."""

from __future__ import annotations

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

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

    def test_rescore_subset_no_duplicates(self):
        """Re-scoring a subset should not leave duplicate columns."""
        adata = _make_synthetic_adata()
        all_sets = {
            "A": ["Gene0", "Gene1"],
            "B": ["Gene2", "Gene3"],
            "C": ["Gene4", "Gene5"],
            "D": ["Gene6", "Gene7"],
        }
        score_gene_sets(adata, all_sets, inplace=True)

        subset = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = score_gene_sets(adata, subset, inplace=True)

        assert list(result.columns) == ["score-A", "score-B"]
        # No duplicated column names in adata.obs
        assert not adata.obs.columns.duplicated().any()

    def test_rescore_updates_values(self):
        """Re-scoring should produce fresh values, not stale ones."""
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result1 = score_gene_sets(adata, gene_sets, inplace=True)
        old_values = result1["score-A"].values.copy()

        # Mutate the underlying data so re-scoring yields different results
        adata.X[:] = 0.0
        result2 = score_gene_sets(adata, gene_sets, inplace=True)

        # Values should have changed (all zeros → different UCell scores)
        assert not np.array_equal(old_values, result2["score-A"].values)

    def test_custom_prefix(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = score_gene_sets(adata, gene_sets, prefix="msp-")

        assert list(result.columns) == ["msp-A", "msp-B"]
        assert "msp-A" in adata.obs.columns

    def test_custom_suffix(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = score_gene_sets(adata, gene_sets, suffix="_v2")

        assert list(result.columns) == ["score-A_v2", "score-B_v2"]

    def test_custom_prefix_and_suffix(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, prefix="msp_score-", suffix="_v2")

        assert list(result.columns) == ["msp_score-A_v2"]

    def test_default_prefix_unchanged(self):
        """Default prefix behaviour is backward compatible."""
        adata = _make_synthetic_adata()
        gene_sets = {"X": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets)

        assert list(result.columns) == ["score-X"]


# ---------------------------------------------------------------------------
# Post-processing tests (clip_pct / normalize)
# ---------------------------------------------------------------------------


class TestScorePostProcessing:
    """Tests for clip_pct and normalize parameters."""

    # -- normalization --

    def test_normalize_stretches_range(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"], "B": ["Gene2", "Gene3"]}
        result = score_gene_sets(adata, gene_sets, normalize=True)

        for col in result.columns:
            assert result[col].min() == pytest.approx(0.0)
            assert result[col].max() == pytest.approx(1.0)

    def test_normalize_constant_scores_to_zero(self):
        """When all cells have the same score, normalization produces 0.0."""
        adata = _make_synthetic_adata(n_cells=10)
        gene_sets = {"A": ["Gene0", "Gene1"]}
        # Score normally, then force constant values
        score_gene_sets(adata, gene_sets, inplace=True)
        adata.obs["score-A"] = 0.5
        # Re-score with a fresh call that will overwrite; instead test helper directly
        from multiscoresplot._scoring import _normalize_scores

        df = adata.obs[["score-A"]].copy()
        _normalize_scores(df)
        assert (df["score-A"] == 0.0).all()

    def test_normalize_preserves_ordering(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        raw = score_gene_sets(adata, gene_sets, normalize=False)
        adata2 = _make_synthetic_adata()
        normed = score_gene_sets(adata2, gene_sets, normalize=True)

        np.testing.assert_array_equal(
            np.argsort(raw["score-A"].values),
            np.argsort(normed["score-A"].values),
        )

    def test_normalize_false_is_noop(self):
        adata1 = _make_synthetic_adata()
        adata2 = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        r1 = score_gene_sets(adata1, gene_sets, normalize=False)
        r2 = score_gene_sets(adata2, gene_sets)
        pd.testing.assert_frame_equal(r1, r2)

    def test_normalize_inplace_updates_adata(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, inplace=True, normalize=True)

        pd.testing.assert_series_equal(adata.obs["score-A"], result["score-A"])
        assert result["score-A"].min() == pytest.approx(0.0)
        assert result["score-A"].max() == pytest.approx(1.0)

    def test_normalize_not_inplace(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, inplace=False, normalize=True)

        assert "score-A" not in adata.obs.columns
        assert result["score-A"].min() == pytest.approx(0.0)
        assert result["score-A"].max() == pytest.approx(1.0)

    # -- clipping --

    def test_clip_pct_float_clips_upper_tail(self):
        adata = _make_synthetic_adata(n_cells=200)
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        raw = score_gene_sets(adata, gene_sets)
        p95 = np.percentile(raw["score-A"].values, 95)

        adata2 = _make_synthetic_adata(n_cells=200)
        clipped = score_gene_sets(adata2, gene_sets, clip_pct=95)
        assert clipped["score-A"].max() <= p95 + 1e-10

    def test_clip_pct_tuple_clips_both_tails(self):
        adata = _make_synthetic_adata(n_cells=200)
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        raw = score_gene_sets(adata, gene_sets)
        p5 = np.percentile(raw["score-A"].values, 5)
        p95 = np.percentile(raw["score-A"].values, 95)

        adata2 = _make_synthetic_adata(n_cells=200)
        clipped = score_gene_sets(adata2, gene_sets, clip_pct=(5, 95))
        assert clipped["score-A"].min() >= p5 - 1e-10
        assert clipped["score-A"].max() <= p95 + 1e-10

    def test_clip_pct_100_is_noop(self):
        adata1 = _make_synthetic_adata()
        adata2 = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        r1 = score_gene_sets(adata1, gene_sets)
        r2 = score_gene_sets(adata2, gene_sets, clip_pct=100)
        pd.testing.assert_frame_equal(r1, r2, check_dtype=False)

    def test_clip_pct_none_is_noop(self):
        adata1 = _make_synthetic_adata()
        adata2 = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        r1 = score_gene_sets(adata1, gene_sets)
        r2 = score_gene_sets(adata2, gene_sets, clip_pct=None)
        pd.testing.assert_frame_equal(r1, r2)

    # -- validation --

    def test_clip_pct_invalid_single_zero_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(ValueError, match=r"clip_pct must be in \(0, 100\]"):
            score_gene_sets(adata, gene_sets, clip_pct=0)

    def test_clip_pct_invalid_single_negative_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(ValueError, match=r"clip_pct must be in \(0, 100\]"):
            score_gene_sets(adata, gene_sets, clip_pct=-5)

    def test_clip_pct_invalid_single_over100_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(ValueError, match=r"clip_pct must be in \(0, 100\]"):
            score_gene_sets(adata, gene_sets, clip_pct=101)

    def test_clip_pct_invalid_tuple_reversed_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(ValueError, match="lo < hi"):
            score_gene_sets(adata, gene_sets, clip_pct=(50, 30))

    def test_clip_pct_invalid_tuple_length_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(ValueError, match="length 2"):
            score_gene_sets(adata, gene_sets, clip_pct=(1, 2, 3))

    def test_clip_pct_invalid_type_raises(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.raises(TypeError, match=r"float or .* tuple"):
            score_gene_sets(adata, gene_sets, clip_pct="99")

    # -- combined --

    def test_clip_then_normalize(self):
        adata = _make_synthetic_adata(n_cells=200)
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        result = score_gene_sets(adata, gene_sets, clip_pct=95, normalize=True)

        for col in result.columns:
            assert result[col].min() == pytest.approx(0.0)
            assert result[col].max() == pytest.approx(1.0)

    def test_clip_then_normalize_ordering(self):
        adata = _make_synthetic_adata(n_cells=200)
        gene_sets = {"A": ["Gene0", "Gene1", "Gene2"]}
        raw = score_gene_sets(adata, gene_sets)

        adata2 = _make_synthetic_adata(n_cells=200)
        result = score_gene_sets(adata2, gene_sets, clip_pct=95, normalize=True)

        # For cells not clipped, rank order is preserved
        raw_vals = raw["score-A"].values
        p95 = np.percentile(raw_vals, 95)
        mask = raw_vals < p95
        if mask.sum() > 1:
            np.testing.assert_array_equal(
                np.argsort(raw_vals[mask]),
                np.argsort(result["score-A"].values[mask]),
            )

    # -- edge cases --

    def test_clip_pct_small_dataset(self):
        adata = _make_synthetic_adata(n_cells=3, n_genes=5)
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, clip_pct=95)
        assert len(result) == 3

    def test_normalize_single_cell(self):
        adata = _make_synthetic_adata(n_cells=1, n_genes=10)
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets, normalize=True)
        assert result["score-A"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Missing gene warning tests (Feature 1)
# ---------------------------------------------------------------------------


class TestMissingGeneWarning:
    """Tests for warnings when gene sets contain missing genes."""

    def test_missing_genes_emits_warning(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "FAKE1", "FAKE2"]}
        with pytest.warns(UserWarning, match=r"Gene set 'A': 2/3 genes \(66\.7%\)"):
            score_gene_sets(adata, gene_sets)

    def test_all_genes_present_no_warning(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            score_gene_sets(adata, gene_sets)

    def test_all_genes_missing_warns_100_pct(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["FAKE1", "FAKE2"]}
        with pytest.warns(UserWarning, match=r"100\.0%"):
            score_gene_sets(adata, gene_sets)

    def test_multiple_sets_only_warns_for_missing(self):
        adata = _make_synthetic_adata()
        gene_sets = {
            "Clean": ["Gene0", "Gene1"],
            "Dirty": ["Gene2", "FAKE1"],
        }
        with pytest.warns(UserWarning) as record:
            score_gene_sets(adata, gene_sets)
        missing_warnings = [w for w in record if "not found" in str(w.message)]
        assert len(missing_warnings) == 1
        assert "Dirty" in str(missing_warnings[0].message)

    def test_warning_contains_gene_names(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "MISSING_X", "MISSING_Y"]}
        with pytest.warns(UserWarning, match="MISSING_X") as record:
            score_gene_sets(adata, gene_sets)
        msg = str(record[0].message)
        assert "MISSING_Y" in msg


# ---------------------------------------------------------------------------
# Data warning tests (Feature 2)
# ---------------------------------------------------------------------------


class TestDataWarnings:
    """Tests for read-only array and negative value warnings."""

    def test_readonly_array_does_not_crash(self):
        adata = _make_synthetic_adata()
        adata.X.flags.writeable = False
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets)
        assert isinstance(result, pd.DataFrame)

    def test_readonly_sparse_does_not_crash(self):
        adata = _make_synthetic_adata()
        adata.X = sp.csr_matrix(adata.X)
        adata.X.data.flags.writeable = False
        gene_sets = {"A": ["Gene0", "Gene1"]}
        result = score_gene_sets(adata, gene_sets)
        assert isinstance(result, pd.DataFrame)

    def test_negative_values_emit_warning(self):
        adata = _make_synthetic_adata()
        adata.X = adata.X - 10.0  # make negative
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.warns(UserWarning, match="negative values"):
            score_gene_sets(adata, gene_sets)

    def test_nonnegative_no_warning(self):
        adata = _make_synthetic_adata()
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            score_gene_sets(adata, gene_sets)

    def test_sparse_negative_values_emit_warning(self):
        adata = _make_synthetic_adata()
        X_dense = adata.X.copy()
        X_dense[0, 0] = -1.0
        adata.X = sp.csr_matrix(X_dense)
        gene_sets = {"A": ["Gene0", "Gene1"]}
        with pytest.warns(UserWarning, match="negative values"):
            score_gene_sets(adata, gene_sets)


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

    def test_clip_and_normalize_real_data(self, adata, marker_genes):
        """clip_pct + normalize on real data should produce values in [0, 1]."""
        result = score_gene_sets(adata, marker_genes, clip_pct=99, normalize=True)

        assert (result.values >= 0).all()
        assert (result.values <= 1).all()
        for col in result.columns:
            assert result[col].min() == pytest.approx(0.0)
            assert result[col].max() == pytest.approx(1.0)
