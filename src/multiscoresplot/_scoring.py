"""Gene set scoring (pipeline step 1)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pyucell

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

__all__ = ["score_gene_sets"]

SCORE_PREFIX = "score-"
_UCELL_SUFFIX = "_UCell"


def _validate_clip_pct(clip_pct: float | tuple[float, float]) -> tuple[float, float]:
    """Validate and normalise *clip_pct* to a ``(lo, hi)`` tuple.

    Raises
    ------
    TypeError
        If *clip_pct* is not a float, int, or length-2 tuple.
    ValueError
        If percentile values are out of range.
    """
    if isinstance(clip_pct, int | float):
        if clip_pct <= 0 or clip_pct > 100:
            raise ValueError(f"clip_pct must be in (0, 100], got {clip_pct}")
        return (0.0, float(clip_pct))
    if isinstance(clip_pct, tuple):
        if len(clip_pct) != 2:
            raise ValueError(f"clip_pct tuple must have length 2, got {len(clip_pct)}")
        lo, hi = clip_pct
        if not (0 <= lo < hi <= 100):
            raise ValueError(f"clip_pct tuple must satisfy 0 <= lo < hi <= 100, got ({lo}, {hi})")
        return (float(lo), float(hi))
    raise TypeError(f"clip_pct must be a float or (lo, hi) tuple, got {type(clip_pct).__name__}")


def _clip_scores(df: pd.DataFrame, lo: float, hi: float) -> None:
    """Percentile-clip each column of *df* in place."""
    for col in df.columns:
        vals = df[col].values
        lo_val, hi_val = np.percentile(vals, [lo, hi])
        df[col] = np.clip(vals, lo_val, hi_val)


def _normalize_scores(df: pd.DataFrame) -> None:
    """Min-max normalise each column of *df* in place to [0, 1]."""
    for col in df.columns:
        vals = df[col].values
        col_min, col_max = vals.min(), vals.max()
        rng = col_max - col_min
        if rng == 0:
            df[col] = 0.0
        else:
            df[col] = (vals - col_min) / rng


def score_gene_sets(
    adata: AnnData,
    gene_sets: dict[str, list[str]],
    *,
    max_rank: int = 1500,
    chunk_size: int = 1000,
    n_jobs: int = -1,
    inplace: bool = True,
    prefix: str = SCORE_PREFIX,
    suffix: str = "",
    clip_pct: float | tuple[float, float] | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Score each cell for each gene set using UCell.

    Parameters
    ----------
    adata
        Annotated data matrix (cells x genes).
    gene_sets
        Mapping of gene set names to lists of gene symbols.
    max_rank
        Rank cap passed to pyUCell (tune to median genes per cell).
    chunk_size
        Number of cells processed per batch.
    n_jobs
        Parallelism (``-1`` = all cores).
    inplace
        If *True* (default), scores are stored in ``adata.obs`` as
        ``{prefix}{name}{suffix}`` columns. If *False*, scores are returned
        but **not** kept in ``adata.obs``.
    prefix
        Column name prefix (default ``"score-"``).
    suffix
        Column name suffix (default ``""``).
    clip_pct
        Percentile clipping (winsorization), applied per gene set. A single
        float (e.g., ``99``) clips the upper tail at that percentile. A tuple
        ``(lo, hi)`` clips both tails. ``None`` (default) disables clipping.
    normalize
        If *True*, per-gene-set min-max rescaling so min → 0 and max → 1.
        Applied **after** clipping. Default *False*.

    Returns
    -------
    DataFrame with index ``adata.obs_names`` and columns
    ``["{prefix}{name}{suffix}" for name in gene_sets]``.
    When *clip_pct* or *normalize* are used, the returned values reflect the
    post-processed scores.

    Notes
    -----
    **Missing genes:** Genes in a gene set that are not found in
    ``adata.var_names`` are imputed by pyUCell with worst-case rank
    (``max_rank``), which degrades the signal. A ``UserWarning`` is emitted
    listing the missing genes so you can verify your gene symbols.

    **Read-only arrays:** After ``sc.pp.scale()`` or similar operations,
    ``adata.X`` may become a read-only numpy array. This function
    automatically copies ``adata.X`` when it detects a read-only array to
    prevent crashes inside pyUCell.

    **Negative values:** UCell is rank-based and designed for raw or
    normalized (non-negative) counts. If ``adata.X`` contains negative
    values (e.g., after ``sc.pp.scale(zero_center=True)``), a
    ``UserWarning`` is emitted. Consider using ``adata.raw.to_adata()``
    or a layer with non-negative values for more meaningful scores.
    """
    # --- validate inputs ---
    if not isinstance(gene_sets, dict) or len(gene_sets) == 0:
        raise ValueError("gene_sets must be a non-empty dict.")
    for name, genes in gene_sets.items():
        if not isinstance(genes, list) or len(genes) == 0:
            raise ValueError(f"Gene set {name!r} must be a non-empty list of gene names.")
        if not all(isinstance(g, str) for g in genes):
            raise ValueError(f"All gene names in {name!r} must be strings.")

    # --- warn about missing genes ---
    var_names_set = set(adata.var_names)
    for name, genes in gene_sets.items():
        missing = [g for g in genes if g not in var_names_set]
        if missing:
            n_missing = len(missing)
            n_total = len(genes)
            pct = 100.0 * n_missing / n_total
            warnings.warn(
                f"Gene set {name!r}: {n_missing}/{n_total} genes ({pct:.1f}%) "
                f"not found in adata.var_names and will be imputed: {missing}",
                UserWarning,
                stacklevel=2,
            )

    # --- ensure X is writeable (work around read-only arrays after sc.pp.scale) ---
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        if not X.data.flags.writeable:
            adata.X = X.copy()
    elif isinstance(X, np.ndarray) and not X.flags.writeable:
        adata.X = X.copy()

    # --- warn about negative values ---
    X = adata.X
    if sp.issparse(X):
        has_negative = X.data.min() < 0 if X.nnz > 0 else False
    else:
        has_negative = bool(np.any(X < 0))
    if has_negative:
        warnings.warn(
            "adata.X contains negative values. UCell is rank-based and designed "
            "for raw or normalized (non-negative) counts. Scoring scaled/centered "
            "data (e.g., after sc.pp.scale()) may produce unexpected results. "
            "Consider using adata.raw.to_adata() or a layer with non-negative values.",
            UserWarning,
            stacklevel=2,
        )

    # --- clean pre-existing columns for the requested gene sets ---
    for name in gene_sets:
        ucell_col = f"{name}{_UCELL_SUFFIX}"
        score_col = f"{prefix}{name}{suffix}"
        for col in (ucell_col, score_col):
            if col in adata.obs.columns:
                adata.obs.drop(columns=[col], inplace=True)

    # --- run UCell ---
    pyucell.compute_ucell_scores(
        adata,
        signatures=gene_sets,
        max_rank=max_rank,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
    )

    # --- rename UCell columns to project convention ---
    rename_map: dict[str, str] = {}
    for name in gene_sets:
        ucell_col = f"{name}{_UCELL_SUFFIX}"
        score_col = f"{prefix}{name}{suffix}"
        rename_map[ucell_col] = score_col

    adata.obs.rename(columns=rename_map, inplace=True)

    score_cols = [f"{prefix}{name}{suffix}" for name in gene_sets]
    result = adata.obs[score_cols].copy()

    # --- optional post-processing ---
    if clip_pct is not None:
        lo, hi = _validate_clip_pct(clip_pct)
        _clip_scores(result, lo, hi)
    if normalize:
        _normalize_scores(result)
    if clip_pct is not None or normalize:
        adata.obs[score_cols] = result

    if not inplace:
        adata.obs.drop(columns=score_cols, inplace=True)

    return result
