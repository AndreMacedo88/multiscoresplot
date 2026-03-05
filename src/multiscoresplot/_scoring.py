"""Gene set scoring (pipeline step 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyucell

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

__all__ = ["score_gene_sets"]

SCORE_PREFIX = "score-"
_UCELL_SUFFIX = "_UCell"


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

    Returns
    -------
    DataFrame with index ``adata.obs_names`` and columns
    ``["{prefix}{name}{suffix}" for name in gene_sets]``.
    """
    # --- validate inputs ---
    if not isinstance(gene_sets, dict) or len(gene_sets) == 0:
        raise ValueError("gene_sets must be a non-empty dict.")
    for name, genes in gene_sets.items():
        if not isinstance(genes, list) or len(genes) == 0:
            raise ValueError(f"Gene set {name!r} must be a non-empty list of gene names.")
        if not all(isinstance(g, str) for g in genes):
            raise ValueError(f"All gene names in {name!r} must be strings.")

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

    if not inplace:
        adata.obs.drop(columns=score_cols, inplace=True)

    return result
