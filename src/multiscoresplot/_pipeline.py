"""One-step convenience function for the full scoring-to-plot pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from pandas import DataFrame

    from multiscoresplot._colorspace import RGBResult

__all__ = ["plot_scores"]

# Parameter names that belong to the plotting functions rather than the reducer.
_PLOT_PARAMS = frozenset(
    {
        "components",
        "legend",
        "legend_style",
        "legend_loc",
        "legend_size",
        "legend_resolution",
        "legend_kwargs",
        "method",
        "gene_set_names",
        "colors",
        "point_size",
        "alpha",
        "figsize",
        "dpi",
        "title",
        "ax",
        "scores",
        "hover_columns",
    }
)


def plot_scores(
    adata: AnnData,
    gene_sets: dict[str, list[str]],
    *,
    # scoring
    max_rank: int = 1500,
    chunk_size: int = 1000,
    n_jobs: int = -1,
    inplace: bool = True,
    clip_pct: float | tuple[float, float] | None = None,
    normalize: bool = False,
    # prefix/suffix
    prefix: str = "score-",
    suffix: str = "",
    # color mapping
    method: str | Callable[..., NDArray] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    n_components: int = 3,
    component_prefix: str | None = None,
    # plotting
    basis: str = "X_umap",
    interactive: bool = False,
    show: bool = True,
    **kwargs: object,
) -> tuple[DataFrame, RGBResult, Axes | Figure | object | None]:
    """Score gene sets, map to RGB, and plot — all in one call.

    This is a convenience wrapper around the 3-step pipeline:
    :func:`score_gene_sets` → :func:`blend_to_rgb` / :func:`reduce_to_rgb`
    → :func:`plot_embedding` / :func:`plot_embedding_interactive`.

    Parameters
    ----------
    adata
        Annotated data matrix (cells x genes) with a precomputed embedding
        in ``.obsm[basis]``.
    gene_sets
        Mapping of gene set names to lists of gene symbols.
    max_rank
        Rank cap passed to pyUCell.
    chunk_size
        Number of cells processed per batch.
    n_jobs
        Parallelism (``-1`` = all cores).
    inplace
        If *True*, scores are stored in ``adata.obs``.
    clip_pct
        Percentile clipping — see :func:`score_gene_sets`.
    normalize
        Min-max rescaling — see :func:`score_gene_sets`.
    prefix
        Column name prefix for score columns (default ``"score-"``).
        Forwarded to scoring, color mapping, and interactive plotting.
    suffix
        Column name suffix for score columns (default ``""``).
        Forwarded to scoring, color mapping, and interactive plotting.
    method
        Color-mapping method.  ``None`` (default) auto-selects: ``"blend"``
        for ≤ 3 gene sets, ``"pca"`` for > 3.  Pass ``"blend"`` explicitly
        to force multiplicative blending, any string/callable for reduction.
    colors
        Base colours for :func:`blend_to_rgb`.
    n_components
        Number of components for reduction (max 3).
    component_prefix
        Label prefix for reduction legend axes.
    basis
        Full obsm key for the embedding (e.g. ``"X_umap"``).
    interactive
        If *True*, use :func:`plot_embedding_interactive` (requires plotly).
    show
        If *True* (default), display the plot.
    **kwargs
        Extra keyword arguments forwarded to the plotting function or
        reducer as appropriate.

    Returns
    -------
    tuple[DataFrame, RGBResult, Axes | Figure | None]
        ``(scores, rgb, plot_result)`` — the score DataFrame, the
        :class:`RGBResult`, and the plot return value (``Axes``, Plotly
        ``Figure``, or ``None``).
    """
    from multiscoresplot._colorspace import blend_to_rgb, reduce_to_rgb
    from multiscoresplot._interactive import plot_embedding_interactive
    from multiscoresplot._plotting import plot_embedding
    from multiscoresplot._scoring import score_gene_sets

    # --- Step 1: Score ---
    scores = score_gene_sets(
        adata,
        gene_sets,
        max_rank=max_rank,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        inplace=inplace,
        clip_pct=clip_pct,
        normalize=normalize,
        prefix=prefix,
        suffix=suffix,
    )

    # --- Step 2: Map to RGB ---
    n_sets = len(gene_sets)

    # Split kwargs into plot vs reducer buckets
    plot_kwargs: dict[str, object] = {}
    reduce_kwargs: dict[str, object] = {}
    for key, val in kwargs.items():
        if key in _PLOT_PARAMS:
            plot_kwargs[key] = val
        else:
            reduce_kwargs[key] = val

    if method is None:
        method = "blend" if n_sets <= 3 else "pca"

    if method == "blend":
        rgb = blend_to_rgb(scores, colors=colors, prefix=prefix, suffix=suffix)
    else:
        rgb = reduce_to_rgb(
            scores,
            method=method,
            n_components=n_components,
            component_prefix=component_prefix,
            prefix=prefix,
            suffix=suffix,
            **reduce_kwargs,
        )

    # --- Step 3: Plot ---
    if interactive:
        plot_result = plot_embedding_interactive(
            adata,
            rgb,
            basis=basis,
            prefix=prefix,
            suffix=suffix,
            show=show,
            **plot_kwargs,  # type: ignore[arg-type]
        )
    else:
        plot_result = plot_embedding(
            adata,
            rgb,
            basis=basis,
            show=show,
            **plot_kwargs,  # type: ignore[arg-type]
        )

    return scores, rgb, plot_result
