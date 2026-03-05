"""Dimensionality reduction plotting (pipeline step 4).

Provides ``plot_embedding`` which renders a scatter plot of cells in
embedding space, coloured by projected RGB values from the color-space
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib
import numpy as np

from multiscoresplot._colorspace import get_component_labels
from multiscoresplot._legend import render_legend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

__all__ = ["plot_embedding"]

# ---------------------------------------------------------------------------
# Inset legend position lookup
# ---------------------------------------------------------------------------

_INSET_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "lower right": (0.68, 0.02, 0.30, 0.30),
    "lower left": (0.02, 0.02, 0.30, 0.30),
    "upper right": (0.68, 0.68, 0.30, 0.30),
    "upper left": (0.02, 0.68, 0.30, 0.30),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_coords(
    adata_or_coords: object,
    basis: str | None,
    components: tuple[int, int],
) -> tuple[NDArray, str | None]:
    """Extract 2-D coordinates from AnnData or a raw array.

    Returns ``(coords, basis_label)`` where *basis_label* is the
    capitalised basis name (for axis labels) or *None* for raw arrays.
    """
    arr = np.asarray(adata_or_coords) if not hasattr(adata_or_coords, "obsm") else None

    if arr is not None:
        # Raw ndarray path
        coords = np.asarray(adata_or_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("Raw coordinate array must be 2-D with at least 2 columns.")
        return coords[:, [components[0], components[1]]], None

    # AnnData path — import lazily
    if basis is None:
        raise ValueError("basis must be provided when passing an AnnData object.")

    adata = adata_or_coords
    key = f"X_{basis}"
    if key not in adata.obsm:  # type: ignore[attr-defined]
        obsm_keys = list(adata.obsm.keys())  # type: ignore[attr-defined]
        raise KeyError(f"'{key}' not found in adata.obsm. Available: {obsm_keys}")

    emb = np.asarray(adata.obsm[key], dtype=np.float64)  # type: ignore[attr-defined]
    return emb[:, [components[0], components[1]]], basis.upper()


def _validate_rgb(rgb: NDArray, n_cells: int) -> NDArray:
    """Validate and return the RGB array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (n_cells, 3), got shape {rgb.shape}.")
    if rgb.shape[0] != n_cells:
        raise ValueError(f"rgb has {rgb.shape[0]} rows but coordinates have {n_cells} rows.")
    return rgb


def _style_axes(
    ax: Axes,
    basis_label: str | None,
    title: str | None,
    components: tuple[int, int],
) -> None:
    """Apply scanpy-like styling to the scatter axes."""
    ax.set_xticks([])
    ax.set_yticks([])

    if basis_label is not None:
        ax.set_xlabel(f"{basis_label}{components[0] + 1}", fontsize=10)
        ax.set_ylabel(f"{basis_label}{components[1] + 1}", fontsize=10)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_facecolor("white")


def _add_legend(
    fig: Figure,
    ax: Axes,
    *,
    legend_style: str,
    legend_loc: str,
    legend_kwargs: dict | None,
    method: str,
    gene_set_names: list[str] | None,
    colors: list[tuple[float, float, float]] | None,
) -> None:
    """Create a legend axes and dispatch to ``render_legend``."""
    kwargs = legend_kwargs or {}

    if legend_style == "inset":
        bounds = _INSET_BOUNDS.get(legend_loc, _INSET_BOUNDS["lower right"])
        legend_ax = ax.inset_axes(bounds)
    else:
        # Side placement: add a new axes to the right
        pos = ax.get_position()
        legend_ax = fig.add_axes((pos.x1 + 0.02, pos.y0, 0.15, pos.height))

    # Derive component labels for reduction methods.
    component_labels = None
    if method != "direct":
        component_labels = get_component_labels(method)

    render_legend(
        legend_ax,
        method,
        gene_set_names=gene_set_names,
        colors=colors,
        component_labels=component_labels,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_embedding(
    adata_or_coords: object,
    rgb: NDArray,
    *,
    basis: str | None = None,
    components: tuple[int, int] = (0, 1),
    # legend
    legend: bool = True,
    legend_style: Literal["inset", "side"] = "inset",
    legend_loc: str = "lower right",
    legend_kwargs: dict | None = None,
    # legend metadata
    method: str | None = None,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    # scatter
    point_size: float | None = None,
    alpha: float = 1.0,
    # figure
    figsize: tuple[float, float] = (4.0, 4.0),
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes | None:
    """Plot embedding coordinates coloured by projected RGB values.

    Parameters
    ----------
    adata_or_coords
        An ``AnnData`` object (with *basis* in ``.obsm``) or a raw
        ``(n_cells, 2)`` coordinate array.
    rgb
        ``(n_cells, 3)`` RGB array from ``project_direct`` or ``project_pca``.
    basis
        Embedding key (e.g. ``"umap"``, ``"pca"``).  Required when
        *adata_or_coords* is AnnData.
    components
        Which two components to plot (0-indexed).
    legend
        Whether to add a colour-space legend.
    legend_style
        ``"inset"`` (default) or ``"side"``.
    legend_loc
        Position for inset legends (``"lower right"``, etc.).
    legend_kwargs
        Extra keyword arguments forwarded to ``render_legend``.
    method
        ``"direct"`` or ``"pca"`` — needed to select the legend type.
        If *None* and ``legend=True``, the legend is silently skipped.
    gene_set_names
        Gene set labels for the legend.
    colors
        Base colours for direct-mode legends.
    point_size
        Scatter point size.  Default: ``120_000 / n_cells``.
    alpha
        Scatter point opacity.
    figsize
        Figure size when creating a new figure.
    title
        Plot title.
    ax
        Pre-existing axes to draw on.  A new figure is created if *None*.
    show
        If *True*, call ``plt.show()`` and return *None*.  If *False*,
        return the axes (scanpy convention).

    Returns
    -------
    Axes or None
        The axes when ``show=False``; *None* when ``show=True``.
    """
    import matplotlib.pyplot as plt

    coords, basis_label = _extract_coords(adata_or_coords, basis, components)
    n_cells = coords.shape[0]
    rgb = _validate_rgb(rgb, n_cells)

    if point_size is None:
        point_size = 120_000 / n_cells

    # Create figure / axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Scatter
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=rgb,
        s=point_size,
        marker=".",
        edgecolors="none",
        alpha=alpha,
        plotnonfinite=True,
    )

    _style_axes(ax, basis_label, title, components)

    # Legend
    if legend and method is not None:
        _add_legend(
            fig,
            ax,
            legend_style=legend_style,
            legend_loc=legend_loc,
            legend_kwargs=legend_kwargs,
            method=method,
            gene_set_names=gene_set_names,
            colors=colors,
        )

    if show:
        if matplotlib.is_interactive():
            plt.show()
        return None

    return ax
