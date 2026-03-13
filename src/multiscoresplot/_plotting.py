"""Dimensionality reduction plotting (pipeline step 4).

Provides ``plot_embedding`` which renders a scatter plot of cells in
embedding space, coloured by projected RGB values from the color-space
module.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import matplotlib
import numpy as np

from multiscoresplot._colorspace import RGBResult, get_component_labels
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

    Returns ``(coords, basis_label)`` where *basis_label* is a label
    derived from the obsm key (for axis labels) or *None* for raw arrays.

    The *basis* parameter is the **full** obsm key name (e.g.
    ``"X_umap"``, ``"umap_consensus"``).  For backward compatibility,
    if *basis* is not found but ``f"X_{basis}"`` exists a
    ``DeprecationWarning`` is emitted and the prefixed key is used.
    """
    arr = np.asarray(adata_or_coords) if not hasattr(adata_or_coords, "obsm") else None

    if arr is not None:
        # Raw ndarray path
        coords = np.asarray(adata_or_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("Raw coordinate array must be 2-D with at least 2 columns.")
        return coords[:, [components[0], components[1]]], None

    # AnnData path
    if basis is None:
        raise ValueError("basis must be provided when passing an AnnData object.")

    adata = adata_or_coords
    obsm = adata.obsm  # type: ignore[attr-defined]

    if basis in obsm:
        key = basis
    elif f"X_{basis}" in obsm:
        key = f"X_{basis}"
        warnings.warn(
            f"Passing basis={basis!r} as a short name is deprecated. "
            f"Use the full obsm key basis={key!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        obsm_keys = list(obsm.keys())
        raise KeyError(f"'{basis}' not found in adata.obsm. Available: {obsm_keys}")

    emb = np.asarray(obsm[key], dtype=np.float64)

    # Derive a nice axis label from the key
    label = key.upper().removeprefix("X_")

    return emb[:, [components[0], components[1]]], label


def _validate_rgb(rgb: NDArray, n_cells: int) -> NDArray:
    """Validate and return the RGB array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (n_cells, 3), got shape {rgb.shape}.")
    if rgb.shape[0] != n_cells:
        raise ValueError(f"rgb has {rgb.shape[0]} rows but coordinates have {n_cells} rows.")
    return rgb


def _unpack_rgb(
    rgb: RGBResult | NDArray,
) -> tuple[NDArray, str | None, list[str] | None, list[tuple[float, float, float]] | None]:
    """Unpack *rgb* into ``(array, method, gene_set_names, colors)``.

    If *rgb* is an :class:`RGBResult`, metadata fields are extracted.
    For a plain ``NDArray`` all metadata fields are ``None``.
    """
    if isinstance(rgb, RGBResult):
        return np.asarray(rgb.rgb), rgb.method, rgb.gene_set_names, rgb.colors
    return np.asarray(rgb), None, None, None


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
    legend_size: float,
    legend_resolution: int,
    legend_kwargs: dict | None,
    method: str,
    gene_set_names: list[str] | None,
    colors: list[tuple[float, float, float]] | None,
) -> None:
    """Create a legend axes and dispatch to ``render_legend``."""
    kwargs = {k: v for k, v in (legend_kwargs or {}).items() if k != "resolution"}

    if legend_style == "inset":
        base = _INSET_BOUNDS.get(legend_loc, _INSET_BOUNDS["lower right"])
        # Scale inset size by legend_size / 0.30 (0.30 is the default size)
        scale = legend_size / 0.30
        bounds = (
            base[0] + base[2] * (1 - scale),  # shift x to keep alignment
            base[1] + base[3] * (1 - scale) if "upper" in legend_loc else base[1],
            base[2] * scale,
            base[3] * scale,
        )
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
        resolution=legend_resolution,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_embedding(
    adata_or_coords: object,
    rgb: RGBResult | NDArray,
    *,
    basis: str | None = None,
    components: tuple[int, int] = (0, 1),
    # legend
    legend: bool = True,
    legend_style: Literal["inset", "side"] = "inset",
    legend_loc: str = "lower right",
    legend_size: float = 0.30,
    legend_resolution: int = 128,
    legend_kwargs: dict | None = None,
    # legend metadata (overrides RGBResult when provided)
    method: str | None = None,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    # scatter
    point_size: float | None = None,
    alpha: float = 1.0,
    # figure
    figsize: tuple[float, float] = (4.0, 4.0),
    dpi: int = 100,
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
        :class:`RGBResult` from ``blend_to_rgb``/``reduce_to_rgb``, or a
        plain ``(n_cells, 3)`` RGB array.  When an ``RGBResult`` is passed
        the ``method``, ``gene_set_names`` and ``colors`` metadata are used
        automatically (explicit parameters still override).
    basis
        Full obsm key name (e.g. ``"X_umap"``, ``"umap_consensus"``).
        Required when *adata_or_coords* is AnnData.
    components
        Which two components to plot (0-indexed).
    legend
        Whether to add a colour-space legend.
    legend_style
        ``"inset"`` (default) or ``"side"``.
    legend_loc
        Position for inset legends (``"lower right"``, etc.).
    legend_size
        Size of the legend as a fraction of the plot area (0-1).
    legend_resolution
        Pixel resolution of the legend image.
    legend_kwargs
        Extra keyword arguments forwarded to ``render_legend``
        (excluding *resolution*, which is controlled by *legend_resolution*).
    method
        ``"direct"``, ``"pca"``, ``"nmf"`` etc. — needed to select the
        legend type.  Inferred from *rgb* when it is an ``RGBResult``.
        If *None* and ``legend=True`` with a plain ndarray, a
        ``ValueError`` is raised.
    gene_set_names
        Gene set labels for the legend.  Inferred from *rgb* when it is
        an ``RGBResult``.
    colors
        Base colours for direct-mode legends.
    point_size
        Scatter point size.  Default: ``120_000 / n_cells``.
    alpha
        Scatter point opacity.
    figsize
        Figure size (inches) when creating a new figure.
    dpi
        Figure resolution when creating a new figure.
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

    # Unpack RGBResult metadata
    rgb_arr, meta_method, meta_names, meta_colors = _unpack_rgb(rgb)
    eff_method = method if method is not None else meta_method
    eff_names = gene_set_names if gene_set_names is not None else meta_names
    eff_colors = colors if colors is not None else meta_colors

    coords, basis_label = _extract_coords(adata_or_coords, basis, components)
    n_cells = coords.shape[0]
    rgb_arr = _validate_rgb(rgb_arr, n_cells)

    if point_size is None:
        point_size = 120_000 / n_cells

    # Create figure / axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Scatter
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=rgb_arr,
        s=point_size,
        marker=".",
        edgecolors="none",
        alpha=alpha,
        plotnonfinite=True,
    )

    _style_axes(ax, basis_label, title, components)

    # Legend
    if legend:
        if eff_method is None:
            raise ValueError(
                "Cannot draw legend: 'method' is unknown. Pass method= explicitly "
                "or use an RGBResult from blend_to_rgb()/reduce_to_rgb()."
            )
        _add_legend(
            fig,
            ax,
            legend_style=legend_style,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_resolution=legend_resolution,
            legend_kwargs=legend_kwargs,
            method=eff_method,
            gene_set_names=eff_names,
            colors=eff_colors,
        )

    if show:
        if matplotlib.is_interactive():
            plt.show()
        return None

    return ax
