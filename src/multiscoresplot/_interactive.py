"""Interactive Plotly-based embedding plots (optional dependency).

Provides ``plot_embedding_interactive`` which renders a WebGL-accelerated
scatter plot of cells in embedding space, coloured by projected RGB values,
with rich hover information.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np

from multiscoresplot._colorspace import RGBResult, get_component_labels
from multiscoresplot._legend import render_legend
from multiscoresplot._plotting import _extract_coords, _unpack_rgb, _validate_rgb
from multiscoresplot._scoring import SCORE_PREFIX

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame

__all__ = ["plot_embedding_interactive"]

# ---------------------------------------------------------------------------
# Legend position lookup (mirrors _INSET_BOUNDS in _plotting.py)
# ---------------------------------------------------------------------------

_PLOTLY_LEGEND_POS: dict[str, dict[str, str | float]] = {
    "lower right": {"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
    "lower left": {"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
    "upper right": {"x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top"},
    "upper left": {"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
    "lower center": {"x": 0.5, "y": 0.02, "xanchor": "center", "yanchor": "bottom"},
    "upper center": {"x": 0.5, "y": 0.98, "xanchor": "center", "yanchor": "top"},
    "center": {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle"},
    "center left": {"x": 0.02, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
    "center right": {"x": 0.98, "y": 0.5, "xanchor": "right", "yanchor": "middle"},
}


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def _render_legend_to_base64(
    method: str,
    *,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    component_labels: list[str] | None = None,
    resolution: int = 128,
    legend_kwargs: dict | None = None,
) -> str:
    """Render the legend to a base64-encoded PNG data URI."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    kwargs = {k: v for k, v in (legend_kwargs or {}).items() if k != "resolution"}

    fig = Figure(figsize=(2, 2), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    render_legend(
        ax,
        method,
        gene_set_names=gene_set_names,
        colors=colors,
        component_labels=component_labels,
        resolution=resolution,
        **kwargs,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=150)
    matplotlib.pyplot.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _add_plotly_legend(
    fig: object,
    *,
    method: str,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    legend_loc: str = "lower right",
    legend_size: float = 0.30,
    legend_resolution: int = 128,
    legend_kwargs: dict | None = None,
) -> None:
    """Render a matplotlib legend and embed it into a Plotly figure."""
    # For non-direct methods, derive component labels
    component_labels = None
    if method != "direct":
        component_labels = get_component_labels(method)

    uri = _render_legend_to_base64(
        method,
        gene_set_names=gene_set_names,
        colors=colors,
        component_labels=component_labels,
        resolution=legend_resolution,
        legend_kwargs=legend_kwargs,
    )

    pos = _PLOTLY_LEGEND_POS.get(legend_loc, _PLOTLY_LEGEND_POS["lower right"])

    fig.add_layout_image(  # type: ignore[attr-defined]
        source=uri,
        xref="paper",
        yref="paper",
        x=pos["x"],
        y=pos["y"],
        xanchor=pos["xanchor"],
        yanchor=pos["yanchor"],
        sizex=legend_size,
        sizey=legend_size,
        sizing="contain",
        layer="above",
    )


def _ensure_plotly():
    """Lazy-import plotly, raising a helpful error if not installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plotting. "
            "Install it with: pip install 'multiscoresplot[interactive]'"
        ) from None
    return go


def _resolve_hover_column(
    adata: object,
    col_name: str,
    n_cells: int,
) -> tuple[object, bool]:
    """Resolve a hover column from adata.obs or adata.X (gene expression).

    Returns ``(values, is_numeric)`` where *values* is an array-like with
    one entry per cell.  Raises ``KeyError`` if the column is not found in
    either ``adata.obs`` or ``adata.var_names``.
    """
    import pandas as _pd

    obs = adata.obs  # type: ignore[attr-defined]
    if col_name in obs.columns:
        col = obs[col_name]
        return col, _pd.api.types.is_numeric_dtype(col)

    # Fall back to gene expression in adata.X
    var_names = list(adata.var_names)  # type: ignore[attr-defined]
    if col_name in var_names:
        gene_idx = var_names.index(col_name)
        X = adata.X  # type: ignore[attr-defined]
        import scipy.sparse as sp

        if sp.issparse(X):
            expr = np.asarray(X[:, gene_idx].toarray()).ravel()
        else:
            expr = np.asarray(X[:, gene_idx]).ravel()
        return expr, True

    raise KeyError(f"'{col_name}' not found in adata.obs columns or adata.var_names.")


def plot_embedding_interactive(
    adata_or_coords: object,
    rgb: RGBResult | NDArray,
    *,
    basis: str | None = None,
    components: tuple[int, int] = (0, 1),
    scores: DataFrame | None = None,
    # legend metadata (overrides RGBResult when provided)
    method: str | None = None,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    # legend
    legend: bool = True,
    legend_loc: str = "lower right",
    legend_size: float = 0.30,
    legend_resolution: int = 128,
    legend_kwargs: dict | None = None,
    # hover / scatter
    hover_columns: list[str] | None = None,
    point_size: float = 2,
    alpha: float = 1.0,
    # figure
    figsize: tuple[float, float] = (6.5, 6.0),
    dpi: int = 100,
    title: str = "",
    show: bool = True,
) -> object | None:
    """Interactive Plotly scatter plot of embedding coordinates coloured by RGB.

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
    scores
        DataFrame with ``score-*`` columns.  If *None* and *adata_or_coords*
        is AnnData, scores are auto-extracted from ``adata.obs``.
    method
        Reduction method (``"pca"``, ``"nmf"``, etc.) used to derive RGB.
        Inferred from *rgb* when it is an ``RGBResult``.  Controls the
        channel labels in hover info and the legend type.  If *None* or
        ``"direct"``, channels are labeled R/G/B.
    gene_set_names
        Human-readable labels for the legend and hover score labels.
        Inferred from *rgb* when it is an ``RGBResult``.
    colors
        Base colours for direct-mode legends.
    legend
        Whether to add a colour-space legend overlay.
    legend_loc
        Position for the legend (``"lower right"``, ``"lower left"``,
        ``"upper right"``, ``"upper left"``).
    legend_size
        Size of the legend as a fraction of the plot (0-1).
    legend_resolution
        Pixel resolution of the legend image.
    legend_kwargs
        Extra keyword arguments forwarded to ``render_legend``
        (excluding *resolution*, which is controlled by *legend_resolution*).
    hover_columns
        Extra columns to include in hover info.  Looked up first in
        ``adata.obs``; if not found there, looked up in ``adata.var_names``
        to display individual gene expression values.
    point_size
        Scatter marker size.
    alpha
        Marker opacity.
    figsize
        Figure size as ``(width_inches, height_inches)``.  Multiplied by
        *dpi* to obtain pixel dimensions.
    dpi
        Resolution (dots per inch).  ``figsize * dpi`` gives the pixel
        dimensions of the Plotly figure.
    title
        Plot title.
    show
        If *True*, call ``fig.show()`` and return *None*.  If *False*,
        return the ``plotly.graph_objects.Figure``.

    Returns
    -------
    Figure or None
        The figure when ``show=False``; *None* when ``show=True``.
    """
    go = _ensure_plotly()

    # Unpack RGBResult metadata
    rgb_arr, meta_method, meta_names, meta_colors = _unpack_rgb(rgb)
    eff_method = method if method is not None else meta_method
    eff_names = gene_set_names if gene_set_names is not None else meta_names
    eff_colors = colors if colors is not None else meta_colors

    coords, basis_label = _extract_coords(adata_or_coords, basis, components)
    n_cells = coords.shape[0]
    rgb_arr = _validate_rgb(rgb_arr, n_cells)

    # Determine if we have an AnnData object
    has_obs = hasattr(adata_or_coords, "obs")

    # --- Build hover text ---
    hover_parts: list[list[str]] = [[] for _ in range(n_cells)]

    # 1. Gene set scores
    score_df: DataFrame | None = scores
    if score_df is None and has_obs:
        obs = adata_or_coords.obs  # type: ignore[attr-defined]
        score_cols = [c for c in obs.columns if c.startswith(SCORE_PREFIX)]
        if score_cols:
            score_df = obs[score_cols]

    if score_df is not None:
        score_cols = [c for c in score_df.columns if c.startswith(SCORE_PREFIX)]
        labels = (
            eff_names
            if eff_names is not None and len(eff_names) == len(score_cols)
            else [c[len(SCORE_PREFIX) :] for c in score_cols]
        )
        score_vals = score_df[score_cols].to_numpy(dtype=np.float64)
        for i in range(n_cells):
            for j, label in enumerate(labels):
                hover_parts[i].append(f"{label}: {score_vals[i, j]:.3f}")

    # 2. RGB channel values
    if eff_method is not None and eff_method != "direct":
        channel_labels = get_component_labels(eff_method)
    else:
        channel_labels = ["R", "G", "B"]

    for i in range(n_cells):
        for j, ch_label in enumerate(channel_labels):
            hover_parts[i].append(f"{ch_label}: {rgb_arr[i, j]:.2f}")

    # 3. Extra columns (adata.obs first, then adata.var_names for gene expr)
    if hover_columns is not None:
        if not has_obs:
            raise ValueError("hover_columns requires an AnnData object, not raw coordinates.")

        for col_name in hover_columns:
            values, is_numeric = _resolve_hover_column(adata_or_coords, col_name, n_cells)
            for i in range(n_cells):
                val = values.iloc[i] if hasattr(values, "iloc") else values[i]  # type: ignore[union-attr,index]
                if is_numeric:
                    hover_parts[i].append(f"{col_name}: {val:.3f}")
                else:
                    hover_parts[i].append(f"{col_name}: {val}")

    hover_text = ["<br>".join(parts) for parts in hover_parts]

    # --- Build color strings ---
    marker_colors = [
        f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})" for r, g, b in rgb_arr
    ]

    # --- Axis labels ---
    if basis_label is not None:
        xaxis_title = f"{basis_label}{components[0] + 1}"
        yaxis_title = f"{basis_label}{components[1] + 1}"
    else:
        xaxis_title = ""
        yaxis_title = ""

    # --- Pixel dimensions from figsize + dpi ---
    width_px = int(figsize[0] * dpi)
    height_px = int(figsize[1] * dpi)

    # --- Create figure ---
    fig = go.Figure(
        data=go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=point_size,
                color=marker_colors,
            ),
            hovertext=hover_text,
            hoverinfo="text",
        ),
    )

    _axis_style = dict(
        showticklabels=False,
        ticks="",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )
    fig.update_layout(
        width=width_px,
        height=height_px,
        title=title,
        xaxis=dict(title=xaxis_title, scaleanchor="y", **_axis_style),
        yaxis=dict(title=yaxis_title, **_axis_style),
        plot_bgcolor="white",
    )

    # Legend — consistent with plot_embedding()
    if legend:
        if eff_method is None:
            raise ValueError(
                "Cannot draw legend: 'method' is unknown. Pass method= explicitly "
                "or use an RGBResult from blend_to_rgb()/reduce_to_rgb()."
            )
        if eff_method == "direct" and eff_names is None:
            raise ValueError(
                "Cannot draw direct-mode legend without gene_set_names. "
                "Pass gene_set_names= or use an RGBResult from blend_to_rgb()."
            )
        _add_plotly_legend(
            fig,
            method=eff_method,
            gene_set_names=eff_names,
            colors=eff_colors,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_resolution=legend_resolution,
            legend_kwargs=legend_kwargs,
        )

    if show:
        fig.show()
        return None

    return fig  # type: ignore[no-any-return]
