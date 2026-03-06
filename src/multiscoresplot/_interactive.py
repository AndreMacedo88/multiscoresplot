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

from multiscoresplot._colorspace import get_component_labels
from multiscoresplot._legend import render_legend
from multiscoresplot._plotting import _extract_coords, _validate_rgb
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
) -> str:
    """Render the legend to a base64-encoded PNG data URI."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

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


def plot_embedding_interactive(
    adata_or_coords: object,
    rgb: NDArray,
    *,
    basis: str | None = None,
    components: tuple[int, int] = (0, 1),
    scores: DataFrame | None = None,
    method: str | None = None,
    gene_set_names: list[str] | None = None,
    # legend
    legend: bool = True,
    legend_loc: str = "lower right",
    legend_size: float = 0.30,
    legend_resolution: int = 128,
    colors: list[tuple[float, float, float]] | None = None,
    # hover / scatter
    hover_columns: list[str] | None = None,
    point_size: float = 2,
    alpha: float = 1.0,
    width: int = 500,
    height: int = 450,
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
        ``(n_cells, 3)`` RGB array from ``blend_to_rgb`` or ``reduce_to_rgb``.
    basis
        Embedding key (e.g. ``"umap"``, ``"pca"``).  Required when
        *adata_or_coords* is AnnData.
    components
        Which two components to plot (0-indexed).
    scores
        DataFrame with ``score-*`` columns.  If *None* and *adata_or_coords*
        is AnnData, scores are auto-extracted from ``adata.obs``.
    method
        Reduction method (``"pca"``, ``"nmf"``, etc.) used to derive RGB.
        Controls the channel labels in hover info.  If *None* or ``"direct"``,
        channels are labeled R/G/B.
    gene_set_names
        Human-readable labels for gene set scores in hover info.
    legend
        Whether to add a colour-space legend overlay.
    legend_loc
        Position for the legend (``"lower right"``, ``"lower left"``,
        ``"upper right"``, ``"upper left"``).
    legend_size
        Size of the legend as a fraction of the plot (0-1).
    legend_resolution
        Pixel resolution of the legend image.
    colors
        Base colours for direct-mode legends.
    hover_columns
        Extra columns from ``adata.obs`` to include in hover info.
    point_size
        Scatter marker size.
    alpha
        Marker opacity.
    width
        Figure width in pixels.
    height
        Figure height in pixels.
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

    coords, basis_label = _extract_coords(adata_or_coords, basis, components)
    n_cells = coords.shape[0]
    rgb = _validate_rgb(rgb, n_cells)

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
            gene_set_names
            if gene_set_names is not None and len(gene_set_names) == len(score_cols)
            else [c[len(SCORE_PREFIX) :] for c in score_cols]
        )
        score_vals = score_df[score_cols].to_numpy(dtype=np.float64)
        for i in range(n_cells):
            for j, label in enumerate(labels):
                hover_parts[i].append(f"{label}: {score_vals[i, j]:.3f}")

    # 2. RGB channel values
    if method is not None and method != "direct":
        channel_labels = get_component_labels(method)
    else:
        channel_labels = ["R", "G", "B"]

    for i in range(n_cells):
        for j, ch_label in enumerate(channel_labels):
            hover_parts[i].append(f"{ch_label}: {rgb[i, j]:.2f}")

    # 3. Extra .obs columns
    if hover_columns is not None:
        if not has_obs:
            raise ValueError("hover_columns requires an AnnData object, not raw coordinates.")
        obs = adata_or_coords.obs  # type: ignore[attr-defined]
        missing = [c for c in hover_columns if c not in obs.columns]
        if missing:
            raise KeyError(f"Columns not found in adata.obs: {missing}")

        import pandas as _pd

        for col_name in hover_columns:
            col = obs[col_name]
            is_numeric = _pd.api.types.is_numeric_dtype(col)
            for i in range(n_cells):
                val = col.iloc[i]
                if is_numeric:
                    hover_parts[i].append(f"{col_name}: {val:.3f}")
                else:
                    hover_parts[i].append(f"{col_name}: {val}")

    hover_text = ["<br>".join(parts) for parts in hover_parts]

    # --- Build color strings ---
    marker_colors = [
        f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})" for r, g, b in rgb
    ]

    # --- Axis labels ---
    if basis_label is not None:
        xaxis_title = f"{basis_label}{components[0] + 1}"
        yaxis_title = f"{basis_label}{components[1] + 1}"
    else:
        xaxis_title = ""
        yaxis_title = ""

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

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(title=xaxis_title, scaleanchor="y"),
        yaxis=dict(title=yaxis_title),
        plot_bgcolor="white",
    )

    # Legend (skip silently when method is None, or direct mode without gene_set_names)
    if legend and method is not None and (method != "direct" or gene_set_names is not None):
        _add_plotly_legend(
            fig,
            method=method,
            gene_set_names=gene_set_names,
            colors=colors,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_resolution=legend_resolution,
        )

    if show:
        fig.show()
        return None

    return fig  # type: ignore[no-any-return]
