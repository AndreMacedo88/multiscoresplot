"""Simplex / ternary legend rendering (pipeline step 5).

Provides ``render_legend`` which dispatches to the correct legend renderer
based on the projection method (direct vs PCA) and the number of gene sets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from multiscoresplot._colorspace import DEFAULT_COLORS_2, DEFAULT_COLORS_3

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = ["render_legend"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _blend_grid_multiplicative(
    s_grid: NDArray,
    colors: list[tuple[float, float, float]],
) -> NDArray:
    """Compute multiplicative blend for a grid of score vectors.

    Replicates the same formula used by ``project_direct``: starting from
    white, each gene set darkens toward its base colour proportional to
    the score.
    """
    rgb = np.ones((s_grid.shape[0], 3), dtype=np.float64)
    for i, c in enumerate(colors):
        rgb *= 1.0 - s_grid[:, i : i + 1] * (1.0 - np.asarray(c, dtype=np.float64))
    return np.clip(rgb, 0.0, 1.0)


def _blend_grid_additive(s_grid: NDArray) -> NDArray:
    """Compute additive blend for a grid of barycentric coordinates.

    Each vertex maps to a pure R, G, or B channel.  The pixel colour is
    the weighted sum of the three basis colours.
    """
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return np.clip(s_grid @ basis, 0.0, 1.0)


def _barycentric_triangle(
    resolution: int,
) -> tuple[NDArray, NDArray, tuple[int, int]]:
    """Build an equilateral-triangle image grid with barycentric coordinates.

    Returns
    -------
    coords : (n_inside, 3)
        Barycentric coordinates for every pixel inside the triangle.
    mask : (H, W) bool
        True where the pixel is inside the triangle.
    shape : (H, W)
        Image dimensions.
    """
    # Equilateral triangle vertices in pixel space (unit triangle).
    # v0 = top, v1 = bottom-left, v2 = bottom-right
    height = int(resolution * np.sqrt(3) / 2)
    width = resolution

    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    # Map pixel coords to barycentric coords of equilateral triangle.
    # Vertices: v0=(width/2, 0), v1=(0, height-1), v2=(width-1, height-1)
    x0, y0 = width / 2, 0.0
    x1, y1 = 0.0, height - 1.0
    x2, y2 = width - 1.0, height - 1.0

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    lam0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    lam1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    lam2 = 1.0 - lam0 - lam1

    eps = -1e-3
    mask = (lam0 >= eps) & (lam1 >= eps) & (lam2 >= eps)

    coords_flat = np.column_stack([lam0[mask], lam1[mask], lam2[mask]])
    # Clip to [0, 1] to handle numerical edge effects.
    coords_flat = np.clip(coords_flat, 0.0, 1.0)
    # Re-normalise so rows sum to 1.
    row_sums = coords_flat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    coords_flat /= row_sums

    return coords_flat, mask, (height, width)


def _add_brightness_bar(
    ax: Axes,
    resolution: int,
    *,
    label_left: str = "low",
    label_right: str = "high",
) -> None:
    """Add a horizontal black-to-white gradient bar below *ax* content.

    Uses ``ax.inset_axes`` to create a small bar region at the bottom.
    """
    bar_ax = ax.inset_axes((0.1, -0.05, 0.8, 0.06))
    gradient = np.linspace(0, 1, resolution).reshape(1, -1)
    gradient_rgb = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
    bar_ax.imshow(gradient_rgb, aspect="auto", origin="lower")
    bar_ax.set_xticks([0, resolution - 1])
    bar_ax.set_xticklabels([label_left, label_right], fontsize=7)
    bar_ax.set_yticks([])
    for spine in bar_ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Legend renderers (private)
# ---------------------------------------------------------------------------


def _legend_direct_square(
    ax: Axes,
    gene_set_names: list[str],
    colors: list[tuple[float, float, float]],
    resolution: int,
) -> None:
    """Render a 2-set colour square legend."""
    s0 = np.linspace(0, 1, resolution)
    s1 = np.linspace(0, 1, resolution)
    S0, S1 = np.meshgrid(s0, s1)
    grid = np.column_stack([S0.ravel(), S1.ravel()])

    img_flat = _blend_grid_multiplicative(grid, colors)
    img = img_flat.reshape(resolution, resolution, 3)

    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="equal")
    ax.set_xlabel(gene_set_names[0], fontsize=8)
    ax.set_ylabel(gene_set_names[1], fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=6)


def _legend_direct_triangle(
    ax: Axes,
    gene_set_names: list[str],
    colors: list[tuple[float, float, float]],
    resolution: int,
) -> None:
    """Render a 3-set simplex (equilateral triangle) legend."""
    coords, mask, (height, width) = _barycentric_triangle(resolution)
    rgb_flat = _blend_grid_multiplicative(coords, colors)

    img = np.ones((height, width, 4), dtype=np.float64)  # RGBA, default white+transparent
    img[:, :, 3] = 0.0  # fully transparent outside
    rgba_inside = np.column_stack([rgb_flat, np.ones(rgb_flat.shape[0])])
    img[mask] = rgba_inside

    ax.imshow(img, origin="upper", aspect="equal")
    ax.set_xlim(-5, width + 5)
    ax.set_ylim(height + 5, -5)
    ax.axis("off")

    # Labels at vertices: v0=top, v1=bottom-left, v2=bottom-right
    ax.text(width / 2, -3, gene_set_names[0], ha="center", va="bottom", fontsize=8)
    ax.text(-3, height, gene_set_names[1], ha="right", va="top", fontsize=8)
    ax.text(width + 3, height, gene_set_names[2], ha="left", va="top", fontsize=8)


def _legend_direct_4set(
    ax: Axes,
    gene_set_names: list[str],
    colors: list[tuple[float, float, float]],
    brightness_alpha: float,
    resolution: int,
) -> None:
    """Render a 4-set legend: hue square + brightness bar."""
    # Hue square (first two gene sets)
    _legend_direct_square(ax, gene_set_names[:2], colors, resolution)

    # Brightness bar (last two gene sets)
    # Ramp from full colour (no dimming) to dimmed (factor = 1 - alpha).
    bar_ax = ax.inset_axes((0.1, -0.12, 0.8, 0.06))
    gradient = np.linspace(0, 1, resolution).reshape(1, -1)
    factor = 1.0 - brightness_alpha * gradient
    bar_rgb = np.ones((1, resolution, 3)) * factor[:, :, np.newaxis]
    bar_rgb = np.clip(bar_rgb, 0.0, 1.0)
    bar_ax.imshow(bar_rgb, aspect="auto", origin="lower")

    brt_label = f"{gene_set_names[2]} + {gene_set_names[3]}"
    bar_ax.set_xticks([0, resolution - 1])
    bar_ax.set_xticklabels(["0", "1"], fontsize=6)
    bar_ax.set_yticks([])
    bar_ax.set_xlabel(brt_label, fontsize=7)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)


def _legend_pca_triangle(
    ax: Axes,
    gene_set_names: list[str] | None,
    resolution: int,
) -> None:
    """Render a PCA legend: additive RGB triangle + brightness bar."""
    coords, mask, (height, width) = _barycentric_triangle(resolution)
    rgb_flat = _blend_grid_additive(coords)

    img = np.zeros((height, width, 4), dtype=np.float64)  # RGBA, transparent outside
    rgba_inside = np.column_stack([rgb_flat, np.ones(rgb_flat.shape[0])])
    img[mask] = rgba_inside

    ax.imshow(img, origin="upper", aspect="equal")
    ax.set_xlim(-5, width + 5)
    ax.set_ylim(height + 5, -5)
    ax.axis("off")

    # Label vertices with PC names
    labels = ["PC1", "PC2", "PC3"]
    ax.text(width / 2, -3, labels[0], ha="center", va="bottom", fontsize=8, color="red")
    ax.text(-3, height, labels[1], ha="right", va="top", fontsize=8, color="green")
    ax.text(width + 3, height, labels[2], ha="left", va="top", fontsize=8, color="blue")

    # Brightness bar
    _add_brightness_bar(ax, resolution)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_legend(
    ax: Axes,
    method: Literal["direct", "pca"],
    *,
    n_sets: int | None = None,
    gene_set_names: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    brightness_alpha: float = 0.6,
    resolution: int = 128,
) -> Axes:
    """Draw a color-space legend onto *ax*.

    Parameters
    ----------
    ax
        Matplotlib axes to draw on.
    method
        ``"direct"`` or ``"pca"`` — must match the projection used to
        generate the RGB values.
    n_sets
        Number of gene sets (required for ``method="direct"``).
    gene_set_names
        Human-readable labels.  For ``"direct"`` mode the length must match
        *n_sets*.  For ``"pca"`` mode, names are optional (vertices are
        labelled PC1/PC2/PC3 regardless).
    colors
        Base colours for direct mode.  Ignored for PCA.
    brightness_alpha
        Brightness modulation strength for 4-set direct legends.
    resolution
        Pixel resolution of the legend image.

    Returns
    -------
    Axes
        The axes with the legend drawn on it.
    """
    if method == "pca":
        _legend_pca_triangle(ax, gene_set_names, resolution)
        return ax

    # --- direct mode ---
    if n_sets is None:
        if gene_set_names is not None:
            n_sets = len(gene_set_names)
        else:
            raise ValueError(
                "For method='direct', provide n_sets or gene_set_names "
                "so the legend type can be determined."
            )

    if n_sets < 2 or n_sets > 4:
        raise ValueError(f"Direct legend supports 2-4 gene sets, got {n_sets}.")

    # Default gene set names if not provided.
    if gene_set_names is None:
        gene_set_names = [f"Set {i + 1}" for i in range(n_sets)]
    elif len(gene_set_names) != n_sets:
        raise ValueError(f"gene_set_names length ({len(gene_set_names)}) != n_sets ({n_sets}).")

    # Default colours.
    if colors is None:
        if n_sets == 2:
            colors = DEFAULT_COLORS_2
        elif n_sets == 3:
            colors = DEFAULT_COLORS_3
        else:  # 4-set: hue pair uses 2-set defaults
            colors = DEFAULT_COLORS_2

    if n_sets == 2:
        _legend_direct_square(ax, gene_set_names, colors, resolution)
    elif n_sets == 3:
        _legend_direct_triangle(ax, gene_set_names, colors, resolution)
    else:
        _legend_direct_4set(ax, gene_set_names, colors, brightness_alpha, resolution)

    return ax
