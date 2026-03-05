"""Color space construction and cell projection (pipeline steps 2-3).

Provides two projection strategies:

* **Direct** (`project_direct`): multiplicative blending from white using
  explicit base colors.  Supports 2-4 gene sets.
* **PCA** (`project_pca`): SVD-based dimensionality reduction to 3 color
  channels.  Works for any number of gene sets (≥ 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from multiscoresplot._scoring import SCORE_PREFIX

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame

__all__ = ["project_direct", "project_pca"]

# ---- default color palettes ------------------------------------------------

DEFAULT_COLORS_2: list[tuple[float, float, float]] = [
    (0.0, 0.0, 1.0),  # blue
    (1.0, 0.0, 0.0),  # red
]

DEFAULT_COLORS_3: list[tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
]


# ---- private helpers --------------------------------------------------------


def _validate_score_columns(scores: DataFrame, prefix: str = SCORE_PREFIX) -> list[str]:
    """Return the ``score-*`` column names, raising if none are found."""
    cols = [c for c in scores.columns if c.startswith(prefix)]
    if not cols:
        msg = (
            "No score columns found. Expected columns starting with "
            f"'{prefix}'. Run score_gene_sets() first."
        )
        raise ValueError(msg)
    return cols


def _order_by_variance(score_matrix: NDArray, cols: list[str]) -> list[int]:
    """Return column indices sorted by descending standard deviation."""
    stds = np.std(score_matrix, axis=0)
    return list(np.argsort(stds)[::-1])


def _multiplicative_blend(
    score_matrix: NDArray,
    colors: list[tuple[float, float, float]],
) -> NDArray:
    """Blend gene set scores into RGB via multiplicative gradients from white.

    For each gene set *i* with base color ``c_i`` and score ``s_i``:
        ``gradient_i = 1 - s_i * (1 - c_i)``

    The final colour is the element-wise product of all gradients.
    """
    n_cells = score_matrix.shape[0]
    rgb = np.ones((n_cells, 3), dtype=np.float64)

    for i, color in enumerate(colors):
        c = np.asarray(color, dtype=np.float64)  # (3,)
        s = score_matrix[:, i : i + 1]  # (n_cells, 1)
        gradient = 1.0 - s * (1.0 - c)  # (n_cells, 3)
        rgb *= gradient

    return np.clip(rgb, 0.0, 1.0)


def _pca_via_svd(X: NDArray, n_components: int) -> NDArray:
    """Project *X* onto its first *n_components* principal components via SVD.

    Returns min-max normalised PC scores, zero-padded to 3 columns when
    ``n_components < 3``.
    """
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Check for constant input (all zeros after centering).
    if np.allclose(X_centered, 0.0):
        return np.zeros((X.shape[0], 3), dtype=np.float64)

    U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    k = min(n_components, U.shape[1])
    pc_scores = U[:, :k] * S[:k]

    # Min-max normalise each PC to [0, 1].
    for j in range(k):
        col = pc_scores[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo > 0:
            pc_scores[:, j] = (col - lo) / (hi - lo)
        else:
            pc_scores[:, j] = 0.0

    # Zero-pad to 3 channels if fewer than 3 PCs.
    if k < 3:
        pad = np.zeros((pc_scores.shape[0], 3 - k), dtype=np.float64)
        pc_scores = np.hstack([pc_scores, pad])

    return np.asarray(pc_scores)


# ---- public API -------------------------------------------------------------


def project_direct(
    scores: DataFrame,
    *,
    colors: list[tuple[float, float, float]] | None = None,
    brightness_alpha: float = 0.6,
    pair_order: Literal["columns", "infer"] = "columns",
) -> NDArray:
    """Map gene set scores to RGB via multiplicative blending from white.

    Parameters
    ----------
    scores
        DataFrame returned by :func:`score_gene_sets`.  Only columns whose
        names start with ``score-`` are used.
    colors
        One ``(R, G, B)`` tuple per gene set.  If *None*, defaults are chosen
        based on the number of gene sets (2 → blue/red, 3 → RGB).  For 4
        gene sets, the first pair is coloured and the second pair modulates
        brightness, so only 2 colours should be supplied (or *None* for the
        blue/red default).
    brightness_alpha
        Strength of brightness modulation for the 4-gene-set case (0 = none,
        1 = full darkening).
    pair_order
        Strategy for splitting 4 gene sets into a hue pair and a brightness
        pair.  ``"columns"`` (default) uses the first two score columns for
        hue and the last two for brightness.  ``"infer"`` assigns the two
        most variable scores to hue.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, 3)`` RGB array with values in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 2 or more than 4 gene sets are present, or if the
        number of supplied colours does not match expectations.
    """
    score_cols = _validate_score_columns(scores)
    n_sets = len(score_cols)

    if n_sets < 2:
        raise ValueError("At least 2 gene sets are required.")
    if n_sets > 4:
        raise ValueError(
            f"Direct projection supports at most 4 gene sets (got {n_sets}). "
            "Use project_pca() for higher dimensions."
        )

    mat = scores[score_cols].to_numpy(dtype=np.float64)

    # ---- 2 or 3 gene sets: straightforward multiplicative blend -----------
    if n_sets in (2, 3):
        default = DEFAULT_COLORS_2 if n_sets == 2 else DEFAULT_COLORS_3
        if colors is None:
            colors = default
        if len(colors) != n_sets:
            raise ValueError(
                f"Expected {n_sets} colors for {n_sets} gene sets, got {len(colors)}."
            )
        return _multiplicative_blend(mat, colors)

    # ---- 4 gene sets: hue pair + brightness pair --------------------------
    # Determine pair assignment.
    if pair_order == "infer":
        order = _order_by_variance(mat, score_cols)
        hue_idx = sorted(order[:2])
        brt_idx = sorted(order[2:])
    else:  # "columns"
        hue_idx = [0, 1]
        brt_idx = [2, 3]

    hue_mat = mat[:, hue_idx]
    brt_mat = mat[:, brt_idx]

    # Default hue colours for the 4-gene-set case.
    if colors is None:
        hue_colors = DEFAULT_COLORS_2
    else:
        if len(colors) != 2:
            raise ValueError(
                f"For 4 gene sets, supply exactly 2 colors (for the hue pair). Got {len(colors)}."
            )
        hue_colors = colors

    base_hue = _multiplicative_blend(hue_mat, hue_colors)

    # Brightness modulation: darken by the mean of the brightness pair.
    brt_mean = brt_mat.mean(axis=1, keepdims=True)  # (n_cells, 1)
    final = base_hue * (1.0 - brightness_alpha * brt_mean)

    return np.asarray(np.clip(final, 0.0, 1.0))


def project_pca(
    scores: DataFrame,
    *,
    n_components: int = 3,
) -> NDArray:
    """Map gene set scores to RGB via PCA (SVD).

    Parameters
    ----------
    scores
        DataFrame returned by :func:`score_gene_sets`.
    n_components
        Number of principal components to retain (max 3 for RGB).

    Returns
    -------
    numpy.ndarray
        ``(n_cells, 3)`` RGB array with values in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 2 gene sets are present.
    """
    score_cols = _validate_score_columns(scores)
    if len(score_cols) < 2:
        raise ValueError("At least 2 gene sets are required.")

    mat = scores[score_cols].to_numpy(dtype=np.float64)
    k = min(n_components, 3)
    return _pca_via_svd(mat, k)
