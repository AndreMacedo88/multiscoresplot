"""Color space construction and cell projection (pipeline steps 2-3).

Provides two projection strategies:

* **Direct** (`blend_to_rgb`): multiplicative blending from white using
  explicit base colors.  Supports 2-3 gene sets.
* **Reduction** (`reduce_to_rgb`): dimensionality reduction (PCA, NMF, ICA,
  or custom) to 3 color channels.  Works for any number of gene sets (≥ 2).

Legacy names ``project_direct`` and ``project_pca`` are kept as thin
deprecation wrappers.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np

from multiscoresplot._scoring import SCORE_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pandas import DataFrame

__all__ = [
    "RGBResult",
    "blend_to_rgb",
    "get_component_labels",
    "project_direct",
    "project_pca",
    "reduce_to_rgb",
    "register_reducer",
]


# ---- RGBResult data class ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RGBResult:
    """Container for RGB mapping results with metadata.

    Returned by :func:`blend_to_rgb` and :func:`reduce_to_rgb`.  Carries
    the ``(n_cells, 3)`` RGB array together with metadata that downstream
    plotting functions can use automatically.

    The object supports the numpy array protocol, so ``np.asarray(result)``
    returns the underlying RGB array and indexing/slicing works transparently.

    Parameters
    ----------
    rgb
        ``(n_cells, 3)`` RGB array with values in [0, 1].
    method
        ``"direct"`` for multiplicative blend, or the reduction method name
        (``"pca"``, ``"nmf"``, ``"ica"``, ...).
    gene_set_names
        Human-readable gene set labels, derived from score column names.
    colors
        Base colours used for blending (only set for ``blend_to_rgb``).
    """

    rgb: NDArray
    method: str
    gene_set_names: list[str]
    colors: list[tuple[float, float, float]] | None = None

    # numpy array protocol ---------------------------------------------------

    def __array__(self, dtype: object = None, copy: object = None) -> NDArray:
        if dtype is not None:
            return np.array(self.rgb, dtype=dtype)  # type: ignore[no-any-return,call-overload]
        return np.asarray(self.rgb)

    def __getitem__(self, key: object) -> object:
        return self.rgb[key]  # type: ignore[call-overload]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rgb.shape  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        return self.rgb.ndim  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.rgb)

    # Comparison operators (delegate to underlying array) --------------------

    def __ge__(self, other: object) -> NDArray:
        return self.rgb >= other  # type: ignore[return-value]

    def __le__(self, other: object) -> NDArray:
        return self.rgb <= other  # type: ignore[return-value]

    def __gt__(self, other: object) -> NDArray:
        return self.rgb > other  # type: ignore[return-value]

    def __lt__(self, other: object) -> NDArray:
        return self.rgb < other  # type: ignore[return-value]

    def __eq__(self, other: object) -> NDArray:  # type: ignore[override]
        return self.rgb == other  # type: ignore[no-any-return,return-value]

    def __ne__(self, other: object) -> NDArray:  # type: ignore[override]
        return self.rgb != other  # type: ignore[no-any-return,return-value]


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


def _minmax_normalize(X: NDArray, n_target: int = 3) -> NDArray:
    """Min-max normalize each column to [0, 1], zero-pad to *n_target* columns."""
    k = X.shape[1]
    for j in range(k):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo > 0:
            X[:, j] = (col - lo) / (hi - lo)
        else:
            X[:, j] = 0.0

    if k < n_target:
        pad = np.zeros((X.shape[0], n_target - k), dtype=np.float64)
        X = np.hstack([X, pad])

    return X


# ---- reducer registry ------------------------------------------------------

ReducerFn = type(lambda: None)  # placeholder for type alias

_REDUCERS: dict[str, Callable[..., NDArray]] = {}
_COMPONENT_PREFIXES: dict[str, str] = {}


def register_reducer(
    name: str,
    fn: Callable[..., NDArray],
    *,
    component_prefix: str | None = None,
) -> None:
    """Register a dimensionality reduction method for use with ``reduce_to_rgb``.

    Parameters
    ----------
    name
        Short identifier (e.g. ``"pca"``, ``"nmf"``).
    fn
        Callable with signature ``(X, n_components, **kwargs) -> NDArray``
        returning an ``(n_cells, 3)`` array with values in [0, 1].
    component_prefix
        Label prefix for legend axes (e.g. ``"PC"`` → PC1, PC2, PC3).
    """
    _REDUCERS[name] = fn
    if component_prefix is not None:
        _COMPONENT_PREFIXES[name] = component_prefix


def get_component_labels(method: str) -> list[str]:
    """Return ``["<prefix>1", "<prefix>2", "<prefix>3"]`` for a registered method."""
    prefix = _COMPONENT_PREFIXES.get(method, "C")
    return [f"{prefix}{i + 1}" for i in range(3)]


# ---- built-in reducer implementations --------------------------------------


def _reduce_pca(X: NDArray, n_components: int, **kwargs: object) -> NDArray:
    """PCA via numpy SVD."""
    mean = X.mean(axis=0)
    X_centered = X - mean

    if np.allclose(X_centered, 0.0):
        return np.zeros((X.shape[0], 3), dtype=np.float64)

    U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    k = min(n_components, U.shape[1])
    pc_scores = U[:, :k] * S[:k]
    return _minmax_normalize(pc_scores, n_target=3)


def _reduce_nmf(X: NDArray, n_components: int, **kwargs: object) -> NDArray:
    """NMF via scikit-learn."""
    from sklearn.decomposition import NMF

    if np.allclose(X, X.mean(axis=0)):
        return np.zeros((X.shape[0], 3), dtype=np.float64)

    k = min(n_components, X.shape[1])
    defaults: dict[str, object] = {"init": "nndsvda", "max_iter": 300}
    defaults.update(kwargs)
    model = NMF(n_components=k, **defaults)  # type: ignore[arg-type]
    W = model.fit_transform(X)
    return _minmax_normalize(W, n_target=3)


def _reduce_ica(X: NDArray, n_components: int, **kwargs: object) -> NDArray:
    """ICA via scikit-learn FastICA."""
    from sklearn.decomposition import FastICA

    if np.allclose(X, X.mean(axis=0)):
        return np.zeros((X.shape[0], 3), dtype=np.float64)

    k = min(n_components, X.shape[1])
    defaults: dict[str, object] = {"max_iter": 300, "tol": 1e-4}
    defaults.update(kwargs)
    model = FastICA(n_components=k, **defaults)  # type: ignore[arg-type]
    S = model.fit_transform(X)
    return _minmax_normalize(S, n_target=3)


# Register built-in reducers
register_reducer("pca", _reduce_pca, component_prefix="PC")
register_reducer("nmf", _reduce_nmf, component_prefix="NMF")
register_reducer("ica", _reduce_ica, component_prefix="IC")


# ---- public API -------------------------------------------------------------


def blend_to_rgb(
    scores: DataFrame,
    *,
    colors: list[tuple[float, float, float]] | None = None,
) -> RGBResult:
    """Map gene set scores to RGB via multiplicative blending from white.

    Parameters
    ----------
    scores
        DataFrame returned by :func:`score_gene_sets`.  Only columns whose
        names start with ``score-`` are used.
    colors
        One ``(R, G, B)`` tuple per gene set.  If *None*, defaults are chosen
        based on the number of gene sets (2 → blue/red, 3 → RGB).

    Returns
    -------
    RGBResult
        Contains the ``(n_cells, 3)`` RGB array with values in [0, 1],
        together with metadata (``method="direct"``, gene set names, colours).

    Raises
    ------
    ValueError
        If fewer than 2 or more than 3 gene sets are present, or if the
        number of supplied colours does not match expectations.
    """
    score_cols = _validate_score_columns(scores)
    n_sets = len(score_cols)

    if n_sets < 2:
        raise ValueError("At least 2 gene sets are required.")
    if n_sets > 3:
        raise ValueError(
            f"Direct projection supports at most 3 gene sets (got {n_sets}). "
            "Use reduce_to_rgb() for higher dimensions."
        )

    mat = scores[score_cols].to_numpy(dtype=np.float64)

    default = DEFAULT_COLORS_2 if n_sets == 2 else DEFAULT_COLORS_3
    if colors is None:
        colors = default
    if len(colors) != n_sets:
        raise ValueError(f"Expected {n_sets} colors for {n_sets} gene sets, got {len(colors)}.")

    prefix_len = len(SCORE_PREFIX)
    gene_set_names = [c[prefix_len:] for c in score_cols]

    return RGBResult(
        rgb=_multiplicative_blend(mat, colors),
        method="direct",
        gene_set_names=gene_set_names,
        colors=colors,
    )


def reduce_to_rgb(
    scores: DataFrame,
    *,
    method: str = "pca",
    n_components: int = 3,
    **kwargs: object,
) -> RGBResult:
    """Map gene set scores to RGB via dimensionality reduction.

    Parameters
    ----------
    scores
        DataFrame returned by :func:`score_gene_sets`.
    method
        Reduction method: ``"pca"`` (default), ``"nmf"``, ``"ica"``, or any
        method registered via :func:`register_reducer`.
    n_components
        Number of components to retain (max 3 for RGB).
    **kwargs
        Extra keyword arguments forwarded to the reducer function.

    Returns
    -------
    RGBResult
        Contains the ``(n_cells, 3)`` RGB array with values in [0, 1],
        together with metadata (method name, gene set names).

    Raises
    ------
    ValueError
        If fewer than 2 gene sets are present or *method* is unknown.
    """
    if method not in _REDUCERS:
        available = ", ".join(sorted(_REDUCERS))
        raise ValueError(f"Unknown reduction method '{method}'. Available: {available}.")

    score_cols = _validate_score_columns(scores)
    if len(score_cols) < 2:
        raise ValueError("At least 2 gene sets are required.")

    mat = scores[score_cols].to_numpy(dtype=np.float64)
    k = min(n_components, 3)

    prefix_len = len(SCORE_PREFIX)
    gene_set_names = [c[prefix_len:] for c in score_cols]

    return RGBResult(
        rgb=_REDUCERS[method](mat, k, **kwargs),
        method=method,
        gene_set_names=gene_set_names,
    )


# ---- deprecated wrappers ---------------------------------------------------


def project_direct(
    scores: DataFrame,
    *,
    colors: list[tuple[float, float, float]] | None = None,
) -> RGBResult:
    """Deprecated: use :func:`blend_to_rgb` instead."""
    warnings.warn(
        "project_direct() is deprecated, use blend_to_rgb() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return blend_to_rgb(scores, colors=colors)


def project_pca(
    scores: DataFrame,
    *,
    n_components: int = 3,
) -> RGBResult:
    """Deprecated: use :func:`reduce_to_rgb` instead."""
    warnings.warn(
        "project_pca() is deprecated, use reduce_to_rgb() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reduce_to_rgb(scores, method="pca", n_components=n_components)
