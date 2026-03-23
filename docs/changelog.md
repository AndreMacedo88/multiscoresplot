# Changelog

## Unreleased

### New features
- **`blend_to_rgb` / `reduce_to_rgb`**: new `prefix` and `suffix` keyword parameters for custom score column naming conventions (e.g., `prefix="msp-"`, `suffix="_v2"`). Defaults match existing `"score-"` behavior.
- **`RGBResult`**: new `prefix` and `suffix` fields so downstream functions auto-detect the naming convention.
- **`plot_embedding_interactive`**: new `prefix` and `suffix` keyword parameters for correct hover auto-extraction with custom column names. Defaults inherit from `RGBResult` when available.
- **`plot_scores`**: new `prefix` and `suffix` keyword parameters forwarded to all pipeline steps (scoring, color mapping, and interactive plotting).
- **`plot_scores`**: new one-step convenience function that wraps the full score → RGB → plot pipeline. Auto-selects `blend_to_rgb` for ≤ 3 gene sets and `reduce_to_rgb(method="pca")` for more.
- **`reduce_to_rgb`**: `method` now accepts a callable with signature `(X, n_components, **kwargs) -> NDArray` for one-off custom reductions. New `component_prefix` parameter overrides legend axis labels.
- **`score_gene_sets`**: emits `UserWarning` listing missing genes per gene set (genes not found in `adata.var_names` are imputed by pyUCell with worst-case rank).
- **`score_gene_sets`**: emits `UserWarning` when `adata.X` contains negative values (e.g., after `sc.pp.scale()`), since UCell is designed for non-negative counts.
- **`score_gene_sets`**: automatically copies read-only `adata.X` arrays to prevent crashes inside pyUCell (works around a pyUCell bug with read-only arrays after `sc.pp.scale()`).
- **`score_gene_sets`**: new `clip_pct` parameter for per-gene-set percentile clipping (winsorization). Accepts a single float for upper-tail clipping or a `(lo, hi)` tuple for both tails.
- **`score_gene_sets`**: new `normalize` parameter for per-gene-set min-max rescaling to [0, 1]. Applied after clipping.

### Documentation
- Added color interpretation caveats for reduction mode in Pipeline Guide.
- Added inline callable example in Examples page.

## 2.0.0

### Breaking changes
- `blend_to_rgb` and `reduce_to_rgb` now return `RGBResult` (carries RGB array + metadata). The object supports numpy array protocol, so `np.asarray(result)`, indexing, and comparisons still work.
- `plot_embedding` and `plot_embedding_interactive`: `basis=` now takes the **full obsm key** (e.g. `"X_umap"`, `"umap_consensus"`). The old short form (`basis="umap"`) still works but emits a `DeprecationWarning`.
- `plot_embedding`: `legend=True` (default) now **requires** a known method. Previously it silently skipped the legend when `method=None`; now it raises `ValueError`. Pass `legend=False` or provide `method=`, or use an `RGBResult`.
- `plot_embedding_interactive`: `width`/`height` replaced by `figsize` (inches) + `dpi`. Pixel dimensions = `figsize * dpi`.

### New features
- **`RGBResult`**: new dataclass returned by `blend_to_rgb` / `reduce_to_rgb`. Carries `method`, `gene_set_names`, and `colors` metadata that plotting functions auto-detect.
- **`plot_embedding`**: new params `legend_size`, `legend_resolution`, `dpi`.
- **`plot_embedding_interactive`**: new param `legend_kwargs`. `figsize`/`dpi` replace `width`/`height`.
- **Both plotting functions**: consistent `legend_size`, `legend_resolution`, and `legend_kwargs` params.
- **`hover_columns`** now falls back to `adata.var_names` for gene expression values (sparse matrices supported).
- **`gene_set_names`** behavior is now consistent between static and interactive plots.

## 1.0.3

- Fix badge display in README

## 1.0.2

- Fix legend not being plotted in interactive "direct" methods
- Fix CI badge path in README

## 1.0.1

- Initial stable release
- 5-step pipeline: score, blend, reduce, plot, legend
- Built-in reducers: PCA, NMF, ICA
- Pluggable reducer registry
- Static matplotlib and interactive Plotly plotting
- Color-space legends for direct and reduction modes
