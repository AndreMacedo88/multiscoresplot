# Changelog

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
