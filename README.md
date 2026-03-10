# multiscoresplot

[![CI](https://github.com/AndreMacedo88/multiscoresplot/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreMacedo88/multiscoresplot/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://AndreMacedo88.github.io/multiscoresplot/)
[![PyPI](https://img.shields.io/pypi/v/multiscoresplot?cacheSeconds=3600)](https://pypi.org/project/multiscoresplot/)
[![Python](https://img.shields.io/pypi/pyversions/multiscoresplot?cacheSeconds=3600)](https://pypi.org/project/multiscoresplot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multi-dimensional gene set scoring and visualization for single-cell transcriptomics.**

Color dimensionality reduction plots (UMAP, PCA, etc.) using a multi-dimensional color space derived from gene set scores — so you can visualize the activity of multiple gene programs simultaneously in a single plot.

## Installation

```bash
pip install multiscoresplot
```

For interactive Plotly plots:

```bash
pip install 'multiscoresplot[interactive]'
```

## Quick Start

```python
import multiscoresplot as msp

# Define gene sets of interest
gene_sets = {
    "qNSCs": ["Id3", "Aldoc", "Slc1a3", ...],
    "aNSCs": ["Egfr", "Ascl1", "Mki67", ...],
    "TAP":   ["Dll1", "Dcx", "Neurod1", ...],
    "NB":    ["Dcx", "Sox11", "Tubb3", ...],
}

# 1. Score gene sets per cell
scores = msp.score_gene_sets(adata, gene_sets, inplace=True)

# 2. Map scores to RGB colors
rgb = msp.reduce_to_rgb(scores, method="pca")

# 3. Plot
msp.plot_embedding(
    adata, rgb,
    basis="umap",
    method="pca",
    gene_set_names=list(gene_sets.keys()),
)
```

## Pipeline

multiscoresplot follows a 3-step pipeline:

### Step 1 — Score gene sets

Calculate per-cell gene set scores using [pyUCell](https://github.com/Cem-Gulec/pyUCell). Scores are stored in `adata.obs` as `score-<name>` columns with values in [0, 1].

```python
scores = msp.score_gene_sets(adata, gene_sets, inplace=True)

# Options
scores = msp.score_gene_sets(
    adata, gene_sets,
    max_rank=1500,    # rank cap (tune to median genes per cell)
    chunk_size=1000,  # cells per batch
    n_jobs=-1,        # parallelism (-1 = all cores)
    inplace=False,    # don't store in adata.obs
)
```

### Step 2 — Map scores to RGB

Convert gene set scores into per-cell RGB colors. Two options depending on how many gene sets you want to visualize:

**Blend (2–3 gene sets)** — multiplicative blending from white, where each gene set darkens toward its base color proportional to the score.

```python
# Default colors (2 sets: blue/red, 3 sets: R/G/B)
rgb = msp.blend_to_rgb(scores)

# Custom colors
rgb = msp.blend_to_rgb(scores, colors=[(1, 0, 0), (0, 0.5, 1)])
```

**Reduce (2+ gene sets)** — dimensionality reduction maps scores to 3 RGB channels. Built-in methods: PCA, NMF, and ICA.

```python
rgb = msp.reduce_to_rgb(scores, method="pca")  # default
rgb = msp.reduce_to_rgb(scores, method="nmf")
rgb = msp.reduce_to_rgb(scores, method="ica")
```

### Step 3 — Plot embedding

Scatter plot of embedding coordinates colored by RGB values, with an integrated color-space legend.

```python
# Static matplotlib plot
msp.plot_embedding(
    adata, rgb,
    basis="umap",
    method="pca",
    gene_set_names=["qNSCs", "aNSCs", "TAP", "NB"],
)

# Options
ax = msp.plot_embedding(
    adata, rgb,
    basis="umap",
    method="pca",
    gene_set_names=["qNSCs", "aNSCs", "TAP", "NB"],
    legend=True,              # show color legend (default)
    legend_style="inset",     # "inset" or "side"
    legend_loc="lower right", # legend position
    point_size=3,
    alpha=0.8,
    figsize=(6, 6),
    title="SVZ lineage",
    show=False,               # return axes instead of displaying
)
```

**Interactive plot (requires plotly)** — WebGL-accelerated scatter plot with hover info showing gene set scores, RGB channel values, and custom metadata.

```python
msp.plot_embedding_interactive(
    adata, rgb,
    basis="umap",
    scores=scores,
    method="nmf",
    gene_set_names=["qNSCs", "aNSCs", "TAP", "NB"],
    hover_columns=["n_counts", "cell_type"],  # extra adata.obs columns
    legend=True,
    legend_loc="lower right",
    point_size=2,
    width=600,
    height=500,
)
```

### Optional — Standalone legend

The plotting functions above include an integrated legend by default. If you need to render the legend separately (e.g., for a custom figure layout):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Direct mode (2-set square or 3-set triangle)
msp.render_legend(ax, "direct", gene_set_names=["A", "B", "C"])

# Reduction mode (RGB triangle with component labels)
msp.render_legend(ax, "pca")
msp.render_legend(ax, "nmf", component_labels=["NMF1", "NMF2", "NMF3"])
```

## Extensibility — Custom reducers

Register your own dimensionality reduction method:

```python
def my_umap_reducer(X, n_components, **kwargs):
    """X: (n_cells, n_gene_sets), returns (n_cells, 3) in [0, 1]."""
    import umap
    embedding = umap.UMAP(n_components=n_components, **kwargs).fit_transform(X)
    # min-max normalize each column to [0, 1]
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            embedding[:, j] = (col - lo) / (hi - lo)
    return embedding

msp.register_reducer("umap", my_umap_reducer, component_prefix="UMAP")

# Now use it like any built-in method
rgb = msp.reduce_to_rgb(scores, method="umap")
```

## Documentation

Full documentation is available at **[AndreMacedo88.github.io/multiscoresplot](https://AndreMacedo88.github.io/multiscoresplot/)**, including:

- [Getting Started](https://AndreMacedo88.github.io/multiscoresplot/getting-started/) — installation and quick start
- [Pipeline Guide](https://AndreMacedo88.github.io/multiscoresplot/pipeline/) — detailed pipeline tutorial
- [API Reference](https://AndreMacedo88.github.io/multiscoresplot/api/) — full function signatures and parameters
- [Examples](https://AndreMacedo88.github.io/multiscoresplot/examples/) — custom reducers, plot customization, and more

## API Reference

| Function                                            | Description                               |
| --------------------------------------------------- | ----------------------------------------- |
| `score_gene_sets(adata, gene_sets)`                 | Score gene sets per cell via pyUCell      |
| `blend_to_rgb(scores)`                              | Multiplicative blend to RGB (2–3 sets)    |
| `reduce_to_rgb(scores, method="pca")`               | Dimensionality reduction to RGB (2+ sets) |
| `plot_embedding(adata, rgb, basis=...)`             | Static matplotlib scatter plot            |
| `plot_embedding_interactive(adata, rgb, basis=...)` | Interactive Plotly scatter plot           |
| `render_legend(ax, method)`                         | Draw color-space legend on axes           |
| `register_reducer(name, fn)`                        | Register a custom reduction method        |
| `get_component_labels(method)`                      | Get axis labels for a reduction method    |

## Development

```bash
git clone https://github.com/AndreMacedo88/multiscoresplot.git
cd multiscoresplot

# Install in editable mode with dev dependencies
pip install -e ".[interactive]"
pip install ruff pre-commit pytest pytest-cov mypy

# Run tests
pytest

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Set up pre-commit hooks
pre-commit install
```

## License

[MIT](LICENSE)
