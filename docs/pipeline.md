# Pipeline Guide

multiscoresplot follows a 5-step pipeline to go from gene sets to a colored embedding plot.

## Step 1 — Score Gene Sets

Calculate per-cell gene set scores using [pyUCell](https://github.com/Cem-Gulec/pyUCell).
Scores are stored in `adata.obs` as `score-<name>` columns with values in [0, 1].

```python
scores = msp.score_gene_sets(adata, gene_sets, inplace=True)

# With tuning parameters
scores = msp.score_gene_sets(
    adata, gene_sets,
    max_rank=1500,    # rank cap (tune to median genes per cell)
    chunk_size=1000,  # cells per batch
    n_jobs=-1,        # parallelism (-1 = all cores)
    inplace=False,    # don't store in adata.obs
)
```

!!! note
    `score_gene_sets` wraps pyUCell's ranking-based scoring. The `max_rank` parameter
    controls how many top-ranked genes per cell are considered — set it close to the
    median number of detected genes per cell for best results.

## Step 2 — Blend to RGB (2–3 Gene Sets)

For **2–3 gene sets**, multiplicative blending maps scores directly to colors. Starting from
white, each gene set darkens the color toward its base hue proportionally to the score.

```python
# Default colors (2 sets: blue/red, 3 sets: R/G/B)
rgb = msp.blend_to_rgb(scores)

# Custom colors
rgb = msp.blend_to_rgb(scores, colors=[(1, 0, 0), (0, 0.5, 1)])
```

**How it works:** Each cell starts as white `(1, 1, 1)`. For each gene set, the cell's
color is multiplied element-wise by a blend between white and the gene set's base color,
weighted by the score. High scores pull the color toward the base hue; low scores leave
it near white. Cells with multiple high scores produce mixed colors.

## Step 3 — Reduce to RGB (2+ Gene Sets)

For **any number of gene sets**, dimensionality reduction projects the score matrix into
3 components that become RGB channels.

```python
rgb = msp.reduce_to_rgb(scores, method="pca")  # default
rgb = msp.reduce_to_rgb(scores, method="nmf")
rgb = msp.reduce_to_rgb(scores, method="ica")
```

### Choosing a Reduction Method

| Method | Best for | Properties |
|--------|----------|------------|
| **PCA** | General use | Linear, orthogonal components, preserves maximum variance. Components can mix positive and negative loadings. |
| **NMF** | Interpretability | Non-negative components — each RGB channel corresponds to a non-negative combination of gene sets. Often more biologically intuitive. |
| **ICA** | Independent signals | Maximizes statistical independence between components. Useful when gene programs are expected to be independent. |

!!! tip
    Start with **PCA** for exploration. Switch to **NMF** if you want components that are
    easier to interpret biologically (since NMF coefficients are always non-negative).
    Use **ICA** when you suspect the gene programs represent statistically independent signals.

## Step 4 — Plot Embedding

Scatter plot of embedding coordinates (UMAP, PCA, etc.) colored by the RGB values,
with an integrated color-space legend.

```python
# Basic usage
msp.plot_embedding(
    adata, rgb,
    basis="umap",
    method="pca",
    gene_set_names=["qNSCs", "aNSCs", "TAP", "NB"],
)

# With customization
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

### Step 4b — Interactive Plot (requires Plotly)

WebGL-accelerated scatter plot with hover info showing gene set scores, RGB channel
values, and custom metadata.

```python
msp.plot_embedding_interactive(
    adata, rgb,
    basis="umap",
    scores=scores,
    method="nmf",
    gene_set_names=["qNSCs", "aNSCs", "TAP", "NB"],
    hover_columns=["n_counts", "cell_type"],
    legend=True,
    legend_loc="lower right",
    point_size=2,
    width=600,
    height=500,
)
```

!!! note
    Interactive plots require the `plotly` extra: `pip install 'multiscoresplot[interactive]'`

## Step 5 — Standalone Legend

Render the color-space legend independently on any matplotlib axes.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Direct mode (2-set square or 3-set triangle)
msp.render_legend(ax, "direct", gene_set_names=["A", "B", "C"])

# Reduction mode (RGB triangle with component labels)
msp.render_legend(ax, "pca")
msp.render_legend(ax, "nmf", component_labels=["NMF1", "NMF2", "NMF3"])
```

## Blend vs. Reduce — When to Use Which

| | `blend_to_rgb` | `reduce_to_rgb` |
|---|---|---|
| **Gene sets** | 2–3 only | 2 or more |
| **Color mapping** | Direct: each gene set has its own color | Learned: RGB channels are linear combinations of scores |
| **Interpretability** | Immediate — colors correspond directly to gene sets | Requires the legend to interpret RGB channels |
| **Best for** | Focused comparisons of 2–3 programs | Exploratory analysis of many programs simultaneously |
