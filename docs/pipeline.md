# Pipeline Guide

multiscoresplot follows a simple 3-step pipeline to go from gene sets to a colored embedding plot.

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

!!! tip "Custom column naming"
All pipeline functions accept `prefix` and `suffix` parameters to customise score
column names. For example, `prefix="msp-"` produces columns like `msp-Dorsal`
instead of `score-Dorsal`. Pass the same `prefix`/`suffix` to downstream functions
(`blend_to_rgb`, `reduce_to_rgb`, `plot_embedding_interactive`) — or use `plot_scores`
which forwards them automatically. `RGBResult` stores these values so interactive
plots auto-detect them from the result object.

### Score Post-Processing

UCell scores are bounded to [0, 1] in theory, but in practice the observed range is
often narrow (e.g., [0.1, 0.6]) because the theoretical extremes are rare. This
underutilizes the color space for visualization. Two optional parameters help:

- **`clip_pct`** — Percentile clipping (winsorization) to reduce the influence of
  outlier cells. A single float clips the upper tail; a tuple `(lo, hi)` clips both.
- **`normalize`** — Per-gene-set min-max rescaling so the scores span the full [0, 1]
  range, maximizing use of the color space.

Processing order: clip first, then normalize (so normalization stretches the clipped
range).

```python
# Clip upper 1% outliers, then stretch to [0, 1]
scores = msp.score_gene_sets(
    adata, gene_sets,
    clip_pct=99,
    normalize=True,
)

# Clip both tails
scores = msp.score_gene_sets(adata, gene_sets, clip_pct=(1, 99))

# Only normalize (no clipping)
scores = msp.score_gene_sets(adata, gene_sets, normalize=True)
```

!!! tip
For most visualization workflows, `clip_pct=99, normalize=True` is a good default.
This removes extreme outliers and stretches the remaining scores to fill the full
color range.

## Step 2 — Map Scores to RGB

Convert gene set scores into per-cell RGB colors. Two options are available depending on
how many gene sets you are visualizing.

### Option A: Blend (2–3 gene sets)

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

### Option B: Reduce (2+ gene sets)

For **any number of gene sets**, dimensionality reduction projects the score matrix into
3 components that become RGB channels.

```python
rgb = msp.reduce_to_rgb(scores, method="pca")  # default
rgb = msp.reduce_to_rgb(scores, method="nmf")
rgb = msp.reduce_to_rgb(scores, method="ica")

# Or pass a callable directly for one-off custom reductions
rgb = msp.reduce_to_rgb(scores, method=my_reducer_fn)
```

#### Choosing a Reduction Method

| Method  | Best for            | Properties                                                                                                                            |
| ------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **PCA** | General use         | Linear, orthogonal components, preserves maximum variance. Components can mix positive and negative loadings.                         |
| **NMF** | Interpretability    | Non-negative components — each RGB channel corresponds to a non-negative combination of gene sets. Often more biologically intuitive. |
| **ICA** | Independent signals | Maximizes statistical independence between components. Useful when gene programs are expected to be independent.                      |

!!! tip
Start with **PCA** for exploration. Switch to **NMF** if you want components that are
easier to interpret biologically. Use **ICA** when you have prior reason to believe
the gene programs are driven by separate, non-overlapping regulatory mechanisms.

#### PCA — Principal Component Analysis

PCA finds the directions of maximum variance in the score matrix and uses the top 3 as
RGB channels. It is the best default choice because it captures the most information
(the largest differences between cells) in the fewest components.

However, PCA components can have both positive and negative loadings on gene sets. This
means a single RGB channel might represent "high in gene set A _and_ low in gene set B"
at the same time, which can make the color mapping less intuitive to interpret. In
practice, this is rarely a problem for visualization — the overall color patterns still
reveal meaningful structure — but it does mean the legend's R/G/B labels are abstract
axes rather than directly corresponding to specific gene programs.

**Use PCA when:** you want a general-purpose overview and don't need each color channel
to map neatly to a biological concept.

#### NMF — Non-negative Matrix Factorization

NMF decomposes the score matrix into non-negative factors. Because both the loadings and
the coefficients are constrained to be ≥ 0, each RGB channel can only be a _positive_
combination of gene sets — it can never represent "high A and low B" in the same
component. This makes the components more naturally interpretable: each color channel
tends to capture a distinct group of co-active gene programs.

For example, if you have gene sets for quiescent stem cells, activated stem cells,
transit-amplifying progenitors, and neuroblasts, NMF might produce one component that
loads mainly on the stem cell sets (appearing as red), another on progenitors (green),
and a third on neuroblasts (blue). This additive, parts-based decomposition often aligns
well with how biologists think about cell states.

The trade-off is that NMF may capture less total variance than PCA, since the
non-negativity constraint limits the solution space.

**Use NMF when:** you want each color channel to represent a positive mixture of gene
programs, making the plot easier to interpret biologically.

#### ICA — Independent Component Analysis

ICA looks for components that are statistically _independent_ — meaning knowing the value
of one component tells you nothing about the others. This is a stronger requirement than
PCA's orthogonality (uncorrelated), which only rules out linear relationships.

In biological terms, this is useful when you believe each gene program is driven by a
separate regulatory mechanism that operates independently of the others. For example,
cell cycle activity and differentiation state are often controlled by distinct pathways.
ICA would try to separate these into different RGB channels, even if they are somewhat
correlated across cells, by finding the most "non-Gaussian" (i.e., structured and
signal-like) directions in the data.

The downside is that ICA can be sensitive to the number of gene sets and may not converge
well when the underlying signals are not truly independent. It also does not rank
components by variance like PCA does, so the R/G/B assignment is less predictable.

**Use ICA when:** you expect your gene programs to reflect distinct, independently
regulated biological processes and want the color channels to separate them as cleanly
as possible.

!!! warning "Interpreting Colors in Reduction Mode"

    1. **Similar colors ≠ similar biology.** Reduction methods (PCA/NMF/ICA) project
       the full score matrix into just 3 dimensions. Only the top 3 axes of variation
       are captured - all other differences between cells are collapsed. Two cells with
       very different gene set score profiles can appear the same color if their
       differences lie along components that were dropped.

    2. **Same gene set score ≠ same gene expression.** UCell computes a rank-based
       aggregate score per gene set. Two cells can achieve the same UCell score by
       highly expressing different subsets of genes within that gene set. The score
       summarizes overall gene set activity, not the identity of which specific genes
       drive it.

    3. **RGB channels are abstract in reduction mode.** Unlike `blend_to_rgb` where
       each color maps directly to a specific gene set, the R, G, and B channels in
       reduction mode represent learned linear combinations of all gene set scores.
       The legend labels (e.g., PC1/PC2/PC3) are statistical axes, not biological
       programs. For more interpretable components, use `method="nmf"` (non-negative
       parts-based decomposition) or `blend_to_rgb` for ≤ 3 gene sets.

### Blend vs. Reduce — When to Use Which

|                      | `blend_to_rgb`                                      | `reduce_to_rgb`                                         |
| -------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| **Gene sets**        | 2–3 only                                            | 2 or more                                               |
| **Color mapping**    | Direct: each gene set has its own color             | Learned: RGB channels are linear combinations of scores |
| **Interpretability** | Immediate — colors correspond directly to gene sets | Requires the legend to interpret RGB channels           |
| **Best for**         | Focused comparisons of 2–3 programs                 | Exploratory analysis of many programs simultaneously    |

## Step 3 — Plot Embedding

Scatter plot of embedding coordinates (UMAP, PCA, etc.) colored by the RGB values,
with an integrated color-space legend.

Both `blend_to_rgb` and `reduce_to_rgb` return an `RGBResult` object that carries
the RGB array together with metadata (method, gene set names, colors). The plotting
functions detect this metadata automatically, so you don't need to repeat `method=`
and `gene_set_names=`.

!!! note "Full obsm key"
The `basis` parameter now takes the **full obsm key** (e.g., `"X_umap"`,
`"umap_consensus"`), not a short name. This lets you use any obsm key, not
just those prefixed with `X_`.

```python
# Basic usage — method and gene_set_names auto-detected from RGBResult
msp.plot_embedding(adata, rgb, basis="X_umap")

# With customization
ax = msp.plot_embedding(
    adata, rgb,
    basis="X_umap",
    legend=True,              # show color legend (default)
    legend_style="inset",     # "inset" or "side"
    legend_loc="lower right", # legend position
    legend_size=0.30,         # legend size (fraction of plot)
    legend_resolution=128,    # legend image resolution
    point_size=3,
    alpha=0.8,
    figsize=(6, 6),
    dpi=100,                  # figure resolution
    title="SVZ lineage",
    show=False,               # return axes instead of displaying
)

# You can still override metadata from RGBResult
msp.plot_embedding(
    adata, rgb,
    basis="X_umap",
    method="pca",             # override RGBResult.method
    gene_set_names=["A", "B", "C", "D"],  # override RGBResult.gene_set_names
)
```

If you pass a raw `(n_cells, 3)` numpy array instead of an `RGBResult`, you must
provide `method=` explicitly when `legend=True`.

### Interactive Plot (requires Plotly)

WebGL-accelerated scatter plot with hover info showing gene set scores, RGB channel
values, and custom metadata. Uses the same `RGBResult` metadata as `plot_embedding`.

```python
msp.plot_embedding_interactive(
    adata, rgb,
    basis="X_umap",
    scores=scores,                         # optional: explicit score DataFrame
    hover_columns=["n_counts", "Dcx"],     # obs columns or gene names
    legend=True,
    legend_loc="lower right",
    point_size=2,
    figsize=(6.5, 6.0),                    # figure size in inches
    dpi=100,                               # pixels = figsize * dpi
)
```

!!! tip "Hover over genes"
`hover_columns` accepts both `adata.obs` column names **and** gene names from
`adata.var_names`. Gene names display the expression value from `adata.X`.

!!! note
Interactive plots require the `plotly` extra: `pip install 'multiscoresplot[interactive]'`

## Optional — Standalone Legend

The plotting functions above already include an integrated legend. If you need to render
the color-space legend separately (e.g., for a custom figure layout), you can use
`render_legend` directly on any matplotlib axes.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Direct mode (2-set square or 3-set triangle)
msp.render_legend(ax, "direct", gene_set_names=["A", "B", "C"])

# Reduction mode (RGB triangle with component labels)
msp.render_legend(ax, "pca")
msp.render_legend(ax, "nmf", component_labels=["NMF1", "NMF2", "NMF3"])
```

## One-Step Convenience Function

For quick exploration, `plot_scores` wraps the entire 3-step pipeline in a single call.
It auto-selects `blend_to_rgb` for ≤ 3 gene sets and `reduce_to_rgb(method="pca")` for
more.

```python
# All-in-one: score → map to RGB → plot
scores, rgb, ax = msp.plot_scores(adata, gene_sets, basis="X_umap")

# With post-processing and custom method
scores, rgb, ax = msp.plot_scores(
    adata, gene_sets,
    clip_pct=99,
    normalize=True,
    method="nmf",
    basis="X_umap",
    show=False,
)

# Force blend even with 2 sets (default already does this, but explicit)
scores, rgb, ax = msp.plot_scores(
    adata, gene_sets_2,
    method="blend",
    basis="X_umap",
)
```

The return value is a `(scores, rgb, plot_result)` tuple, so you can reuse
the intermediate outputs for further analysis.
