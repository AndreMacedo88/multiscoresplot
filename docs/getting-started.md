# Getting Started

## Installation

Install from PyPI:

```bash
pip install multiscoresplot
```

For interactive Plotly plots:

```bash
pip install 'multiscoresplot[interactive]'
```

## Prerequisites

multiscoresplot expects an [AnnData](https://anndata.readthedocs.io/) object with:

- A precomputed dimensionality reduction embedding (e.g., UMAP, PCA, or Scanorama) stored in `adata.obsm`
- Gene expression data (raw or normalized) accessible for scoring

## Minimal Working Example

```python
import multiscoresplot as msp

# 1. Define your gene sets
gene_sets = {
    "Stem":       ["Sox2", "Pax6", "Nes"],
    "Neuronal":   ["Dcx", "Tubb3", "Neurod1"],
    "Astrocytic": ["Gfap", "Aqp4", "Aldh1l1"],
}

# 2. Score gene sets per cell (stores in adata.obs as score-<name>)
scores = msp.score_gene_sets(adata, gene_sets, inplace=True)

# 3. Map scores to RGB — choose one:
#    Blend (2–3 gene sets only)
rgb = msp.blend_to_rgb(scores)
#    Or reduce via dimensionality reduction (any number of gene sets)
rgb = msp.reduce_to_rgb(scores, method="pca")

# 4. Plot on a UMAP embedding — method & labels auto-detected from RGBResult
msp.plot_embedding(adata, rgb, basis="X_umap")
```

!!! tip "Which color mapping to use?"
    - **`blend_to_rgb`** — Best for 2–3 gene sets. Uses intuitive multiplicative blending from white, where each gene set darkens toward its assigned color.
    - **`reduce_to_rgb`** — Works for any number of gene sets (2+). Uses PCA, NMF, or ICA to project scores into a 3D RGB space.

    See the [Pipeline Guide](pipeline.md) for a detailed comparison.

## Next Steps

- [Pipeline Guide](pipeline.md) — Understand each step and when to use each method
- [API Reference](api/index.md) — Full function signatures and parameters
- [Examples](examples.md) — Custom reducers, plot customization, and more
