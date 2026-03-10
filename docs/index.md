# multiscoresplot

**Multi-dimensional gene set scoring and visualization for single-cell transcriptomics.**

Color dimensionality reduction plots (UMAP, PCA, etc.) using a multi-dimensional
color space derived from gene set scores — visualize the activity of multiple gene
programs simultaneously in a single plot.

## Key Features

- **Score** gene sets per cell using [UCell](https://github.com/Cem-Gulec/pyUCell)
- **Blend** 2–3 gene sets to RGB via multiplicative blending
- **Reduce** any number of gene sets to RGB via PCA / NMF / ICA
- **Plot** static matplotlib or interactive Plotly scatter plots
- **Extend** with custom dimensionality reduction methods

## Quick Example

```python
import multiscoresplot as msp

# Define gene sets of interest
gene_sets = {
    "qNSCs": ["Id3", "Aldoc", "Slc1a3"],
    "aNSCs": ["Egfr", "Ascl1", "Mki67"],
    "TAP":   ["Dll1", "Dcx", "Neurod1"],
    "NB":    ["Dcx", "Sox11", "Tubb3"],
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

[Get Started](getting-started.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
