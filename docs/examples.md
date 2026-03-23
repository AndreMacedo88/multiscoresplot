# Examples

## Custom Reducer — UMAP

Register your own dimensionality reduction method and use it like any built-in:

```python
import multiscoresplot as msp


def umap_reducer(X, n_components, **kwargs):
    """Reduce score matrix to RGB via UMAP.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_gene_sets)
        Score matrix.
    n_components : int
        Number of output components (always 3 for RGB).

    Returns
    -------
    ndarray of shape (n_cells, 3)
        Embedding with values in [0, 1].
    """
    import umap

    embedding = umap.UMAP(n_components=n_components, **kwargs).fit_transform(X)
    # min-max normalize each column to [0, 1]
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            embedding[:, j] = (col - lo) / (hi - lo)
    return embedding


# Register it
msp.register_reducer("umap", umap_reducer, component_prefix="UMAP")

# Use it — method auto-detected from RGBResult
rgb = msp.reduce_to_rgb(scores, method="umap")
msp.plot_embedding(adata, rgb, basis="X_umap")
```

## Inline Callable Reducer

For one-off custom reductions, pass a callable directly to `reduce_to_rgb` instead of
registering it:

```python
import multiscoresplot as msp
import umap


def umap_reducer(X, n_components, **kwargs):
    embedding = umap.UMAP(n_components=n_components, **kwargs).fit_transform(X)
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            embedding[:, j] = (col - lo) / (hi - lo)
    return embedding


rgb = msp.reduce_to_rgb(scores, method=umap_reducer, component_prefix="UMAP")
msp.plot_embedding(adata, rgb, basis="X_umap")
```

This is equivalent to `register_reducer` + `reduce_to_rgb(method="umap")`, but more
convenient when you only need the reducer once.

## Different Embeddings

Plot the same RGB coloring on different embeddings to compare:

```python
import matplotlib.pyplot as plt

scores = msp.score_gene_sets(adata, gene_sets, inplace=True)
rgb = msp.reduce_to_rgb(scores, method="pca")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, basis in zip(axes, ["X_umap", "X_pca", "X_scanorama"]):
    msp.plot_embedding(
        adata, rgb,
        basis=basis,
        ax=ax,
        title=basis.upper(),
        show=False,
    )

plt.tight_layout()
plt.show()
```

## Comparing Reduction Methods

Visualize how PCA, NMF, and ICA produce different colorings:

```python
import matplotlib.pyplot as plt

scores = msp.score_gene_sets(adata, gene_sets, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, method in zip(axes, ["pca", "nmf", "ica"]):
    rgb = msp.reduce_to_rgb(scores, method=method)
    msp.plot_embedding(
        adata, rgb,
        basis="X_umap",
        ax=ax,
        title=method.upper(),
        show=False,
    )

plt.tight_layout()
plt.show()
```

## Interactive Plot with Hover Metadata

Explore individual cells with hover tooltips:

```python
scores = msp.score_gene_sets(adata, gene_sets, inplace=True)
rgb = msp.reduce_to_rgb(scores, method="nmf")

msp.plot_embedding_interactive(
    adata, rgb,
    basis="X_umap",
    scores=scores,
    hover_columns=["n_counts", "cell_type", "Dcx"],  # obs columns or gene names
    point_size=2,
    figsize=(8.0, 6.0),
    dpi=100,
)
```

## Customizing Plot Appearance

```python
ax = msp.plot_embedding(
    adata, rgb,
    basis="X_umap",
    point_size=5,
    alpha=0.6,
    figsize=(8, 8),
    dpi=150,
    title="SVZ Neural Lineage",
    legend=True,
    legend_style="side",       # legend in a separate panel
    legend_loc="upper right",
    legend_size=0.35,          # legend size (fraction of plot)
    show=False,
)

# Further customize the returned axes
ax.set_xlabel("UMAP 1", fontsize=14)
ax.set_ylabel("UMAP 2", fontsize=14)
```

## Direct Blend for 2–3 Gene Sets

When you have exactly 2 or 3 gene sets, direct blending gives the most intuitive
color mapping:

```python
two_sets = {
    "Stem": ["Sox2", "Pax6", "Nes"],
    "Neuronal": ["Dcx", "Tubb3", "Neurod1"],
}

scores = msp.score_gene_sets(adata, two_sets, inplace=True)
rgb = msp.blend_to_rgb(scores)  # blue/red by default

# method="direct" and gene_set_names auto-detected from RGBResult
msp.plot_embedding(adata, rgb, basis="X_umap")
```
