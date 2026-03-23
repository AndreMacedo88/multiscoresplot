# API Reference

## Overview

| Function / Class | Description | Pipeline Step |
|------------------|-------------|---------------|
| [`score_gene_sets`](scoring.md) | Score gene sets per cell via pyUCell | Step 1 |
| [`blend_to_rgb`](colorspace.md#multiscoresplot.blend_to_rgb) | Multiplicative blend to RGB (2–3 sets) | Step 2 |
| [`reduce_to_rgb`](colorspace.md#multiscoresplot.reduce_to_rgb) | Dimensionality reduction to RGB (2+ sets) | Step 2 |
| [`RGBResult`](colorspace.md#multiscoresplot.RGBResult) | Return type of `blend_to_rgb` / `reduce_to_rgb` with metadata | Step 2 |
| [`plot_scores`](pipeline.md) | One-step convenience: score → RGB → plot | Steps 1–3 |
| [`plot_embedding`](plotting.md) | Static matplotlib scatter plot | Step 3 |
| [`plot_embedding_interactive`](interactive.md) | Interactive Plotly scatter plot | Step 3 |
| [`render_legend`](legend.md) | Draw color-space legend on axes | Optional |
| [`register_reducer`](colorspace.md#multiscoresplot.register_reducer) | Register a custom reduction method | Extensibility |
| [`get_component_labels`](colorspace.md#multiscoresplot.get_component_labels) | Get axis labels for a reduction method | Utility |

## Module Layout

All public functions are available directly from the top-level `multiscoresplot` namespace:

```python
import multiscoresplot as msp

msp.score_gene_sets(...)
msp.blend_to_rgb(...)
msp.reduce_to_rgb(...)
msp.plot_embedding(...)
```
