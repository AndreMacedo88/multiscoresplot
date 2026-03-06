"""multiscoresplot -- multi-dimensional gene set scoring visualization."""

from multiscoresplot._colorspace import (
    blend_to_rgb,
    get_component_labels,
    project_direct,
    project_pca,
    reduce_to_rgb,
    register_reducer,
)
from multiscoresplot._interactive import plot_embedding_interactive
from multiscoresplot._legend import render_legend
from multiscoresplot._plotting import plot_embedding
from multiscoresplot._scoring import score_gene_sets

__all__ = [
    "blend_to_rgb",
    "get_component_labels",
    "plot_embedding",
    "plot_embedding_interactive",
    "project_direct",
    "project_pca",
    "reduce_to_rgb",
    "register_reducer",
    "render_legend",
    "score_gene_sets",
]
__version__ = "0.1.0"
