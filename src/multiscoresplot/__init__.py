"""multiscoresplot -- multi-dimensional gene set scoring visualization."""

from multiscoresplot._colorspace import project_direct, project_pca
from multiscoresplot._legend import render_legend
from multiscoresplot._plotting import plot_embedding
from multiscoresplot._scoring import score_gene_sets

__all__ = ["plot_embedding", "project_direct", "project_pca", "render_legend", "score_gene_sets"]
__version__ = "0.1.0"
