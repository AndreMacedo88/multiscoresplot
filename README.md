# Description

This repo will establish a python package that takes as input:
1- an AnnData object (with UMAP, PCA, or other dimensionality reduction method's coordinates calculated) of single-cell counts
2- a dictionary where the values are lists of genes and the keys are geneset names

And does the following:
1- calculates a score for each geneset (using scanpy's geneset scoring methods - or the best gene set scoring method) for each cell
2- produce a color space where each vertice is a geneset score
3- projects each cell to the color space coordinates (each cell has a gene set score, so they fall in one point in the multidimensional color space)
4- uses that projection for each cell to color the plot of the dimensionality reduction (UMAP, PCA, etc.)
5- plots a "simplex plot" of the color space as the colorbar/legend. So, for 3 genesets it plots a ternary plot colored by all the possible values in the color space so that the user knows where that cell falls on the space

## Notes

- plotting code will be independent of scanpy plotting tools, but mimicking scanpy plotting aesthetics, API, and functionality
- gene set scores will optionally be assigned to the .obs of the AnnData object with naming as such: score-<geneset name>; and in this case the Anndata
  is modified in place
- the tool will be placed in PyPI under a totally open license for others to modify

## Tests

- I got some test data which already has been preprocessed and has the UMAP to test this on.
- check_scdata.ipynb has some small test data exploration, and some marker genes that can be used as genesets to test the tool. In particular the variable "marker_genes_svz_lineage_dict_collapsed"
