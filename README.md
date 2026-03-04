# multiscoresplot

[![CI](https://github.com/andrecmacedo/multiscoresplot/actions/workflows/ci.yml/badge.svg)](https://github.com/andrecmacedo/multiscoresplot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/multiscoresplot)](https://pypi.org/project/multiscoresplot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-dimensional gene set scoring and visualization for single-cell transcriptomics.

Color dimensionality reduction plots (UMAP, PCA, etc.) using a multi-dimensional color space derived from gene set scores.

## Installation

```bash
pip install multiscoresplot
```

## Quick Start

```python
import multiscoresplot  # more to come!
```

## Pipeline

1. **Score** -- Calculate gene set scores per cell
2. **Color space** -- Build a color space where each axis/vertex maps to a gene set
3. **Project** -- Map each cell into the color space based on its scores
4. **Plot** -- Color dimensionality reduction coordinates using the projected colors
5. **Legend** -- Render a simplex/ternary plot as the colorbar

## Development

```bash
# Install in editable mode with all dev dependencies
pip install -e ".[dev,test,type]"

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
