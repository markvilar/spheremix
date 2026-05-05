# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
uv sync --all-extras --dev
```

**Build:**
```bash
uv build
```

**Test:**
```bash
uv run pytest tests                                      # all tests
uv run pytest tests/distributions/test_spherical_cauchy.py  # single file
uv run pytest tests/path/to/test.py::test_function_name     # single test
```

**Lint / Format:**
```bash
uv run ruff check .
uv run ruff format .
```

## Architecture

**Spheremix** is a Python library for spherical probability distributions and mixture models on unit hyperspheres.

### Package layout (`src/spheremix/`)

| Module | Purpose |
|---|---|
| `distributions/` | Probability distributions on unit spheres — `SphericalCauchyDistribution` (location `mu`, concentration `rho`), `PKBDistribution` (Projected Kent-Bingham) |
| `mixtures/` | Mixture models — `SphericalCauchyMixture` with EM algorithm support, tracking component weights, means, and concentrations |
| `cluster/` | `SphericalKMeans` — scikit-learn-compatible K-means for spherical data (Dhillon & Modha 2001) |
| `algorithm/` | Core algorithm skeletons — `mixture_em.py` (EM for mixture models, not yet fully implemented) |
| `utils/` | `generators.py` — orthogonal unit vector generation via QR decomposition |

## Skills

When developing this library, always invoke these skills from the `python-library-complete` plugin:
- `/python-library-complete:api-design`
- `/python-library-complete:code-quality`
- `/python-library-complete:cli-development`
- `/python-library-complete:performance`

Also invoke the following skills when relevant to the work at hand:
- `/python-library-complete:documentation` — when writing docstrings or user guides
- `/python-library-complete:testing-strategy` — when writing tests
- `/python-library-complete:packaging` — when updating packaging or publishing
- `/python-library-complete:release-management` — when cutting a release
- `/python-library-complete:security-audit` — when reviewing security
- `/python-library-complete:community` — when setting up contributor infrastructure
- `/python-library-complete:library-review` — when doing a broad quality audit

### Key design patterns
- Distributions expose a consistent interface (fit, pdf/log-pdf, log-kernel, sample).
- `SphericalKMeans` follows the scikit-learn estimator API (`fit`, `predict`, `fit_predict`).
- `loguru` is used for logging throughout.
- `scripts/` contains standalone CLI scripts (mixture fitting, vector reduction) using `click`.
