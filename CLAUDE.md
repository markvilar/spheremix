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

**Type check:**
```bash
uv run mypy src
```

## Collaboration rules

- Never implement a feature without first confirming with the user that you should proceed.
- Prefer incremental changes — deliver the smallest coherent step, then pause for feedback before continuing.
- If context is missing (intent behind a requirement, constraints, prior decisions), ask before proceeding rather than assuming.
- When a design decision has clear trade-offs (e.g. API shape, algorithmic choice, abstraction boundary), surface the options and ask the user to decide before implementing.

## Git conventions

- Never reference Claude in git commit messages, pull requests, or issues — no `Co-Authored-By` trailers, no body text attributions, and no mentions in source code or documentation.
- Use backtick (`) for inline code and triple backtick (```) for code blocks in GitHub issues and pull requests. Do not use escaped backticks (`\``).

## Code style conventions

### Tooling

| Tool | Command |
|---|---|
| Linting | `uv run ruff check .` |
| Formatting | `uv run ruff format .` |
| Type checking | `uv run mypy .` |
| Tests | `uv run pytest` |

All code must pass all four tools with no errors.

### Types

Always add type hints to return types, arguments, fields, and variables. Use the Python 3.12 `type` statement for type aliases:

```python
type WeightVector = np.ndarray
type ComponentIndex = int
```

### Strings

Use double quotes for all strings. Single quotes are acceptable inside f-strings to avoid escaping:

```python
label: str = "spherical-cauchy"
message: str = f"component {component['name']} converged"
```

### Variable names

Prefer full words over abbreviations — use `message` not `msg`, `result` not `res`, `error` not `err`, `config` not `cfg`. This applies to names of any length; three-character abbreviations are just as discouraged as one- or two-character ones. Exception: short or single-character names are acceptable for class member fields where the meaning is established by context (e.g. `mu`, `rho` on a distribution dataclass).

### Imports

#### Ordering

Group imports in the following order, and separate the `import ...` groups from the `from ... import ...` groups with a blank line:

```python
import math
import os

import numpy as np
import torch

import spheremix

from typing import Sequence

from numpy.typing import NDArray

from spheremix.distributions import PKBDistribution, SphericalCauchyDistribution
from spheremix.mixtures import SphericalCauchyMixture
```

#### Multi-item imports

When importing multiple names from the same package or module, use parentheses with one name per line:

```python
from spheremix.distributions import (
    PKBDistribution,
    SphericalCauchyDistribution,
)
```

#### Internal package imports

Import classes and functions from the package, not from the module within it. This decouples callers from the internal file layout:

```python
# Preferred
from spheremix.distributions import SphericalCauchyDistribution

# Avoid
from spheremix.distributions.spherical_cauchy import SphericalCauchyDistribution
```

### Docstrings

Add a line of hyphens under `Arguments`, `Returns`, and `Attributes` section headers.

Function example:

```python
def fit(
    data: NDArray[np.float64],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SphericalCauchyDistribution:
    """
    Fit a spherical Cauchy distribution to data via EM.

    Arguments
    ---------
    data: Unit vectors of shape (n, d).
    max_iter: Maximum number of EM iterations.
    tol: Convergence tolerance on the log-likelihood.

    Returns
    -------
    Fitted SphericalCauchyDistribution.
    """
```

Class example:

```python
@dataclass
class SphericalCauchyMixtureParams:
    """
    Fitted parameters for a Spherical Cauchy mixture model.

    Attributes
    ----------
    weights: Mixing weights, shape (K,), sums to 1.
    means: Component means (unit vectors), shape (K, D).
    concentrations: Concentration parameters rho, shape (K,), values in [0, 1).
    """

    weights: np.ndarray
    means: np.ndarray
    concentrations: np.ndarray
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

When developing Python code in this project, use these skills at the appropriate moments:

- `/python-library-complete:api-design` — before finalising any new public interface (functions, types, exceptions). Use it to evaluate naming, composability, and error handling before the design is locked in.
- `/python-library-complete:code-quality` — after implementing a module or non-trivial function. Use it to check for Pythonic idioms, type hint completeness, and mypy strictness issues that ruff won't catch.
- `/python-library-complete:testing-strategy` — when adding a new module or a protocol feature with structured input space. Use it to design fixtures, parametrization, and property-based tests before writing test code.

Also invoke the following skills when relevant to the work at hand:

- `/python-library-complete:cli-development` — when adding or modifying CLI scripts
- `/python-library-complete:performance` — when optimising hot paths or benchmarking
- `/python-library-complete:documentation` — when writing docstrings or user guides
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

## Agent skills

### Domain docs

Single-context repo — one `CONTEXT.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.
