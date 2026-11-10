# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

**Spheremix** is a Python library for spherical probability distributions and mixture models on unit hyperspheres. Unlike a research repository, the code here is a **library with a public API**: downstream projects (e.g. `benthloc-experiments`) depend on it through tag-pinned versions. Changes to the public API are deliberate, versioned, and covered by tests. Prototypes belong in the downstream repositories until they are general-purpose, thoroughly tested, and fit the scope of spheremix — promotion into the library is a feature proposal with a public API, tests, and a version bump, not a file move.

**Before writing new code for a sub-problem, search spheremix for a primitive that already solves it** — chiefly in `distributions/`, `utils/`, and `algorithm/`. Re-rolling existing functionality is how duplication starts.

## Commands

```bash
# Install / sync dependencies
uv sync --all-extras --dev

# Build
uv build

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src

# Run tests
uv run pytest tests                                         # all tests
uv run pytest tests/distributions/test_spherical_cauchy.py  # single file
uv run pytest tests/path/to/test.py::test_function_name     # single test
```

## Architecture

### Package layout (`src/spheremix/`)

| Module | Purpose |
|---|---|
| `distributions/` | Probability distributions on unit spheres — `SphericalCauchyDistribution` (location `mu`, concentration `rho`), `PKBDistribution` (Projected Kent-Bingham) |
| `mixtures/` | Mixture models — `SphericalCauchyMixture` with EM algorithm support, tracking component weights, means, and concentrations |
| `cluster/` | `SphericalKMeans` — scikit-learn-compatible K-means for spherical data (Dhillon & Modha 2001) |
| `algorithm/` | Core algorithm skeletons — `mixture_em.py` (EM for mixture models, not yet fully implemented) |
| `utils/` | `generators.py` — orthogonal unit vector generation via QR decomposition |

### Key design patterns

- Distributions expose a consistent interface (fit, pdf/log-pdf, log-kernel, sample).
- `SphericalKMeans` follows the scikit-learn estimator API (`fit`, `predict`, `fit_predict`).
- `loguru` is used for logging throughout.
- `scripts/` contains standalone CLI scripts (mixture fitting, vector reduction) using `click`.

## Collaboration rules

- Never implement a feature without first confirming with the user that you should proceed.
- Never commit code or open a pull request without first asking for confirmation, unless the user has explicitly said you may do so.
- Never reference Claude, Anthropic, or AI authorship in commit messages, pull request descriptions, GitHub issue text, source code, or documentation — no `Co-Authored-By` trailers and no body text attributions.
- Prefer incremental changes — deliver the smallest coherent step, then pause for feedback before continuing.
- If context is missing (intent behind a requirement, constraints, prior decisions), ask before proceeding rather than assuming.
- When a design decision has clear trade-offs (e.g. API shape, algorithmic choice, abstraction boundary), surface the options and ask the user to decide before implementing.

## GitHub workflow

Use backtick (`) for inline code and triple backtick (```) for code blocks in GitHub issues and pull requests. Do not use escaped backticks (`\``).

### Branches and pull requests

- `main` is the release branch. Day-to-day work goes into `dev`.
- Each issue gets its own branch from `dev`, named `<type>/<issue>-<slug>`, e.g. `feature/63-vmf-sampling` or `refactor/64-mixture-params`.
- Open pull requests into `dev`, one issue per pull request.
- Merges into `dev` do not close issues automatically. After a merge, close the issue by hand with a comment such as "Implemented in #<pr>.", and set its status to Done.

### Project status

Issues are tracked on project #17 (`github.com/users/markvilar/projects/17`). Move an issue through the statuses as the work progresses:

| Status      | When                                           |
|-------------|------------------------------------------------|
| Deferred    | Deliberately postponed; not planned for now    |
| Backlog     | Created, design not settled                    |
| Ready       | Design settled and approved for implementation |
| In progress | Implementation has started on a branch         |
| In review   | The pull request is open                       |
| Done        | Merged and the issue closed                    |

### Issues

- Assign new issues to `markvilar` and give them the Backlog status.
- Structure the issue body as numbered sections:

  ```markdown
  ## 1. Problem and motivation
  ## 2. Proposed solution
  ### 2.1 <Subsection>
  ### 2.2 <Subsection>
  ```

  Add further top-level sections (e.g. `## 3. Open questions`) only when they are needed. State dependencies on other issues in one closing line, e.g. "Depends on #62."
- Design issues incrementally. Agree the subsection layout first, post it with empty subsections, then draft, review and post one subsection at a time. Never post a draft that has not been reviewed.
- Convey the design with code blocks: signatures, dataclasses, module trees, CLI invocations. Keep prose to the short text each block needs. Code blocks follow the Python code style in this file.
- For complex functions, such as an EM loop, outline the flow as skeleton code: the signature, followed by numbered step comments that say what each step does. Separate the steps with a blank line, and keep the code under each step to a sketch, or leave it out:

  ```python
  def fit_mixture(
      data: NDArray[np.float64],
      num_components: int,
      max_iter: int = 100,
  ) -> SphericalCauchyMixture:
      # 1. Initialise the component means with spherical K-means.

      # 2. E-step: compute the responsibilities of each component.

      # 3. M-step: update weights, means and concentrations.

      # 4. Stop when the log-likelihood change falls below the tolerance.

      # 5. Return the fitted mixture.
  ```

- Keep the problem section about why the change is needed and what it enables, not how it is implemented.
- Move a decision that has grown beyond the issue's scope into a new issue, rather than widening the issue.

## Python code style

### Tooling

| Tool | Command |
|---|---|
| Linting | `uv run ruff check .` |
| Formatting | `uv run ruff format .` |
| Type checking | `uv run mypy src` |
| Tests | `uv run pytest` |

All code must pass all four tools with no errors.

### Test run time

Keep unit tests fast; the suite runs in parallel (`pytest -n auto`, via pytest-xdist) and should stay quick enough to run on every change. Do not add excessively slow tests, such as Monte Carlo tests with a high number of samples: use the smallest sample count that still makes the assertion meaningful, widen the tolerance to match (the sampling error shrinks as 1/sqrt(n)), and check that a deliberately broken implementation still fails. Prefer vectorised code over Python loops in test set-up, and avoid spawning a fresh interpreter unless the test truly needs one (e.g. import-order tests).

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

Group imports in the following order, separating `import ...` groups from `from ... import ...` groups with a blank line:

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

### Module layout

Place public types, functions, and classes near the top of a module. Place private types, functions, and classes (those prefixed with `_`) further down, after all public definitions.

### Function signatures

When a signature does not fit on one line, put each argument on its own line and end the last one with a trailing comma. The trailing comma keeps `ruff format` from collapsing the arguments back onto one line:

```python
def log_density(
    points: Tensor,
    mean_directions: Tensor,
    concentrations: Tensor,
) -> Tensor:
```

### Docstrings

Add a line of hyphens under `Arguments`, `Returns`, and `Attributes` section headers.

State the expected shape of every array or tensor argument and return value, e.g. `points: Unit vectors, shape (B, d).`

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
    data: Unit vectors, shape (n, d).
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

### Exceptions

Prefer built-in exceptions with descriptive messages. Define a custom exception type only when it belongs to a widely used component and callers need to catch it separately from the built-in exceptions, e.g. a `ConvergenceError` raised by the EM algorithm. This keeps the number of exception types small enough to manage.

- **Ask before defining an exception type.** Name the type, the package it would live in, and where it would be raised and caught, so that the decision is made explicitly.
- **Raise a built-in exception for a one-off error,** such as `ValueError`, `KeyError`, `FileNotFoundError` or `RuntimeError`. Make the message say what failed and how to fix it:

  ```python
  raise ValueError(
      f"concentration rho must lie in [0, 1), got {rho}; clip or rescale the input"
  )
  ```

- **Define a custom type in the package of the component that raises it,** never in a caller such as a CLI script, and export it from that package.
- **Subclass the closest built-in exception,** e.g. `class ConvergenceError(RuntimeError)`, so that callers that catch the built-in still catch it.

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

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`github.com/markvilar/spheremix`). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.
