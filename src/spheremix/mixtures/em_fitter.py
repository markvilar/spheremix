"""
Module for the expectation-maximisation loop shared by the spherical mixtures.

The fitter takes the E-step and M-step as functions, so each mixture keeps its
own components type and M-step while sharing the loop, the convergence check
and the logging.
"""

import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

from .convergence import SphericalMixtureConvergence
from .protocols import SphericalMixtureComponents
from .scoring import compute_log_likelihoods, compute_responsibilities

type ExpectationStep[C: SphericalMixtureComponents] = Callable[
    [np.ndarray, C], np.ndarray
]
type MaximizationStep[C: SphericalMixtureComponents] = Callable[
    [np.ndarray, np.ndarray, C], C
]


@dataclass(slots=True, frozen=True)
class FitProgress:
    """
    The progress of a mixture fit after one EM iteration.

    Attributes
    ----------
    n_iter:
        Number of completed iterations, from 1 to max_iter.
    max_iter:
        Maximum number of iterations of the fit.
    log_likelihood:
        Observed-data log-likelihood after the iteration.
    log_likelihood_change:
        Absolute change of the log-likelihood from the previous iteration;
        infinite after the first iteration.
    converged:
        Whether the fit converged at this iteration.
    """

    n_iter: int
    max_iter: int
    log_likelihood: float
    log_likelihood_change: float
    converged: bool


type FitProgressCallback = Callable[[FitProgress], None]


@runtime_checkable
class EMSettings(Protocol):
    """
    The settings of the EM loop, met by the params of every mixture.

    Attributes
    ----------
    max_iter:
        Maximum number of EM iterations, at least one.
    tol:
        Absolute tolerance on the change in log-likelihood.
    rtol:
        Relative tolerance on the change in log-likelihood.
    verbose:
        Whether to log the log-likelihood of each iteration.
    """

    @property
    def max_iter(self) -> int: ...

    @property
    def tol(self) -> float: ...

    @property
    def rtol(self) -> float: ...

    @property
    def verbose(self) -> bool: ...


def fit_mixture_em[C: SphericalMixtureComponents](
    X: np.ndarray,
    initial_components: C,
    maximization: MaximizationStep[C],
    settings: EMSettings,
    *,
    expectation: ExpectationStep[C] = compute_responsibilities,
    progress_callback: FitProgressCallback | None = None,
) -> tuple[C, SphericalMixtureConvergence]:
    """
    Fit mixture components to data with expectation-maximisation.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    initial_components:
        Components to start from, of dimension D.
    maximization:
        M-step: given X, responsibilities of shape (N, K) and the current
        components, return the updated components.
    settings:
        Iteration limit, tolerances and verbosity of the loop.
    expectation:
        E-step: given X and the components, return responsibilities of
        shape (N, K).
    progress_callback:
        Called with a FitProgress after each iteration, if given. An exception
        it raises ends the fit.

    Returns
    -------
    The fitted components, and how EM converged to them.
    """
    # 1. Check that the data match the dimension of the components.
    if X.ndim != 2 or X.shape[1] != initial_components.dimension:
        raise ValueError(
            f"X must have shape (N, {initial_components.dimension}) to match the "
            f"initial components, got {X.shape}"
        )

    components: C = initial_components
    log_likelihood: float = -np.inf
    n_iter: int = 0
    converged: bool = False

    for iteration in range(settings.max_iter):
        # 2. E-step: compute the responsibilities of the components.
        responsibilities: np.ndarray = expectation(X, components)

        # 3. M-step: update the components.
        components = maximization(X, responsibilities, components)

        # 4. Compute the log-likelihood and its change, and check convergence.
        previous_log_likelihood: float = log_likelihood
        log_likelihood = float(compute_log_likelihoods(X, components).sum())
        if settings.verbose:
            logger.info(f"Iter {iteration:03d} – loglik = {log_likelihood:.5f}")
        change: float = abs(log_likelihood - previous_log_likelihood)
        n_iter = iteration + 1
        converged = _has_converged(
            change, previous_log_likelihood, settings.tol, settings.rtol
        )

        # 5. Report the progress to the callback, if given.
        if progress_callback is not None:
            progress_callback(
                FitProgress(
                    n_iter=n_iter,
                    max_iter=settings.max_iter,
                    log_likelihood=log_likelihood,
                    log_likelihood_change=change,
                    converged=converged,
                )
            )

        # 6. Stop if converged.
        if converged:
            if settings.verbose:
                logger.info("Converged!")
            break

    # 7. Return the components and how EM converged to them.
    convergence: SphericalMixtureConvergence = SphericalMixtureConvergence(
        log_likelihood=log_likelihood,
        n_iter=n_iter,
        converged=converged,
    )
    return components, convergence


def _has_converged(
    change: float,
    previous_log_likelihood: float,
    tol: float,
    rtol: float,
) -> bool:
    """
    Return whether the absolute change in log-likelihood is below the absolute
    tolerance, or below the relative tolerance times max(|previous|, 1).
    """
    scale: float = max(abs(previous_log_likelihood), 1.0)
    return change < tol or change < rtol * scale
