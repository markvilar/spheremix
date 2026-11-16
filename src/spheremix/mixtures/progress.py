"""
Module for the progress events that a mixture fit reports after each EM
iteration.
"""

from collections.abc import Callable
from dataclasses import dataclass


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
