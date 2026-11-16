"""
Module for the convergence of a spherical mixture fit.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SphericalMixtureConvergence:
    """
    How EM converged to the components of a spherical mixture.

    Attributes
    ----------
    log_likelihood:
        Observed-data log-likelihood of the fitted data.
    n_iter:
        Number of EM iterations run, at least one.
    converged:
        Whether the change in log-likelihood fell below the tolerance within
        max_iter iterations.
    """

    log_likelihood: float
    n_iter: int
    converged: bool
