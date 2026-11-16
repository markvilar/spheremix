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
    log_likelihood_history:
        Log-likelihood after each iteration, length n_iter; the last value
        equals log_likelihood.
    """

    log_likelihood: float
    n_iter: int
    converged: bool
    log_likelihood_history: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.log_likelihood_history) != self.n_iter:
            raise ValueError(
                f"log_likelihood_history must have n_iter = {self.n_iter} values, "
                f"got {len(self.log_likelihood_history)}"
            )
        if (
            self.log_likelihood_history
            and self.log_likelihood_history[-1] != self.log_likelihood
        ):
            raise ValueError(
                "the last value of log_likelihood_history must equal "
                f"log_likelihood = {self.log_likelihood}, "
                f"got {self.log_likelihood_history[-1]}"
            )
