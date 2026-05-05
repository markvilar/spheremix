"""
Module for Spherical Cauchy mixture model.
"""

import numpy as np
import scipy.optimize

from dataclasses import dataclass, field
from typing import Literal
from loguru import logger
from spheremix.distributions import spherical_cauchy_logkernel
from spheremix.distributions import spherical_cauchy_logpdf
from .initializer import MeanInitStrategy, create_mean_initializer


# ---------------------------------------------------------------------------
# Mixture components — fitted parameters, replaced atomically on each fit
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SphericalCauchyMixtureComponents:
    """
    Fitted parameters for a Spherical Cauchy mixture model.

    Attributes
    ----------
    weights:
        Mixing weights, shape (K,), sums to 1.
    means:
        Component means (unit vectors), shape (K, D).
    concentrations:
        Concentration parameters rho, shape (K,), values in [0, 1).
    """

    weights: np.ndarray
    means: np.ndarray
    concentrations: np.ndarray


# ---------------------------------------------------------------------------
# Mixture model — facade; delegates EM to fit_spherical_cauchy_mixture()
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SphericalCauchyMixture:
    """
    Mixture model of Spherical Cauchy distributions.

    Fit via EM by calling ``fit()``. The learned parameters are stored
    atomically in ``params_`` and replaced in full on each re-fit.

    Attributes
    ----------
    n_components:
        Number of mixture components.
    init:
        Mean initialization strategy.
    params_:
        Fitted mixture components; ``None`` until ``fit()`` is called.
    log_likelihood_:
        Observed-data log-likelihood at the end of EM.
    n_iter_:
        Number of EM iterations performed.
    """

    n_components: int
    init: Literal["random", "kmeans"] | MeanInitStrategy = "random"

    # Fitted state — set by fit(), replaced atomically on each call
    params_: SphericalCauchyMixtureComponents | None = field(default=None, init=False)
    log_likelihood_: float = field(default=-np.inf, init=False)
    n_iter_: int = field(default=0, init=False)

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        rtol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        """
        Run EM until convergence or ``max_iter``.
        Updates ``params_``, ``log_likelihood_``, and ``n_iter_``.
        """
        self.params_, self.log_likelihood_, self.n_iter_ = fit_spherical_cauchy_mixture(
            X,
            n_components=self.n_components,
            init=self.init,
            max_iter=max_iter,
            tol=tol,
            rtol=rtol,
            verbose=verbose,
        )

    def get_params(self) -> SphericalCauchyMixtureComponents:
        """
        Return the fitted mixture components.
        Raises ``ValueError`` if the model has not been fitted yet.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before get_params().")
        return self.params_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(z=k | x_n, Θ) for each sample.
        Returns an array of shape (N, K).
        """
        p = self.get_params()
        return _compute_responsibilities(X, p.weights, p.means, p.concentrations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the most probable component.
        Returns an array of shape (N,) with integer component indices.
        """
        return self.predict_proba(X).argmax(axis=1)  # type: ignore[no-any-return]

    def fit_predict(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        rtol: float = 1e-6,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Fit the model and return hard component assignments.
        Returns an array of shape (N,) with integer component indices.
        """
        self.fit(X, max_iter=max_iter, tol=tol, rtol=rtol, verbose=verbose)
        return self.predict(X)


# ---------------------------------------------------------------------------
# Fitter function — owns all EM logic
# ---------------------------------------------------------------------------


def fit_spherical_cauchy_mixture(
    X: np.ndarray,
    n_components: int,
    init: Literal["random", "kmeans"] | MeanInitStrategy = "random",
    max_iter: int = 100,
    tol: float = 1e-4,
    rtol: float = 1e-6,
    verbose: bool = False,
) -> tuple[SphericalCauchyMixtureComponents, float, int]:
    """
    Fit a Spherical Cauchy mixture model via EM.

    Returns the fitted components, final log-likelihood, and iteration count.
    """
    K: int = n_components
    D: int = X.shape[1]

    if D < 2:
        raise ValueError("number of features must be 2 or higher")

    mean_init: MeanInitStrategy = (
        create_mean_initializer(init) if isinstance(init, str) else init
    )

    weights: np.ndarray = _init_component_weights(K)
    means: np.ndarray = mean_init(X, K)
    concentrations: np.ndarray = _init_component_concentrations(K)
    log_likelihood: float = -np.inf
    n_iter: int = 0

    for i in range(max_iter):
        resp: np.ndarray = _compute_responsibilities(X, weights, means, concentrations)
        weights, means, concentrations = _maximization_step(
            X, resp, means, concentrations
        )
        converged, log_likelihood = _check_convergence(
            X, weights, means, concentrations, i, log_likelihood, tol, rtol, verbose
        )
        n_iter = i + 1
        if converged:
            break

    components = SphericalCauchyMixtureComponents(
        weights=weights,
        means=means,
        concentrations=concentrations,
    )
    return components, log_likelihood, n_iter


# ---------------------------------------------------------------------------
# Private EM helpers
# ---------------------------------------------------------------------------


def _compute_responsibilities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    Compute r_{nk} = p(z=k | x_n, Θ) for all samples via log-sum-exp. Returns (N, K).
    """
    N: int = X.shape[0]
    K: int = len(weights)

    log_resp: np.ndarray = np.empty((N, K))
    for k in range(K):
        log_kernel: np.ndarray = spherical_cauchy_logkernel(
            X, means[k], concentrations[k]
        )
        log_resp[:, k] = np.log(weights[k]) + log_kernel

    log_norm: np.ndarray = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
    log_resp -= log_norm
    return np.exp(log_resp)  # type: ignore[no-any-return]


def _maximization_step(
    X: np.ndarray,
    resp: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update weights, means, and concentrations given responsibilities.
    Returns updated (weights, means, concentrations).
    """
    N, K = resp.shape
    new_weights: np.ndarray = resp.sum(axis=0) / N  # type: ignore[assignment]
    new_means: np.ndarray = means.copy()
    new_concentrations: np.ndarray = concentrations.copy()

    for k in range(K):

        def neg_weighted_loglik(params: np.ndarray, k: int = k) -> float:
            mu_raw = params[:-1]
            rho = float(np.clip(params[-1], 0.0, 0.9999))
            mu = mu_raw / np.linalg.norm(mu_raw)
            ll = spherical_cauchy_logpdf(X, mu, rho)
            return float(-np.dot(resp[:, k], ll))

        mu_init = means[k]
        rho_init = concentrations[k]
        init_params = np.concatenate([mu_init, [rho_init]])
        bounds = [(-np.inf, np.inf)] * mu_init.size + [(0.0, 1.0)]

        # NOTE: L-BFGS-B optimization is the dominant cost per EM iteration
        res = scipy.optimize.minimize(
            neg_weighted_loglik,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
        )

        mu_opt = res.x[: mu_init.size]
        mu_opt = mu_opt / np.linalg.norm(mu_opt)
        rho_opt = float(np.clip(res.x[mu_init.size], 0.0, 0.9999))

        new_means[k] = mu_opt
        new_concentrations[k] = rho_opt

    return new_weights, new_means, new_concentrations


def _check_convergence(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
    iteration: int,
    prev_loglik: float,
    tol: float,
    rtol: float,
    verbose: bool,
) -> tuple[bool, float]:
    """
    Compute observed-data log-likelihood and check absolute and relative convergence.
    Returns (converged, loglik).
    """
    K: int = len(weights)
    N: int = X.shape[0]

    loglik: float = 0.0
    for n in range(N):
        log_dense: float = float(
            np.logaddexp.reduce(
                [
                    np.log(weights[k])
                    + spherical_cauchy_logkernel(
                        X[n : n + 1], means[k], concentrations[k]
                    ).squeeze()
                    for k in range(K)
                ]
            )
        )
        loglik += log_dense

    if verbose:
        logger.info(f"Iter {iteration:03d} – loglik = {loglik:.5f}")

    delta: float = abs(loglik - prev_loglik)
    abs_converged: bool = delta < tol
    rel_converged: bool = delta < rtol * max(abs(prev_loglik), 1.0)
    converged: bool = abs_converged or rel_converged

    if converged and verbose:
        logger.info("Converged!")

    return converged, loglik


def _init_component_weights(K: int) -> np.ndarray:
    """Return uniform mixing weights summing to 1."""
    weights: np.ndarray = np.ones(K)
    return weights / np.sum(weights)  # type: ignore[no-any-return]


def _init_component_concentrations(K: int, init_value: float = 0.1) -> np.ndarray:
    """Return initial concentration parameters, all set to 0.1."""
    return np.full(shape=(K,), fill_value=init_value)
