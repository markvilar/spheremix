"""
Module for Spherical Cauchy mixture model.
"""

import numpy as np

from dataclasses import dataclass, field
from typing import Literal
from loguru import logger
from scipy.optimize import minimize
from spheremix.distributions import SphericalCauchyDistribution
from spheremix.distributions import spherical_cauchy_logkernel
from spheremix.distributions import spherical_cauchy_logpdf
from .initializer import MeanInitStrategy, create_mean_initializer


@dataclass(slots=True)
class SphericalCauchyMixtureState:
    """
    State holding component weights and parameters for a Spherical Cauchy mixture model.
    """

    component_weights: tuple[float, ...]
    component_params: tuple[SphericalCauchyDistribution, ...]

    @property
    def n_components(self) -> int:
        """
        Return the number of components in the mixture.
        """
        return len(self.component_weights)


@dataclass(slots=True)
class SphericalCauchyMixture:
    """
    Mixture model of Spherical Cauchy distributions.

    Attributes
    ----------
    weights:
        numpy array with shape (K,) and sums to 1
    means:
        numpy array with shape (K,D)
    concentrations:
        numpy array with shape (K,)
    """

    # Mixture configuration
    n_components: int
    init: Literal["random", "kmeans"] | MeanInitStrategy = "random"

    # Mixture state — set by fit()
    weights: np.ndarray | None = None
    means: np.ndarray | None = None
    concentrations: np.ndarray | None = None

    # Bookkeeping for convergence — set by fit()
    log_likelihood: float = field(default=-np.inf, init=False)
    n_iter: int = field(default=0, init=False)

    def _expectation_step(self, X: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities r_{nk} = p(z=k | x_n, Θ).
        Returns an array of shape (N, K).
        """
        assert (
            self.weights is not None
            and self.means is not None
            and self.concentrations is not None
        )
        N, D = X.shape
        K = self.n_components

        # log‑weights + log‑kernel for each component - shape (N, K)
        log_resp: np.ndarray = np.empty((N, K))
        for k in range(K):
            # Compute unnormalized log-densities - shape (N,)
            log_kernel: np.ndarray = spherical_cauchy_logkernel(
                X, self.means[k], self.concentrations[k]
            )
            log_resp[:, k] = np.log(self.weights[k]) + log_kernel

        # log‑sum‑exp over components (numerically stable) - shape (N, 1)
        log_norm: np.ndarray = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)

        # subtract log norm from responsibilities - shape (N, K)
        log_resp -= log_norm

        # responsibilities in linear space - shape (N, K)
        resp: np.ndarray = np.exp(log_resp)

        return resp

    def _maximization_step(
        self,
        X: np.ndarray,
        resp: np.ndarray,
    ) -> None:
        """
        Update the mixture parameters given responsibilities.
        Modifies self.weights, self.means, self.concentrations.
        """
        assert self.means is not None and self.concentrations is not None
        N, K = resp.shape

        # 1) update mixing weights
        self.weights = resp.sum(axis=0) / N  # (K,)

        # 2) update each component's (μ, ρ) by maximising the weighted log‑likelihood
        for k in range(K):
            # ---- objective: negative weighted log‑likelihood for component k ----
            def neg_weighted_loglik(params: np.ndarray) -> float:
                """
                params = [mu_raw (D,), rho]
                mu_raw is later normalised to unit length.
                """
                mu_raw, rho = params[:-1], params[-1]
                mu = mu_raw / np.linalg.norm(mu_raw)  # enforce ‖mu‖ = 1
                rho = np.clip(rho, 0.0, 0.9999)  # keep in valid range

                # log‑pdf of the spherical Cauchy for all data points
                ll = spherical_cauchy_logpdf(X, mu, rho)  # (N,)
                # weighted sum (negative because we minimise)
                return float(-np.dot(resp[:, k], ll))

            # ---- initialise from current parameters ----
            mu_init = self.means[k]  # already unit‑norm
            rho_init = self.concentrations[k]
            init_params = np.concatenate([mu_init, [rho_init]])

            # ---- bounds: mu_raw unrestricted, rho in [0,1) ----
            bounds = [(-np.inf, np.inf)] * mu_init.size + [(0.0, 1.0)]

            # NOTE: The minimize step takes a lot of time
            res = minimize(
                neg_weighted_loglik,
                init_params,
                bounds=bounds,
                method="L-BFGS-B",
            )

            # ---- extract and store updated parameters ----
            mu_opt = res.x[: mu_init.size]
            mu_opt = mu_opt / np.linalg.norm(mu_opt)  # re‑normalise
            rho_opt = np.clip(res.x[mu_init.size], 0.0, 0.9999)

            self.means[k] = mu_opt
            self.concentrations[k] = rho_opt

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        rtol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        """
        Run EM until convergence or max_iter.
        After fitting, the attributes ``weights``, ``means`` and
        ``concentrations`` hold the learned parameters.
        """
        K: int = self.n_components
        D: int = X.shape[1]

        if D < 2:
            raise ValueError("number of features must be 2 or higher")

        mean_init: MeanInitStrategy = (
            create_mean_initializer(self.init)
            if isinstance(self.init, str)
            else self.init
        )

        # Initialize weights, means, and concentrations
        self.weights = _init_component_weights(K)
        self.means = mean_init(X, K)
        self.concentrations = _init_component_concentrations(K)

        # ----- EM loop --------------------------------------------------
        for i in range(max_iter):
            # E‑step
            logger.info("Expectation step:")
            responsibilities = self._expectation_step(X)  # (N, K)

            # M‑step
            logger.info("Maximization step:")
            self._maximization_step(X, responsibilities)

            if self._check_convergence(X, i, tol, rtol, verbose):
                break

            self.n_iter += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(z=k | x_n, Θ) for each sample.
        Returns an array of shape (N, K).
        """
        if self.weights is None or self.means is None or self.concentrations is None:
            raise ValueError(
                "Model is not fitted yet. Call fit() before predict_proba()."
            )
        return self._expectation_step(X)

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
        Equivalent to calling fit() followed by predict(X).
        Returns an array of shape (N,) with integer component indices.
        """
        self.fit(X, max_iter=max_iter, tol=tol, rtol=rtol, verbose=verbose)
        return self.predict(X)

    def _check_convergence(
        self,
        X: np.ndarray,
        iteration: int,
        tol: float,
        rtol: float,
        verbose: bool,
    ) -> bool:
        """
        Compute the observed-data log-likelihood, log progress, check for
        convergence, and update ``self.log_likelihood``.

        Convergence is declared when either the absolute change in
        log-likelihood is below ``tol``, or the relative change is below
        ``rtol``.  Using both criteria makes the check robust across
        problem scales: the absolute criterion catches near-zero likelihoods
        while the relative criterion handles high-dimensional data where the
        log-likelihood magnitude grows with D and N.

        Returns True if either convergence criterion is satisfied.
        """
        assert (
            self.weights is not None
            and self.means is not None
            and self.concentrations is not None
        )
        K: int = self.n_components
        N: int = X.shape[0]

        # log Σ_k w_k * p_k(x_n) for each n, summed over n.
        # Computed sample-by-sample with log-sum-exp for numerical stability.
        loglik: float = 0.0
        for n in range(N):
            log_dense: float = float(
                np.logaddexp.reduce(
                    [
                        np.log(self.weights[k])
                        + spherical_cauchy_logkernel(
                            X[n : n + 1], self.means[k], self.concentrations[k]
                        ).squeeze()
                        for k in range(K)
                    ]
                )
            )
            loglik += log_dense

        if verbose:
            logger.info(f"Iter {iteration:03d} – loglik = {loglik:.5f}")

        delta: float = abs(loglik - self.log_likelihood)
        abs_converged: bool = delta < tol
        rel_converged: bool = delta < rtol * max(abs(self.log_likelihood), 1.0)
        converged: bool = abs_converged or rel_converged

        if converged and verbose:
            logger.info("Converged!")

        self.log_likelihood = loglik
        return converged


def _expectation_step() -> None:
    """
    Performs the expectation step of the EM algorithm.
    """
    raise NotImplementedError


def _maximization_step() -> None:
    """
    Performs the maximization step of the EM algorithm.
    """
    raise NotImplementedError


def _init_component_weights(K: int) -> np.ndarray:
    weights: np.ndarray = np.ones(K)
    return weights / np.sum(weights)  # type: ignore[no-any-return]


def _init_component_concentrations(K: int) -> np.ndarray:
    return np.full(shape=(K,), fill_value=0.1)
