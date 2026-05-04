"""
Module for Spherical Cauchy mixture model.
"""

import numpy as np

from dataclasses import dataclass, field
from loguru import logger
from scipy.optimize import minimize
from spheremix.distributions import SphericalCauchyDistribution
from spheremix.distributions import spherical_cauchy_logkernel
from spheremix.distributions import spherical_cauchy_logpdf


@dataclass(slots=True)
class SphericalCauchyMixtureState:
    """
    State holding component weights and parameters for a Spherical Cauchy mixture model.
    """

    component_weights: list[float] = field(default_factory=list)
    component_params: list[SphericalCauchyDistribution] = field(default_factory=list)


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

    n_components: int
    weights: np.ndarray | None = None
    means: np.ndarray | None = None
    concentrations: np.ndarray | None = None

    # optional: bookkeeping for convergence
    log_likelihood: float = field(default=-np.inf, init=False)
    n_iter: int = field(default=0, init=False)

    def expectation_step(self, X: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities r_{nk} = p(z=k | x_n, Θ).
        Returns an array of shape (N, K).
        """
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

    def maximization_step(
        self,
        X: np.ndarray,
        resp: np.ndarray,
    ) -> None:
        """
        Update the mixture parameters given responsibilities.
        Modifies self.weights, self.means, self.concentrations.
        """
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
                return -np.dot(resp[:, k], ll)

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
        verbose: bool = False,
    ) -> None:
        """
        Run EM until convergence or max_iter.
        After fitting, the attributes ``weights``, ``means`` and
        ``concentrations`` hold the learned parameters.
        """
        K: int = self.n_components
        N: int = X.shape[0]
        D: int = X.shape[1]

        # Initialize weights, means, and concentrations
        self.weights = _init_component_weights(K)
        self.means = _init_component_means(K, D)
        self.concentrations = _init_component_concentrations(K)

        logger.info(f"K: {K}")
        logger.info(f"N: {N}")
        logger.info(f"D: {D}")

        logger.info(f"Initial weights:          {self.weights}")
        logger.info(f"Initial means:            {self.means.shape}")
        logger.info(f"Initial concentrations:   {self.concentrations.shape}")

        logger.info(f"Means:        {self.means}")
        logger.info(f"Means norm:   {np.linalg.norm(self.means, axis=1)}")

        # ----- EM loop --------------------------------------------------
        for i in range(max_iter):
            # E‑step
            logger.info("Expectation step:")
            responsibilities = self.expectation_step(X)  # (N, K)

            # M‑step
            logger.info("Maximization step:")
            self.maximization_step(X, responsibilities)

            # ----- compute observed‑data log‑likelihood (for convergence) -----
            # Using log‑sum‑exp for stability:
            #   loglik = Σ_n log Σ_k w_k * exp(logpdf_k(x_n))
            loglik = 0.0
            for n in range(N):
                # compute log of mixture density for a single datum
                log_dense = np.logaddexp.reduce(
                    [
                        np.log(self.weights[k])
                        + spherical_cauchy_logkernel(
                            X[n : n + 1], self.means[k], self.concentrations[k]
                        ).squeeze()
                        for k in range(K)
                    ]
                )
                loglik += log_dense
            # loglik now holds Σ_n log p(x_n)

            if verbose:
                logger.info(f"Iter {i:03d} – loglik = {loglik:.5f}")

            # convergence check
            if np.abs(loglik - self.log_likelihood) < tol:
                if verbose:
                    logger.info("Converged!")
                break

            self.log_likelihood = loglik
            self.n_iter += 1


def normalize_rows(X: np.ndarray, eps=1e-12) -> np.ndarray:
    """L2-normalize each row of X."""
    norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms


def _init_component_weights(K: int) -> np.ndarray:
    weights: np.ndarray = np.ones(K)
    return weights / np.sum(weights)


def _init_component_means(
    K: int, D: int, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    means: np.ndarray = np.random.uniform(low=0.0, high=1.0, size=(K, D))
    if D == 1:
        return means
    else:
        return normalize_rows(means)


def _init_component_concentrations(K: int) -> np.ndarray:
    return np.full(shape=(K,), fill_value=0.1)
