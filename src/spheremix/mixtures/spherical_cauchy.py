"""
Module for Spherical Cauchy mixture model.
"""

import numpy as np
import scipy.optimize

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Self
from loguru import logger
from spheremix.distributions import spherical_cauchy_logkernel
from spheremix.distributions import spherical_cauchy_logpdf
from .base import SerializableMixin, _classification_entropy
from .initializer import MeanInitStrategy, create_mean_initializer


# ---------------------------------------------------------------------------
# Mixture components — fitted parameters, replaced atomically on each fit
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SphericalCauchyMixtureParams:
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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict with nested Python lists."""
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "concentrations": self.concentrations.tolist(),
        }


# ---------------------------------------------------------------------------
# Mixture model — facade; delegates EM to fit_spherical_cauchy_mixture()
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SphericalCauchyMixture(SerializableMixin):
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
    max_iter:
        Maximum number of EM iterations.
    tol:
        Absolute log-likelihood convergence tolerance.
    rtol:
        Relative log-likelihood convergence tolerance.
    verbose:
        Log EM iteration progress.
    seed:
        Seed for the mean initialiser; the same seed gives the same fit.
        ``None`` draws a fresh unseeded generator on each fit.
    params_:
        Fitted mixture components; ``None`` until ``fit()`` is called.
    log_likelihood_:
        Observed-data log-likelihood at the end of EM.
    n_iter_:
        Number of EM iterations performed.
    """

    _SERIAL_VERSION: ClassVar[int] = 2

    n_components: int
    init: Literal["random", "kmeans"] = "random"
    max_iter: int = 100
    tol: float = 1e-4
    rtol: float = 1e-6
    verbose: bool = False
    seed: int | None = None

    # Fitted state — set by fit(), replaced atomically on each call
    params_: SphericalCauchyMixtureParams | None = field(default=None, init=False)
    log_likelihood_: float = field(default=-np.inf, init=False)
    n_iter_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """
        Run EM until convergence or ``max_iter``.
        Updates ``params_``, ``log_likelihood_``, and ``n_iter_``.
        """
        self.params_, self.log_likelihood_, self.n_iter_ = fit_spherical_cauchy_mixture(
            X,
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            rtol=self.rtol,
            verbose=self.verbose,
            seed=self.seed,
        )
        return self

    def get_params(self) -> SphericalCauchyMixtureParams:
        """
        Return the fitted mixture components.
        Raises ``ValueError`` if the model has not been fitted yet.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before get_params().")
        return self.params_

    def is_fitted(self) -> bool:
        """Return True if the model has been fitted, False otherwise."""
        return self.params_ is not None

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

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit the model and return hard component assignments.
        Returns an array of shape (N,) with integer component indices.
        """
        return self.fit(X).predict(X)

    def aic(self, X: np.ndarray) -> float:
        """
        Akaike Information Criterion for the fitted model on X.

        Lower is better. Penalises complexity with ``2 × n_parameters``.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).

        Returns
        -------
        float
        """
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet. Call fit() before aic().")
        n_params = self.n_components * (X.shape[1] + 1) - 1
        return float(-2.0 * self.score_samples(X).sum() + 2.0 * n_params)

    def bic(self, X: np.ndarray) -> float:
        """
        Bayesian Information Criterion for the fitted model on X.

        Lower is better. Penalises complexity with ``n_parameters × log(N)``.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).

        Returns
        -------
        float
        """
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet. Call fit() before bic().")
        n_params = self.n_components * (X.shape[1] + 1) - 1
        return float(-2.0 * self.score_samples(X).sum() + n_params * np.log(X.shape[0]))

    def icl(self, X: np.ndarray) -> float:
        """
        Integrated Completed Likelihood for the fitted model on X.

        Lower is better. Extends BIC with a classification entropy term that
        increases when component assignments are uncertain.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).

        Returns
        -------
        float
        """
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet. Call fit() before icl().")
        entropy: float = _classification_entropy(self.predict_proba(X))
        return float(self.bic(X) + 2.0 * entropy)

    def bici(self, X: np.ndarray) -> float:
        """
        Complete-data BIC for the fitted model on X.

        Lower is better. Extends BIC using the complete-data log-likelihood
        under hard MAP component assignments.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).

        Returns
        -------
        float
        """
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet. Call fit() before bici().")
        responsibilities: np.ndarray = self.predict_proba(X)
        assignments: np.ndarray = responsibilities.argmax(axis=1)
        log_map_prob: np.ndarray = np.log(
            responsibilities[np.arange(X.shape[0]), assignments]
        )
        return float(self.bic(X) - 2.0 * log_map_prob.sum())

    def dic(self, X: np.ndarray) -> float:
        """
        Deviance Information Criterion (complete-data) for the fitted model on X.

        Lower is better. Uses the EM Q-function (expected complete-data
        log-likelihood) with a flat 2 × n_parameters complexity penalty.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).

        Returns
        -------
        float
        """
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet. Call fit() before dic().")
        n_params: int = self.n_components * (X.shape[1] + 1) - 1
        entropy: float = _classification_entropy(self.predict_proba(X))
        log_likelihood: float = float(self.score_samples(X).sum())
        return float(-2.0 * log_likelihood + 2.0 * entropy + 2.0 * n_params)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log p(x | Θ), shape (N,). Must use logpdf not logkernel —
        the normalising constant varies with ρ and directly affects AIC/BIC."""
        p = self.get_params()
        K = self.n_components
        N = X.shape[0]
        log_joint = np.empty((N, K))
        for k in range(K):
            log_joint[:, k] = np.log(p.weights[k]) + spherical_cauchy_logpdf(
                X, p.means[k], p.concentrations[k]
            )
        return np.logaddexp.reduce(log_joint, axis=1)  # type: ignore[no-any-return]

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """
        Mean per-sample log-likelihood of X under the fitted model.

        Mirrors the sklearn ``score`` convention. Higher is better.
        Requires the model to be fitted.

        Parameters
        ----------
        X:
            Unit-vector data, shape (N, D).
        y:
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        float
        """
        return float(self.score_samples(X).mean())

    def _params_to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "rtol": self.rtol,
            "verbose": self.verbose,
            "seed": self.seed,
            "log_likelihood_": self.log_likelihood_,
            "n_iter_": self.n_iter_,
            "params_": self.params_.to_dict() if self.params_ is not None else None,
        }

    @classmethod
    def _params_from_dict(cls, data: dict) -> "SphericalCauchyMixture":
        model = cls(
            n_components=data["n_components"],
            init=data["init"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            rtol=data["rtol"],
            verbose=data["verbose"],
            seed=data.get("seed"),
        )
        model.log_likelihood_ = data["log_likelihood_"]
        model.n_iter_ = data["n_iter_"]

        if data["params_"] is not None:
            p = data["params_"]
            model.params_ = SphericalCauchyMixtureParams(
                weights=np.array(p["weights"]),
                means=np.array(p["means"]),
                concentrations=np.array(p["concentrations"]),
            )

        return model


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
    seed: int | None = None,
) -> tuple[SphericalCauchyMixtureParams, float, int]:
    """
    Fit a Spherical Cauchy mixture model via EM.

    ``seed`` seeds the named initialisers; it is ignored when ``init`` is a
    ``MeanInitStrategy``, which owns its own randomness.

    Returns the fitted components, final log-likelihood, and iteration count.
    """
    K: int = n_components
    D: int = X.shape[1]

    if D < 2:
        raise ValueError("number of features must be 2 or higher")

    mean_init: MeanInitStrategy = (
        create_mean_initializer(init, seed=seed) if isinstance(init, str) else init
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

    components = SphericalCauchyMixtureParams(
        weights=weights,
        means=means,
        concentrations=concentrations,
    )
    return components, log_likelihood, n_iter


# ---------------------------------------------------------------------------
# Private EM helpers
# ---------------------------------------------------------------------------


def _compute_log_joint_densities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    Compute log p(x_n, z=k | Θ) = log w_k + log_kernel(x_n | μ_k, ρ_k) for all
    samples and components. Returns shape (N, K).

    Values use the log-kernel rather than the log-pdf, so they are offset by the
    rho-dependent normalising constant of the Spherical Cauchy distribution.
    """
    K: int = len(weights)
    N: int = X.shape[0]

    log_joint: np.ndarray = np.empty((N, K))
    for k in range(K):
        log_kernel: np.ndarray = spherical_cauchy_logkernel(
            X, means[k], concentrations[k]
        )
        log_joint[:, k] = np.log(weights[k]) + log_kernel

    return log_joint  # type: ignore[no-any-return]


def _compute_log_marginal_densities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    Compute log(Σ_k w_k · kernel_k(x_n)) for each sample. Returns shape (N,).

    Values are non-normalised (see _compute_log_joint_densities), so the result
    is valid for convergence checking but is not the true log-likelihood.
    """
    log_joint: np.ndarray = _compute_log_joint_densities(
        X, weights, means, concentrations
    )
    return np.logaddexp.reduce(log_joint, axis=1)  # type: ignore[no-any-return]


def _compute_posterior_densities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    Compute p(z=k | x_n, Θ) for all samples via log-sum-exp. Returns shape (N, K).
    """
    log_joint: np.ndarray = _compute_log_joint_densities(
        X, weights, means, concentrations
    )
    log_marginal: np.ndarray = np.logaddexp.reduce(log_joint, axis=1)
    log_resp: np.ndarray = log_joint - log_marginal[:, np.newaxis]
    return np.exp(log_resp)  # type: ignore[no-any-return]


def _compute_responsibilities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    EM-specific alias for _compute_posterior_densities. Returns shape (N, K).
    """
    return _compute_posterior_densities(X, weights, means, concentrations)  # type: ignore[no-any-return]


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
    loglik: float = float(
        _compute_log_marginal_densities(X, weights, means, concentrations).sum()
    )

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
