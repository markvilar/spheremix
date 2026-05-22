"""
Module for von Mises-Fisher (vMF) mixture model.
"""

import numpy as np
import scipy.optimize

from dataclasses import dataclass, field
from scipy.special import ive
from typing import ClassVar, Literal, Self
from loguru import logger
from spheremix.distributions import von_mises_fisher_logkernel
from spheremix.distributions import von_mises_fisher_logpdf
from .base import SerializableMixin
from .initializer import MeanInitStrategy, create_mean_initializer


# ---------------------------------------------------------------------------
# Mixture components — fitted parameters, replaced atomically on each fit
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class VonMisesFisherMixtureParams:
    """
    Fitted parameters for a von Mises-Fisher mixture model.

    Attributes
    ----------
    weights:
        Mixing weights, shape (K,), sums to 1.
    means:
        Component means (unit vectors), shape (K, D).
    concentrations:
        Concentration parameters kappa, shape (K,), values >= 0.
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
# Mixture model — facade; delegates EM to fit_von_mises_fisher_mixture()
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VonMisesFisherMixture(SerializableMixin):
    """
    Mixture model of von Mises-Fisher distributions.

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
    params_:
        Fitted mixture components; ``None`` until ``fit()`` is called.
    log_likelihood_:
        Observed-data log-likelihood at the end of EM.
    n_iter_:
        Number of EM iterations performed.
    """

    _SERIAL_VERSION: ClassVar[int] = 1

    n_components: int
    init: Literal["random", "kmeans"] = "random"
    max_iter: int = 100
    tol: float = 1e-4
    rtol: float = 1e-6
    verbose: bool = False

    params_: VonMisesFisherMixtureParams | None = field(default=None, init=False)
    log_likelihood_: float = field(default=-np.inf, init=False)
    n_iter_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """
        Run EM until convergence or ``max_iter``.
        Updates ``params_``, ``log_likelihood_``, and ``n_iter_``.
        """
        self.params_, self.log_likelihood_, self.n_iter_ = fit_von_mises_fisher_mixture(
            X,
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            rtol=self.rtol,
            verbose=self.verbose,
        )
        return self

    def get_params(self) -> VonMisesFisherMixtureParams:
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

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit the model and return hard component assignments.
        Returns an array of shape (N,) with integer component indices.
        """
        return self.fit(X).predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log p(x | Θ), shape (N,). Uses logpdf (normalized) — the
        normalising constant varies with kappa and directly affects AIC/BIC."""
        p = self.get_params()
        K = self.n_components
        N = X.shape[0]
        log_joint = np.empty((N, K))
        for k in range(K):
            log_joint[:, k] = np.log(p.weights[k]) + von_mises_fisher_logpdf(
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
        n_params = self.n_components * (X.shape[1] + 1) - 1
        return float(-2.0 * self.score_samples(X).sum() + n_params * np.log(X.shape[0]))

    def _params_to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "rtol": self.rtol,
            "verbose": self.verbose,
            "log_likelihood_": self.log_likelihood_,
            "n_iter_": self.n_iter_,
            "params_": self.params_.to_dict() if self.params_ is not None else None,
        }

    @classmethod
    def _params_from_dict(cls, data: dict) -> "VonMisesFisherMixture":
        model = cls(
            n_components=data["n_components"],
            init=data["init"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            rtol=data["rtol"],
            verbose=data["verbose"],
        )
        model.log_likelihood_ = data["log_likelihood_"]
        model.n_iter_ = data["n_iter_"]

        if data["params_"] is not None:
            p = data["params_"]
            model.params_ = VonMisesFisherMixtureParams(
                weights=np.array(p["weights"]),
                means=np.array(p["means"]),
                concentrations=np.array(p["concentrations"]),
            )

        return model


# ---------------------------------------------------------------------------
# Fitter function — owns all EM logic
# ---------------------------------------------------------------------------


def fit_von_mises_fisher_mixture(
    X: np.ndarray,
    n_components: int,
    init: Literal["random", "kmeans"] | MeanInitStrategy = "random",
    max_iter: int = 100,
    tol: float = 1e-4,
    rtol: float = 1e-6,
    verbose: bool = False,
) -> tuple[VonMisesFisherMixtureParams, float, int]:
    """
    Fit a von Mises-Fisher mixture model via EM.

    The M-step uses an analytical mean update and 1D root-finding for
    the concentration parameter.

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
        weights, means, concentrations = _maximization_step(X, resp, D)
        converged, log_likelihood = _check_convergence(
            X, weights, means, concentrations, i, log_likelihood, tol, rtol, verbose
        )
        n_iter = i + 1
        if converged:
            break

    components = VonMisesFisherMixtureParams(
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
    Compute log p(x_n, z=k | Θ) = log w_k + log_kernel(x_n | μ_k, κ_k).
    Returns shape (N, K). Uses logkernel (unnormalized) for EM efficiency.
    """
    K: int = len(weights)
    N: int = X.shape[0]

    log_joint: np.ndarray = np.empty((N, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + von_mises_fisher_logkernel(
            X, means[k], concentrations[k]
        )

    return log_joint  # type: ignore[no-any-return]


def _compute_log_marginal_densities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
) -> np.ndarray:
    """
    Compute log(Σ_k w_k · kernel_k(x_n)) for each sample. Returns shape (N,).
    Non-normalized; valid for convergence checking only.
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
    """EM alias for _compute_posterior_densities. Returns shape (N, K)."""
    return _compute_posterior_densities(X, weights, means, concentrations)  # type: ignore[no-any-return]


def _maximization_step(
    X: np.ndarray,
    resp: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update weights, means, and concentrations given responsibilities.

    Mean direction is updated analytically as the normalized weighted sum of
    data. Concentration is updated via 1D root-finding on A_d(kappa) = r_bar,
    where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) and r_bar is the
    mean resultant length.

    Returns updated (weights, means, concentrations).
    """
    N, K = resp.shape
    new_weights: np.ndarray = resp.sum(axis=0) / N  # type: ignore[assignment]
    new_means: np.ndarray = np.empty((K, dim))
    new_concentrations: np.ndarray = np.empty(K)

    for k in range(K):
        R_k: float = float(resp[:, k].sum())
        S_k: np.ndarray = (resp[:, k, np.newaxis] * X).sum(axis=0)
        norm_S_k: float = float(np.linalg.norm(S_k))

        # Analytical mean update: normalize weighted sum
        new_means[k] = S_k / max(norm_S_k, 1e-12)

        # Mean resultant length r_bar in [0, 1]; solve A_d(kappa) = r_bar
        r_bar: float = norm_S_k / max(R_k, 1e-12)
        r_bar = float(np.clip(r_bar, 0.0, 1.0 - 1e-9))
        new_concentrations[k] = _solve_concentration(r_bar, dim)

    return new_weights, new_means, new_concentrations


def _solve_concentration(r_bar: float, dim: int, kappa_max: float = 1e4) -> float:
    """
    Solve A_d(kappa) = r_bar for kappa, where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa).

    Uses scipy.optimize.brentq for 1D root-finding. Returns 0.0 when r_bar is
    effectively zero (uniform distribution).
    """
    if r_bar < 1e-10:
        return 0.0

    def objective(kappa: float) -> float:
        # ive cancels the e^{-kappa} factor, leaving the Bessel ratio
        return float(ive(dim / 2, kappa) / ive(dim / 2 - 1, kappa)) - r_bar

    # Guard: ensure the root is bracketed within [0, kappa_max]
    if objective(kappa_max) < 0:
        return kappa_max

    return float(scipy.optimize.brentq(objective, 0.0, kappa_max))


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


def _init_component_concentrations(K: int, init_value: float = 1.0) -> np.ndarray:
    """Return initial concentration parameters, all set to 1.0."""
    return np.full(shape=(K,), fill_value=init_value)
