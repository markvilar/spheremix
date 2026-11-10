"""
Module for von Mises-Fisher (vMF) mixture model.
"""

import numpy as np
import scipy.optimize

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Self
from loguru import logger
from spheremix.distributions import VmfLogNormalizerMethod
from spheremix.distributions import von_mises_fisher_logkernel
from spheremix.distributions import von_mises_fisher_logpdf
from .base import SerializableMixin, _classification_entropy
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
    seed:
        Seed for the mean initialiser; the same seed gives the same fit.
        ``None`` draws a fresh unseeded generator on each fit.
    log_normalizer_method:
        Method for computing the log normalizing constant, used in the scoring
        functions and in the M-step concentration update.
        Defaults to AMOS_UPPER for numerical stability.
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
    log_normalizer_method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER
    seed: int | None = None

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
            seed=self.seed,
            log_normalizer_method=self.log_normalizer_method,
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

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log p(x | Θ), shape (N,). Uses logpdf (normalized) — the
        normalising constant varies with kappa and directly affects AIC/BIC."""
        p = self.get_params()
        K = self.n_components
        N = X.shape[0]
        log_joint = np.empty((N, K))
        for k in range(K):
            log_joint[:, k] = np.log(p.weights[k]) + von_mises_fisher_logpdf(
                X, p.means[k], p.concentrations[k], method=self.log_normalizer_method
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

    def _params_to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "rtol": self.rtol,
            "verbose": self.verbose,
            "seed": self.seed,
            "log_normalizer_method": self.log_normalizer_method.value,
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
            seed=data.get("seed"),
            log_normalizer_method=VmfLogNormalizerMethod(
                data.get(
                    "log_normalizer_method", VmfLogNormalizerMethod.AMOS_UPPER.value
                )
            ),
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
    seed: int | None = None,
    log_normalizer_method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER,
) -> tuple[VonMisesFisherMixtureParams, float, int]:
    """
    Fit a von Mises-Fisher mixture model via EM.

    The M-step uses an analytical mean update. The concentration update follows
    ``log_normalizer_method``: a closed form for the Amos bounds, and a
    root-finder on the exact Bessel ratio for ``EXACT``.

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
            X, resp, D, log_normalizer_method
        )
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
    log_normalizer_method: VmfLogNormalizerMethod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update weights, means, and concentrations given responsibilities.

    Mean direction is updated analytically as the normalized weighted sum of
    data. Concentration is the maximiser of r_bar * kappa + log C_d(kappa) for
    the normaliser given by ``log_normalizer_method``, where r_bar is the mean
    resultant length.

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
        new_concentrations[k] = _solve_concentration(r_bar, dim, log_normalizer_method)

    return new_weights, new_means, new_concentrations


def _solve_concentration(
    r_bar: float,
    dim: int,
    log_normalizer_method: VmfLogNormalizerMethod,
) -> float:
    """
    Concentration update for mean resultant length ``r_bar`` in [0, 1).

    Dispatches on the normaliser so that the M-step maximises the same
    objective as the normaliser used for scoring.
    """
    return _CONCENTRATION_SOLVER_REGISTRY[log_normalizer_method](r_bar, dim)


def _solve_concentration_amos_upper(r_bar: float, dim: int) -> float:
    """
    Closed-form kappa for the AMOS_UPPER normaliser.

    Solves kappa / (a + sqrt(kappa^2 + (a + 1)^2)) = r_bar, with a = (d - 1) / 2.
    """
    a: float = (dim - 1) / 2.0
    one_minus: float = 1.0 - r_bar * r_bar
    return float(r_bar * (a + np.sqrt(a * a + one_minus * dim)) / one_minus)


def _solve_concentration_amos_lower(r_bar: float, dim: int) -> float:
    """
    Closed-form kappa for the AMOS_LOWER normaliser.

    Solves kappa / (a + sqrt(kappa^2 + a^2)) = r_bar, with a = (d - 1) / 2.
    """
    return float(r_bar * (dim - 1) / (1.0 - r_bar * r_bar))


def _solve_concentration_exact(r_bar: float, dim: int) -> float:
    """
    Solve A_d(kappa) = r_bar for kappa with the exact Bessel ratio.

    The AMOS_LOWER and AMOS_UPPER closed forms bracket the root.
    """
    lower: float = _solve_concentration_amos_lower(r_bar, dim)
    upper: float = _solve_concentration_amos_upper(r_bar, dim)
    if lower <= 0.0:
        return 0.0

    def objective(kappa: float) -> float:
        return _bessel_ratio(kappa, dim) - r_bar

    if objective(upper) <= 0.0:
        return upper
    if objective(lower) >= 0.0:
        return lower
    return float(scipy.optimize.brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12))


def _bessel_ratio(
    kappa: float,
    dim: int,
    tol: float = 1e-15,
    max_terms: int = 100_000,
) -> float:
    """
    Ratio I_{d/2}(kappa) / I_{d/2-1}(kappa) by continued fraction (modified Lentz).

    Uses the Gauss continued fraction
    I_{v+1} / I_v = 1 / (2 (v + 1) / kappa + 1 / (2 (v + 2) / kappa + ...)),
    with v = d / 2 - 1, so no Bessel function is formed and nothing underflows.

    Raises
    ------
    RuntimeError
        If the fraction does not converge within ``max_terms`` terms.
    """
    if kappa <= 0.0:
        return 0.0
    order: float = dim / 2.0 - 1.0
    tiny: float = 1e-300
    fraction: float = tiny
    numerator_term: float = tiny
    denominator_term: float = 0.0
    for term in range(1, max_terms + 1):
        coefficient: float = 2.0 * (order + term) / kappa
        denominator_term = coefficient + denominator_term
        denominator_term = 1.0 / (denominator_term if denominator_term != 0.0 else tiny)
        numerator_term = coefficient + 1.0 / numerator_term
        numerator_term = numerator_term if numerator_term != 0.0 else tiny
        delta: float = numerator_term * denominator_term
        fraction *= delta
        if abs(delta - 1.0) < tol:
            return fraction
    raise RuntimeError(
        f"Bessel ratio continued fraction did not converge in {max_terms} terms "
        f"for kappa={kappa}, dim={dim}; increase max_terms"
    )


_CONCENTRATION_SOLVER_REGISTRY: dict[
    VmfLogNormalizerMethod, Callable[[float, int], float]
] = {
    VmfLogNormalizerMethod.EXACT: _solve_concentration_exact,
    VmfLogNormalizerMethod.AMOS_UPPER: _solve_concentration_amos_upper,
    VmfLogNormalizerMethod.AMOS_LOWER: _solve_concentration_amos_lower,
}


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
