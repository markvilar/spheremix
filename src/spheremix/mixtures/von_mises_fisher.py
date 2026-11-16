"""
Module for the von Mises-Fisher mixture model.
"""

import dataclasses

import numpy as np
import scipy.optimize

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

from spheremix.distributions import (
    VmfLogNormalizerMethod,
    von_mises_fisher_logpdf,
)

from .convergence import SphericalMixtureConvergence
from .em_fitter import FitProgressCallback, fit_mixture_em
from .initializer import MeanInitStrategy, create_mean_initializer
from .scoring import (
    compute_aic,
    compute_bic,
    compute_bici,
    compute_dic,
    compute_icl,
    compute_log_likelihoods,
    compute_responsibilities,
)


@dataclass(slots=True, frozen=True)
class VonMisesFisherMixtureParams:
    """
    The settings of a von Mises-Fisher mixture fit.

    Attributes
    ----------
    n_components:
        Number of mixture components, K, at least one.
    init:
        How the component means are initialised.
    max_iter:
        Maximum number of EM iterations, at least one.
    tol:
        Absolute tolerance of the change in log-likelihood.
    rtol:
        Relative tolerance of the change in log-likelihood.
    verbose:
        Whether to log the log-likelihood of each iteration.
    log_normalizer_method:
        Method of the log normalising constant, used in the densities and in
        the M-step concentration update.
    seed:
        Seed of the mean initialiser, or None for a random seed.
    """

    n_components: int
    init: Literal["random", "kmeans"] = "random"
    max_iter: int = 100
    tol: float = 1e-4
    rtol: float = 1e-6
    verbose: bool = False
    log_normalizer_method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1, got {self.n_components}"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")


@dataclass(slots=True, frozen=True)
class VonMisesFisherMixtureComponents:
    """
    The fitted components of a von Mises-Fisher mixture.

    Attributes
    ----------
    weights:
        Mixing weights, shape (K,), sums to 1.
    means:
        Component means (unit vectors), shape (K, D).
    concentrations:
        Concentration parameters kappa, shape (K,), values >= 0.
    log_normalizer_method:
        Method of the log normalising constant of the densities.
    """

    weights: np.ndarray
    means: np.ndarray
    concentrations: np.ndarray
    log_normalizer_method: VmfLogNormalizerMethod

    @property
    def component_count(self) -> int:
        """Number of components, K."""
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        """Dimension of the ambient space, D, of the unit vectors."""
        return int(self.means.shape[1])

    @property
    def free_parameter_count(self) -> int:
        """
        Number of free parameters, K * (D + 1) - 1: K - 1 weights, and D - 1
        for the unit mean and 1 for the concentration of each component.
        """
        return self.component_count * (self.dimension + 1) - 1

    def log_joint_densities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log w_k + log p(x_n | mu_k, kappa_k) of each sample and component.

        Arguments
        ---------
        X:
            Unit vectors, shape (N, D).

        Returns
        -------
        Log joint densities, shape (N, K).
        """
        return _compute_log_joint_densities(
            X,
            self.weights,
            self.means,
            self.concentrations,
            self.log_normalizer_method,
        )


@dataclass(slots=True, frozen=True)
class VonMisesFisherMixtureState:
    """
    The state of a von Mises-Fisher mixture.

    Attributes
    ----------
    params:
        Settings of the fit.
    components:
        Fitted components, params.n_components of them, or None if the
        mixture is not fitted.
    convergence:
        How EM converged to the components, or None if the mixture is not
        fitted.
    """

    params: VonMisesFisherMixtureParams
    components: VonMisesFisherMixtureComponents | None = None
    convergence: SphericalMixtureConvergence | None = None

    def __post_init__(self) -> None:
        if (self.components is None) != (self.convergence is None):
            raise ValueError(
                "components and convergence must both be set or both be None"
            )
        if (
            self.components is not None
            and self.components.component_count != self.params.n_components
        ):
            raise ValueError(
                f"components hold {self.components.component_count} components, "
                f"but params.n_components is {self.params.n_components}"
            )

    @property
    def is_fitted(self) -> bool:
        """Whether the components and convergence are set."""
        return self.components is not None and self.convergence is not None


class VonMisesFisherMixture:
    """
    A mixture of von Mises-Fisher distributions on the unit hypersphere.

    Every method returns a new mixture or a result, and leaves this mixture
    unchanged.

    Attributes
    ----------
    params_type:
        Params type of the mixture.
    components_type:
        Components type of the mixture.
    state_type:
        State type of the mixture.
    """

    params_type: type[VonMisesFisherMixtureParams] = VonMisesFisherMixtureParams
    components_type: type[VonMisesFisherMixtureComponents] = (
        VonMisesFisherMixtureComponents
    )
    state_type: type[VonMisesFisherMixtureState] = VonMisesFisherMixtureState

    def __init__(self, params: VonMisesFisherMixtureParams) -> None:
        self._state: VonMisesFisherMixtureState = VonMisesFisherMixtureState(
            params=params
        )

    @classmethod
    def from_state(cls, state: VonMisesFisherMixtureState) -> Self:
        """Create a mixture from a state."""
        mixture: Self = cls(state.params)
        mixture._state = state
        return mixture

    @property
    def params(self) -> VonMisesFisherMixtureParams:
        """Settings of the fit."""
        return self._state.params

    @property
    def state(self) -> VonMisesFisherMixtureState:
        """State of the mixture."""
        return self._state

    @property
    def is_fitted(self) -> bool:
        """Whether the mixture is fitted."""
        return self._state.is_fitted

    def fit(
        self,
        X: np.ndarray,
        *,
        progress_callback: FitProgressCallback | None = None,
    ) -> Self:
        """
        Fit the mixture to data with EM.

        Arguments
        ---------
        X:
            Unit vectors, shape (N, D).
        progress_callback:
            Called with a FitProgress after each EM iteration, if given.

        Returns
        -------
        A new fitted mixture with the same params.
        """
        components, convergence = fit_von_mises_fisher_mixture(
            X, self.params, progress_callback=progress_callback
        )
        return type(self).from_state(
            VonMisesFisherMixtureState(
                params=self.params, components=components, convergence=convergence
            )
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the responsibilities of unit vectors X, shape (N, K)."""
        return compute_responsibilities(X, self._fitted_components())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most probable component of unit vectors X, shape (N,)."""
        return self.predict_proba(X).argmax(axis=1)  # type: ignore[no-any-return]

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return the log-likelihood of unit vectors X, shape (N,)."""
        return compute_log_likelihoods(X, self._fitted_components())

    def score(self, X: np.ndarray) -> float:
        """Return the mean log-likelihood of unit vectors X, shape (N, D)."""
        return float(self.score_samples(X).mean())

    def aic(self, X: np.ndarray) -> float:
        """Return the Akaike Information Criterion on unit vectors X."""
        return compute_aic(X, self._fitted_components())

    def bic(self, X: np.ndarray) -> float:
        """Return the Bayesian Information Criterion on unit vectors X."""
        return compute_bic(X, self._fitted_components())

    def icl(self, X: np.ndarray) -> float:
        """Return the Integrated Completed Likelihood on unit vectors X."""
        return compute_icl(X, self._fitted_components())

    def bici(self, X: np.ndarray) -> float:
        """Return the complete-data BIC on unit vectors X."""
        return compute_bici(X, self._fitted_components())

    def dic(self, X: np.ndarray) -> float:
        """Return the Deviance Information Criterion on unit vectors X."""
        return compute_dic(X, self._fitted_components())

    def _fitted_components(self) -> VonMisesFisherMixtureComponents:
        if self._state.components is None:
            raise RuntimeError("von Mises-Fisher mixture is not fitted; call fit first")
        return self._state.components


def fit_von_mises_fisher_mixture(
    X: np.ndarray,
    params: VonMisesFisherMixtureParams,
    *,
    progress_callback: FitProgressCallback | None = None,
) -> tuple[VonMisesFisherMixtureComponents, SphericalMixtureConvergence]:
    """
    Fit a von Mises-Fisher mixture to data with EM.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    params:
        Settings of the fit.
    progress_callback:
        Called with a FitProgress after each EM iteration, if given.

    Returns
    -------
    The fitted components, and how EM converged to them.
    """
    K: int = params.n_components
    D: int = X.shape[1]

    if D < 2:
        raise ValueError("number of features must be 2 or higher")

    mean_init: MeanInitStrategy = create_mean_initializer(params.init, seed=params.seed)
    initial_components: VonMisesFisherMixtureComponents = (
        VonMisesFisherMixtureComponents(
            weights=np.full(K, 1.0 / K),
            means=mean_init(X, K),
            concentrations=np.full(K, 1.0),
            log_normalizer_method=params.log_normalizer_method,
        )
    )
    return fit_mixture_em(
        X,
        initial_components,
        _maximization_step,
        params,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Private EM helpers
# ---------------------------------------------------------------------------


def _compute_log_joint_densities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    concentrations: np.ndarray,
    log_normalizer_method: VmfLogNormalizerMethod,
) -> np.ndarray:
    """
    Compute log p(x_n, z=k | Θ) = log w_k + log_pdf(x_n | μ_k, κ_k) for all
    samples and components, with the normaliser given by ``log_normalizer_method``.
    Returns shape (N, K).
    """
    K: int = len(weights)
    N: int = X.shape[0]

    log_joint: np.ndarray = np.empty((N, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + von_mises_fisher_logpdf(
            X, means[k], concentrations[k], method=log_normalizer_method
        )

    return log_joint  # type: ignore[no-any-return]


def _maximization_step(
    X: np.ndarray,
    resp: np.ndarray,
    components: VonMisesFisherMixtureComponents,
) -> VonMisesFisherMixtureComponents:
    """
    Update weights, means, and concentrations given responsibilities.

    Mean direction is updated analytically as the normalized weighted sum of
    data. Concentration is the maximiser of r_bar * kappa + log C_d(kappa) for
    the normaliser given by ``log_normalizer_method``, where r_bar is the mean
    resultant length.

    Returns the updated components.
    """
    dim: int = components.dimension
    log_normalizer_method: VmfLogNormalizerMethod = components.log_normalizer_method
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

    return dataclasses.replace(
        components,
        weights=new_weights,
        means=new_means,
        concentrations=new_concentrations,
    )


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
