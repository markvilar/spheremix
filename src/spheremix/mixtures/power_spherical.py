"""
Module for the Power Spherical mixture model.
"""

import dataclasses

import numpy as np
import scipy.optimize

from dataclasses import dataclass
from typing import Literal, Self

from spheremix.distributions import power_spherical_logpdf

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
class PowerSphericalMixtureParams:
    """
    The settings of a Power Spherical mixture fit.

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
    seed:
        Seed of the mean initialiser, or None for a random seed.
    """

    n_components: int
    init: Literal["random", "kmeans"] = "random"
    max_iter: int = 100
    tol: float = 1e-4
    rtol: float = 1e-6
    verbose: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1, got {self.n_components}"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")


@dataclass(slots=True, frozen=True)
class PowerSphericalMixtureComponents:
    """
    The fitted components of a Power Spherical mixture.

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
            X, self.weights, self.means, self.concentrations
        )


@dataclass(slots=True, frozen=True)
class PowerSphericalMixtureState:
    """
    The state of a Power Spherical mixture.

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

    params: PowerSphericalMixtureParams
    components: PowerSphericalMixtureComponents | None = None
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


class PowerSphericalMixture:
    """
    A mixture of Power Spherical distributions on the unit hypersphere.

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

    params_type: type[PowerSphericalMixtureParams] = PowerSphericalMixtureParams
    components_type: type[PowerSphericalMixtureComponents] = (
        PowerSphericalMixtureComponents
    )
    state_type: type[PowerSphericalMixtureState] = PowerSphericalMixtureState

    def __init__(self, params: PowerSphericalMixtureParams) -> None:
        self._state: PowerSphericalMixtureState = PowerSphericalMixtureState(
            params=params
        )

    @classmethod
    def from_state(cls, state: PowerSphericalMixtureState) -> Self:
        """Create a mixture from a state."""
        mixture: Self = cls(state.params)
        mixture._state = state
        return mixture

    @property
    def params(self) -> PowerSphericalMixtureParams:
        """Settings of the fit."""
        return self._state.params

    @property
    def state(self) -> PowerSphericalMixtureState:
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
        components, convergence = fit_power_spherical_mixture(
            X, self.params, progress_callback=progress_callback
        )
        return type(self).from_state(
            PowerSphericalMixtureState(
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

    def _fitted_components(self) -> PowerSphericalMixtureComponents:
        if self._state.components is None:
            raise RuntimeError("Power Spherical mixture is not fitted; call fit first")
        return self._state.components


def fit_power_spherical_mixture(
    X: np.ndarray,
    params: PowerSphericalMixtureParams,
    *,
    progress_callback: FitProgressCallback | None = None,
) -> tuple[PowerSphericalMixtureComponents, SphericalMixtureConvergence]:
    """
    Fit a Power Spherical mixture to data with EM.

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
    initial_components: PowerSphericalMixtureComponents = (
        PowerSphericalMixtureComponents(
            weights=np.full(K, 1.0 / K),
            means=mean_init(X, K),
            concentrations=np.full(K, 1.0),
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
) -> np.ndarray:
    """
    Compute log p(x_n, z=k | Θ) = log w_k + log_pdf(x_n | μ_k, θ_k) for all
    samples and components. Returns shape (N, K).
    """
    K: int = len(weights)
    N: int = X.shape[0]

    log_joint: np.ndarray = np.empty((N, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + power_spherical_logpdf(
            X, means[k], concentrations[k]
        )

    return log_joint  # type: ignore[no-any-return]


def _maximization_step(
    X: np.ndarray,
    resp: np.ndarray,
    components: PowerSphericalMixtureComponents,
) -> PowerSphericalMixtureComponents:
    """
    Update weights, means, and concentrations given responsibilities.
    Returns the updated components.
    """
    means: np.ndarray = components.means
    concentrations: np.ndarray = components.concentrations
    N, K = resp.shape
    new_weights: np.ndarray = resp.sum(axis=0) / N  # type: ignore[assignment]
    new_means: np.ndarray = means.copy()
    new_concentrations: np.ndarray = concentrations.copy()

    for k in range(K):

        def neg_weighted_loglik(params: np.ndarray, k: int = k) -> float:
            mu_raw = params[:-1]
            kappa = float(np.clip(params[-1], 0.0, np.inf))
            mu = mu_raw / np.linalg.norm(mu_raw)
            ll = power_spherical_logpdf(X, mu, kappa)
            return float(-np.dot(resp[:, k], ll))

        mu_init = means[k]
        kappa_init = concentrations[k]
        init_params = np.concatenate([mu_init, [kappa_init]])
        bounds = [(-np.inf, np.inf)] * mu_init.size + [(0.0, None)]

        res = scipy.optimize.minimize(
            neg_weighted_loglik,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
        )

        mu_opt = res.x[: mu_init.size]
        mu_opt = mu_opt / np.linalg.norm(mu_opt)
        kappa_opt = float(np.clip(res.x[mu_init.size], 0.0, np.inf))

        new_means[k] = mu_opt
        new_concentrations[k] = kappa_opt

    return dataclasses.replace(
        components,
        weights=new_weights,
        means=new_means,
        concentrations=new_concentrations,
    )
