"""
Module for the protocols of spherical mixtures and their state.
"""

import numpy as np

from typing import Protocol, Self, runtime_checkable

from .convergence import SphericalMixtureConvergence


@runtime_checkable
class SphericalMixtureParams(Protocol):
    """
    The settings of a spherical mixture fit.

    Attributes
    ----------
    n_components:
        Number of mixture components, K, at least one.
    """

    @property
    def n_components(self) -> int: ...


@runtime_checkable
class SphericalMixtureComponents(Protocol):
    """
    The fitted components of a spherical mixture, with any setting their
    densities depend on.

    Attributes
    ----------
    weights:
        Mixing weights, shape (K,), sums to 1.
    means:
        Component means (unit vectors), shape (K, D).
    component_count:
        Number of components, K.
    dimension:
        Dimension of the ambient space, D, of the unit vectors.
    """

    @property
    def weights(self) -> np.ndarray: ...

    @property
    def means(self) -> np.ndarray: ...

    @property
    def component_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def log_joint_densities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log w_k + log p(x_n | component k) of each sample and component.

        Arguments
        ---------
        X:
            Unit vectors, shape (N, D).

        Returns
        -------
        Log joint densities, shape (N, K).
        """
        ...


@runtime_checkable
class SphericalMixtureState[
    P: SphericalMixtureParams,
    C: SphericalMixtureComponents,
](Protocol):
    """
    The state of a spherical mixture.

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
    is_fitted:
        Whether the components and convergence are set.
    """

    @property
    def params(self) -> P: ...

    @property
    def components(self) -> C | None: ...

    @property
    def convergence(self) -> SphericalMixtureConvergence | None: ...

    @property
    def is_fitted(self) -> bool: ...


@runtime_checkable
class SphericalMixture[
    P: SphericalMixtureParams,
    C: SphericalMixtureComponents,
    S: SphericalMixtureState,
](Protocol):
    """
    A mixture of distributions on the unit hypersphere. Every method returns a
    new mixture or a result, and leaves this mixture unchanged.

    Attributes
    ----------
    params_type:
        Params type of the mixture.
    components_type:
        Components type of the mixture.
    state_type:
        State type of the mixture.
    params:
        Settings of the fit.
    state:
        State of the mixture.
    is_fitted:
        Whether the mixture is fitted.
    """

    params_type: type[P]
    components_type: type[C]
    state_type: type[S]

    def __init__(self, params: P) -> None: ...

    @classmethod
    def from_state(cls, state: S) -> Self:
        """Create a mixture from a state."""
        ...

    @property
    def params(self) -> P: ...

    @property
    def state(self) -> S: ...

    @property
    def is_fitted(self) -> bool: ...

    def fit(self, X: np.ndarray) -> Self:
        """Return a new mixture fitted to unit vectors X, shape (N, D)."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the responsibilities of each sample, shape (N, K)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most probable component of each sample, shape (N,)."""
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return the log-likelihood of each sample, shape (N,)."""
        ...

    def score(self, X: np.ndarray) -> float:
        """Return the mean log-likelihood of the samples."""
        ...

    def aic(self, X: np.ndarray) -> float:
        """Return the Akaike Information Criterion on X."""
        ...

    def bic(self, X: np.ndarray) -> float:
        """Return the Bayesian Information Criterion on X."""
        ...

    def icl(self, X: np.ndarray) -> float:
        """Return the Integrated Completed Likelihood on X."""
        ...

    def bici(self, X: np.ndarray) -> float:
        """Return the complete-data BIC on X."""
        ...

    def dic(self, X: np.ndarray) -> float:
        """Return the Deviance Information Criterion on X."""
        ...
