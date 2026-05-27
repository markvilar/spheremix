"""
Public protocol interfaces for spherical models.
"""

from typing import Protocol, Self, runtime_checkable

import numpy as np


@runtime_checkable
class SphericalClusterer(Protocol):
    """
    Structural interface for spherical clustering models.

    Any class that implements the methods below with the expected signatures
    satisfies this protocol without requiring explicit inheritance.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit the model to unit-vector data X."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its most probable cluster."""
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return hard cluster assignments."""
        ...


@runtime_checkable
class SphericalMixtureModel(Protocol):
    """
    Structural interface for probabilistic spherical mixture models.

    Any class that implements the methods below with the expected signatures
    satisfies this protocol without requiring explicit inheritance.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit the model to unit-vector data X."""
        ...

    def is_fitted(self) -> bool:
        """Return True if the model has been fitted, False otherwise."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its most probable component."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample posterior probabilities over components."""
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return hard component assignments."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """Return the mean per-sample log-likelihood of X under the fitted model."""
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample log-likelihood of X, shape (N,)."""
        ...

    def aic(self, X: np.ndarray) -> float:
        """Return the Akaike Information Criterion for the fitted model on X."""
        ...

    def bic(self, X: np.ndarray) -> float:
        """Return the Bayesian Information Criterion for the fitted model on X."""
        ...

    def icl(self, X: np.ndarray) -> float:
        """Return the Integrated Completed Likelihood for the fitted model on X."""
        ...

    def bici(self, X: np.ndarray) -> float:
        """Return the complete-data BIC for the fitted model on X."""
        ...

    def dic(self, X: np.ndarray) -> float:
        """Return the Deviance Information Criterion for the fitted model on X."""
        ...
