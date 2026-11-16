"""
Public protocol interfaces for spherical clustering models.
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
