"""
Module for utility functionality for spherical probability distributions.
"""

import numpy as np


def sample_uniform_sphere(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample from a uniform distribution on S^{d-1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    Z: np.ndarray = rng.normal(size=(n_samples, n_features))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z
