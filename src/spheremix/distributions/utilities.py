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
    Sample from the uniform distribution on the unit sphere S^{d-1}.

    Normalises standard normal draws, which gives directions that are uniform
    on the sphere for any dimension.

    Arguments
    ---------
    n_samples: Number of samples to draw.
    n_features: Ambient dimension d of the sphere S^{d-1}.
    rng: Random number generator. A fresh generator is created if not provided.

    Returns
    -------
    Unit vectors, shape (n_samples, n_features).
    """
    if rng is None:
        rng = np.random.default_rng()

    Z: np.ndarray = rng.normal(size=(n_samples, n_features))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return Z
