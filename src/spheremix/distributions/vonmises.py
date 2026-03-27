"""
Module for the von Mises-Fisher distribution.
"""

import numpy as np


def log_von_mises_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """
    Compute log-density (up to additive constant) of von Mises-Fisher spherical
    distribution for all data points, given one component (mu, kappa).

    Arguments:
        X: (N, D) array, rows on unit sphere
        mu: (D,) array, mean unit vector
        kappa: float, scalar
    Returns:
        (N,) log-pdf values (unnormalized)
    """
    raise NotImplementedError
