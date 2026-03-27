"""
Module for Spherical Cauchy distribution.
"""

import numpy as np


def log_spherical_cauchy_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    Compute log-density (up to additive constant) of spherical Cauchy
    for all data points, given one component (mu, rho).

    X: (N, D) array, rows on unit sphere
    mu: (D,) unit vector
    rho: scalar in [0, 1)
    returns: (N,) log-pdf values (unnormalized)
    """
    # X and mu must be unit norm
    # ||x - rho mu||^2 = ||x||^2 + rho^2||mu||^2 - 2 rho x^T mu
    #                   = 1 + rho^2 - 2 rho (x^T mu)
    dot: np.ndarray = X @ mu  # (N,)
    sq_norm = 1.0 + rho**2 - 2.0 * rho * dot
    # avoid log of zero
    sq_norm = np.maximum(sq_norm, 1e-12)
    # log f(x) ∝ log(1 - rho^2) - (D-1)/2 * log(||x - rho mu||^2)
    D = X.shape[1]
    log_num = np.log(np.maximum(1.0 - rho**2, 1e-12))
    log_den = 0.5 * (D - 1) * np.log(sq_norm)
    return log_num - log_den
