"""
Module for Spherical Cauchy distribution.
"""

import numpy as np
from scipy.special import gammaln


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

    Z = rng.normal(size=(n_samples, n_features))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return Z


def sample_spherical_cauchy(
    n_samples: int,
    mu: np.ndarray,
    rho: float,
    rng: np.random.Generator | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Sample from a spherical Cauchy distribution on S^{d-1}.

    Arguments
    ----------
    n_samples : int
        Number of samples.
    mu : array-like of shape (d,)
        Unit vector location / mode.
    rho : float
        Concentration parameter in [0, 1).
    rng : np.random.Generator, optional
        Random number generator.
    eps : float
        Numerical stability constant.

    Returns
    -------
    X : ndarray of shape (n_samples, d)
        Samples on the unit sphere.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    d = mu.shape[0]

    mu_norm = np.linalg.norm(mu)
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")
    mu = mu / mu_norm

    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    # Step 1: uniform sphere samples
    U = sample_uniform_sphere(n_samples, d, rng=rng)

    # Step 2: spherical Möbius transform
    # This construction is described as an efficient sampling route
    # for the spherical Cauchy family.
    dot = U @ mu
    denom = 1.0 + rho**2 + 2.0 * rho * dot
    denom = np.maximum(denom, eps)

    X = ((1.0 - rho**2) * U + 2.0 * rho * (1.0 + rho * dot)[:, None] * mu) / denom[
        :, None
    ]

    # Re-normalize to control floating-point drift
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def spherical_cauchy_logpdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input=True,
    eps=1e-12,
) -> np.ndarray:
    """
    Compute log-density (up to additive constant) of spherical Cauchy
    for all data points, given one component (mu, rho).

    Arguments
    ---------
    X : array-like of shape (n_samples, n_features)
        Samples on the unit sphere.
    mu : array-like of shape (n_features,)
        Mean direction / mode, expected to have unit norm.
    rho : float
        Concentration parameter in [0, 1).
    check_input : bool, default=True
        If True, validates shapes, rho range, and approximate unit norms.
    eps : float, default=1e-12
        Numerical tolerance for norm checks and denominator clipping.

    Returns
    -------
    logpdf : ndarray of shape (n_samples,)
        log-pdf values (unnormalized)
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if X.shape[1] != mu.shape[0]:
        raise ValueError("X and mu have incompatible shapes.")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    mu_norm = np.linalg.norm(mu)
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")

    if check_input:
        if not np.isclose(mu_norm, 1.0, atol=1e-7):
            raise ValueError("mu must have unit norm.")
        x_norms = np.linalg.norm(X, axis=1)
        if not np.allclose(x_norms, 1.0, atol=1e-7):
            raise ValueError("Each row of X must have unit norm.")
    else:
        mu = mu / max(mu_norm, eps)

    d: int = X.shape[1]
    dot: np.ndarray = X @ mu
    denom: np.ndarray = 1.0 + rho**2 - 2.0 * rho * dot
    denom: np.ndarray = np.maximum(denom, eps)

    log_surface_const = gammaln(d / 2.0) - np.log(2.0) - (d / 2.0) * np.log(np.pi)
    numerator_term: float = (d - 1.0) * np.log(1.0 - rho**2)
    denominator_term: float = (d - 1.0) * np.log(denom)

    return (log_surface_const + numerator_term - denominator_term)


def spherical_cauchy_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input=True,
    eps=1e-12,
) -> np.ndarray:
    """
    Evaluate the spherical Cauchy PDF for each row of X.

    Arguments
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples on the unit sphere.
    mu : array-like of shape (n_features,)
        Mean direction / mode, expected to have unit norm.
    rho : float
        Concentration parameter in [0, 1).
    check_input : bool, default=True
        If True, validates shapes, rho range, and approximate unit norms.
    eps : float, default=1e-12
        Numerical tolerance for norm checks and denominator clipping.

    Returns
    -------
    pdf : ndarray of shape (n_samples,)
        Density value for each row of X.
    """

    log_pdf: np.ndarray = spherical_cauchy_logpdf(
        X,
        mu,
        rho,
        check_input=check_input,
        eps=eps,
    )
    return np.exp(log_pdf)
