"""
Module for Poisson kernel-based distribution.
"""

import numpy as np
from scipy.special import gammaln


# TODO: Implement function to sample from PKBD


def pkbd_log_normalizing_constant(dim: int) -> float:
    """Calculates the PKBD log normalizing constant."""
    numerator: float = gammaln((dim + 1.0) / 2.0)
    denominator: float = np.log(2.0) + ((dim + 1.0) / 2.0) * np.log(np.pi)
    return numerator - denominator


def pkbd_log_kernel(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate log-kernel, i.e. un-normalized log-PDF values, of PKBD for all data
    points, given mean and concentration (mu, rho).

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
    log_kernel : ndarray of shape (n_samples,)
        Unnormalized log-densities values
    """

    X: np.ndarray = np.asarray(X, dtype=float)
    mu: np.ndarray = np.asarray(mu, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if X.shape[1] != mu.shape[0]:
        raise ValueError("X and mu have incompatible shapes.")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    mu_norm: np.ndarray = np.linalg.norm(mu)
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")

    if check_input:
        if not np.isclose(mu_norm, 1.0, atol=1e-7):
            raise ValueError("mu must have unit norm.")
        x_norms: np.ndarray = np.linalg.norm(X, axis=1)
        if not np.allclose(x_norms, 1.0, atol=1e-7):
            raise ValueError("Each row of X must have unit norm.")
    else:
        mu: np.ndarray = mu / max(mu_norm, eps)

    dim: int = X.shape[1]
    dot: np.ndarray = X @ mu

    numerator: float = 1.0 - rho**2
    denominator: np.ndarray = 1 + rho**2 - 2.0 * rho * dot

    log_numerator: float = np.log(numerator)
    log_denominator: np.ndarray = ((dim + 1.0) / 2.0) * np.log(denominator)

    return log_numerator - log_denominator


def pkbd_logpdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate log-PDF of PKBD for all data points, given mean and
    concentration (mu, rho).

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
        normalized log-densities values
    """

    X: np.ndarray = np.asarray(X, dtype=float)
    mu: np.ndarray = np.asarray(mu, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if X.shape[1] != mu.shape[0]:
        raise ValueError("X and mu have incompatible shapes.")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    mu_norm: np.ndarray = np.linalg.norm(mu)
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")

    if check_input:
        if not np.isclose(mu_norm, 1.0, atol=1e-7):
            raise ValueError("mu must have unit norm.")
        x_norms: np.ndarray = np.linalg.norm(X, axis=1)
        if not np.allclose(x_norms, 1.0, atol=1e-7):
            raise ValueError("Each row of X must have unit norm.")
    else:
        mu: np.ndarray = mu / max(mu_norm, eps)

    dim: int = X.shape[1]
    log_constant: float = pkbd_log_normalizing_constant(dim)
    log_kernel: np.ndarray = pkbd_log_kernel(
        X=X,
        mu=mu,
        rho=rho,
        check_input=check_input,
        eps=eps,
    )
    return log_constant + log_kernel


def pkbd_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input=True,
    eps=1e-12,
) -> np.ndarray:
    """
    Evaluate the PKBD for each row of X.

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

    log_pdf: np.ndarray = pkbd_logpdf(
        X=X,
        mu=mu,
        rho=rho,
        check_input=check_input,
        eps=eps,
    )
    return np.exp(log_pdf)
