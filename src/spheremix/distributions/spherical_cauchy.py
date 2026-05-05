"""
Module for Spherical Cauchy distribution.
"""

import numpy as np

from dataclasses import dataclass
from scipy.special import gammaln
from .utilities import sample_uniform_sphere


@dataclass(slots=True, frozen=True)
class SphericalCauchyDistribution:
    """
    Spherical Cauchy distribution on the unit sphere.

    Parameters
    ----------
    mu : ndarray of shape (n_features,)
        Mean direction / mode (unit vector).
    rho : float
        Concentration parameter in [0, 1).
    """

    mu: np.ndarray
    rho: float

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generates random samples from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        samples: ndarray of shape (n_samples, n_features)
        """
        return sample_spherical_cauchy(
            n_samples=n_samples,
            mu=self.mu,
            rho=self.rho,
        )

    def logkernel(self, X: np.ndarray) -> np.ndarray:
        """
        Unnormalized log-density (log-kernel) at samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Points on the unit sphere.

        Returns
        -------
        logkernel : ndarray of shape (n_samples,)
        """
        return spherical_cauchy_logkernel(X=X, mu=self.mu, rho=self.rho)

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized log-density at samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Points on the unit sphere.

        Returns
        -------
        logpdf : ndarray of shape (n_samples,)
        """
        return spherical_cauchy_logpdf(X=X, mu=self.mu, rho=self.rho)


def sample_spherical_cauchy(
    n_samples: int,
    mu: np.ndarray,
    rho: float,
    rng: np.random.Generator | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Sample from a spherical Cauchy distribution on S^{d-1}.

    Uses a Möbius transform approach: uniform samples on S^{d-1} are pushed
    forward through a spherical Möbius transform parametrised by (mu, rho).
    If U is uniform on S^{d-1}, then

        X = ((1 - rho^2) * U + 2 * rho * (1 + rho * <U, mu>) * mu)
            / (1 + rho^2 + 2 * rho * <U, mu>)

    follows a spherical Cauchy distribution with mode mu and concentration
    rho. This construction is exact (no rejection step) and works in any
    dimension d >= 2.

    Arguments
    ----------
    n_samples : int
        Number of samples.
    mu : array-like of shape (d,)
        Unit vector location / mode. Need not be pre-normalised; the function
        normalises it internally.
    rho : float
        Concentration parameter in [0, 1). rho = 0 gives the uniform
        distribution; rho -> 1 concentrates mass around mu.
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh Generator is created via
        ``np.random.default_rng()``.
    eps : float
        Small constant added to the denominator to guard against
        floating-point division by zero when rho -> 1.

    Returns
    -------
    X : ndarray of shape (n_samples, d)
        Samples on the unit sphere S^{d-1}.

    Raises
    ------
    ValueError
        If ``mu`` is the zero vector.
    ValueError
        If ``rho`` is not in [0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    d: int = mu.shape[0]

    mu_norm: float = float(np.linalg.norm(mu))
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")
    mu = mu / mu_norm

    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    # Step 1: uniform sphere samples
    U: np.ndarray = sample_uniform_sphere(n_samples, d, rng=rng)

    # Step 2: spherical Möbius transform
    dot: np.ndarray = U @ mu  # shape (n_samples,)
    denom: np.ndarray = np.maximum(
        1.0 + rho**2 + 2.0 * rho * dot, eps
    )  # shape (n_samples,)

    X: np.ndarray = (
        (1.0 - rho**2) * U + 2.0 * rho * (1.0 + rho * dot)[:, None] * mu
    ) / denom[:, None]

    # Re-normalise to correct floating-point drift
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def spherical_cauchy_log_normalizing_constant(dim: int) -> float:
    """Calculates the spherical Cauchy log normalizing constant."""
    numerator: float = gammaln(dim / 2.0)
    denominator: float = np.log(2.0) + (dim / 2.0) * np.log(np.pi)
    return numerator - denominator


def spherical_cauchy_logkernel(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate unnormalized log-density for a spherical Cauchy distribution for all
    data points, given mean and concentration (mu, rho).

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
    logkernel: ndarray of shape (n_samples,)
        unnormalized log-density values
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

    dim: int = X.shape[1]
    dot: np.ndarray = X @ mu
    denom: np.ndarray = 1.0 + rho**2 - 2.0 * rho * dot
    denom = np.maximum(denom, eps)

    log_numerator: float = (dim - 1.0) * np.log(1.0 - rho**2)
    log_denominator: np.ndarray = (dim - 1.0) * np.log(denom)

    return log_numerator - log_denominator


def spherical_cauchy_logpdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate normalized log-density of spherical Cauchy for all data points, given mean
    and concentration (mu, rho).

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
        normalized log-density values
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

    dim: int = X.shape[1]
    log_constant: float = spherical_cauchy_log_normalizing_constant(dim)
    logkernel: np.ndarray = spherical_cauchy_logkernel(
        X=X,
        mu=mu,
        rho=rho,
        check_input=check_input,
        eps=eps,
    )

    return log_constant + logkernel


def spherical_cauchy_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
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
    return np.exp(log_pdf)  # type: ignore[no-any-return]
