"""
Module for Poisson kernel-based distribution.
"""

import numpy as np

from dataclasses import dataclass
from scipy.special import gammaln


@dataclass(slots=True, frozen=True)
class PKBDistribution:
    """
    Poisson kernel–based distribution on the unit sphere.

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
        return sample_pkbd(
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
        return pkbd_logkernel(X=X, mu=self.mu, rho=self.rho)

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
        return pkbd_logpdf(X=X, mu=self.mu, rho=self.rho)


def sample_pkbd(
    n_samples: int,
    mu: np.ndarray,
    rho: float,
    rng: np.random.Generator | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Sample from a PKBD on S^{d-1}.

    Uses the ACG-envelope rejection sampler of Sablica, Hornik & Leydold (2023).

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
    d: int = mu.shape[0]

    # --- Reparametrize: lambda = 2*rho / (1 + rho^2) ---
    lam: float = 2.0 * rho / (1.0 + rho * rho)

    # --- Normalise mu; fall back to uniform if rho == 0 or mu == 0 ---
    norm_mu: float = float(np.linalg.norm(mu))
    if abs(lam) < eps or norm_mu < eps:
        Z: np.ndarray = rng.standard_normal((n_samples, d))
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)  # type: ignore[no-any-return]

    mu = mu / norm_mu

    # --- Find optimal ACG concentration parameter b ---
    p: float = float(d)
    coe: np.ndarray = np.array(
        [
            -4.0 * (p - 1.0),
            4.0 * p - lam**2 * (p - 2.0) ** 2,
            2.0 * p * (p - 2.0) * lam**2,
            -(p**2) * lam**2,
        ]
    )
    roots: np.ndarray = np.roots(coe)
    real_roots: np.ndarray = np.sort(
        np.real(roots[np.isreal(roots) | (np.abs(np.imag(roots)) < eps)])
    )
    b: float = float(real_roots[1])

    # --- Pre-compute scalar constants ---
    minus_log_M: float = np.log((1.0 + np.sqrt(1.0 - lam**2 / b)) / 2.0)
    b2: float = -1.0 + np.sqrt(1.0 / (1.0 - b))
    b1: float = b / (1.0 - b)

    # --- Rejection loop ---
    out: np.ndarray = np.empty((n_samples, d))
    count: int = 0

    while count < n_samples:
        # 1. Draw candidate from N(0, I_d)
        z: np.ndarray = rng.standard_normal(d)

        # 2. Compute projection onto mu and rescale to ACG proposal
        mu_t_z: float = float(mu @ z)
        norm_x: float = float(np.sqrt(z @ z + b1 * mu_t_z**2))
        mut_x: float = mu_t_z * (1.0 + b2) / norm_x

        # 3. Log-ratio: log[PKBD kernel / ACG density] + log-normalising offset
        log_pkbd: float = -np.log(1.0 - lam * mut_x + eps)
        log_acg: float = np.log(1.0 - b * mut_x**2 + eps)
        log_ratio: float = 0.5 * p * (log_pkbd + log_acg + minus_log_M)

        # 4. Accept / reject
        if np.log(rng.random()) < log_ratio:
            candidate: np.ndarray = (z + b2 * mu_t_z * mu) / norm_x
            out[count] = candidate
            count += 1

    return out


def pkbd_log_normalizing_constant(dim: int) -> float:
    """Calculates the PKBD log normalizing constant."""
    numerator: float = gammaln(dim / 2.0)
    denominator: float = np.log(2.0) + (dim / 2.0) * np.log(np.pi)
    return numerator - denominator


def pkbd_logkernel(
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
    logkernel : ndarray of shape (n_samples,)
        Unnormalized log-densities values
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

    mu_norm: float = float(np.linalg.norm(mu))
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")

    if check_input:
        if not np.isclose(mu_norm, 1.0, atol=1e-7):
            raise ValueError("mu must have unit norm.")
        x_norms: np.ndarray = np.linalg.norm(X, axis=1)
        if not np.allclose(x_norms, 1.0, atol=1e-7):
            raise ValueError("Each row of X must have unit norm.")
    else:
        mu = mu / max(mu_norm, eps)

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

    mu_norm: float = float(np.linalg.norm(mu))
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")

    if check_input:
        if not np.isclose(mu_norm, 1.0, atol=1e-7):
            raise ValueError("mu must have unit norm.")
        x_norms: np.ndarray = np.linalg.norm(X, axis=1)
        if not np.allclose(x_norms, 1.0, atol=1e-7):
            raise ValueError("Each row of X must have unit norm.")
    else:
        mu = mu / max(mu_norm, eps)

    dim: int = X.shape[1]
    log_constant: float = pkbd_log_normalizing_constant(dim)
    logkernel: np.ndarray = pkbd_logkernel(
        X=X,
        mu=mu,
        rho=rho,
        check_input=check_input,
        eps=eps,
    )
    return log_constant + logkernel


def pkbd_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    rho: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
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
    return np.exp(log_pdf)  # type: ignore[no-any-return]
