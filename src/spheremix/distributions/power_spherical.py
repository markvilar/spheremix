"""
Module for Power Spherical distribution.

Implements functions to generate samples, calculate un-normalized and normalized
log-densities.

The Power Spherical distribution is described in the following paper:
https://doi.org/10.48550/arXiv.2006.04437
"""

import numpy as np

from dataclasses import dataclass
from scipy.special import gammaln


@dataclass(slots=True, frozen=True)
class PowerSphericalDistribution:
    """
    Power Spherical distribution on the unit sphere.

    Parameters
    ----------
    mu : ndarray of shape (n_features,)
        Mean direction / mode (unit vector).
    kappa : float
        Concentration parameter >= 0. kappa=0 gives the uniform distribution;
        larger kappa concentrates mass around mu.
    """

    mu: np.ndarray
    kappa: float

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generates random samples from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_features)
        """
        return sample_power_spherical(
            n_samples=n_samples,
            mu=self.mu,
            kappa=self.kappa,
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
        return power_spherical_logkernel(X=X, mu=self.mu, kappa=self.kappa)

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
        return power_spherical_logpdf(X=X, mu=self.mu, kappa=self.kappa)


def sample_power_spherical(
    n_samples: int,
    mu: np.ndarray,
    kappa: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample from a Power Spherical distribution on S^{d-1}.

    Decomposes the density via the projection t = mu^T x. The marginal of t
    satisfies p(t) ∝ (1+t)^kappa (1-t^2)^{(d-3)/2}, which after the change of
    variables u = (1+t)/2 becomes Beta(kappa+(d-1)/2, (d-1)/2). Given t, the
    remaining component is drawn uniformly from the (d-2)-sphere embedded in the
    hyperplane orthogonal to mu. This sampler is exact (no rejection step).

    Reference: De Cao & Aziz, "The Power Spherical distribution" (2020).

    Arguments
    ----------
    n_samples : int
        Number of samples.
    mu : array-like of shape (d,)
        Unit vector location / mode. Need not be pre-normalised.
    kappa : float
        Concentration parameter >= 0.
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh Generator is created via
        ``np.random.default_rng()``.

    Returns
    -------
    X : ndarray of shape (n_samples, d)
        Samples on the unit sphere S^{d-1}.

    Raises
    ------
    ValueError
        If ``mu`` is the zero vector.
    ValueError
        If ``kappa`` is negative.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    d: int = mu.shape[0]

    mu_norm: float = float(np.linalg.norm(mu))
    if mu_norm == 0:
        raise ValueError("mu must be non-zero.")
    mu = mu / mu_norm

    if kappa < 0.0:
        raise ValueError("kappa must satisfy kappa >= 0.")

    # Marginal t = 2u - 1 where u ~ Beta(kappa + (d-1)/2, (d-1)/2)
    alpha: float = kappa + (d - 1) / 2.0
    beta: float = (d - 1) / 2.0
    u: np.ndarray = rng.beta(alpha, beta, size=n_samples)
    t: np.ndarray = 2.0 * u - 1.0  # shape (n_samples,)

    # Uniform sample on S^{d-2} embedded in the hyperplane orthogonal to mu
    Z: np.ndarray = rng.standard_normal((n_samples, d))
    Z -= (Z @ mu)[:, None] * mu
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    # Combine radial and equatorial components
    X: np.ndarray = (
        t[:, None] * mu + np.sqrt(np.maximum(1.0 - t[:, None] ** 2, 0.0)) * Z
    )

    # Re-normalise to correct floating-point drift
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def power_spherical_log_normalizing_constant(dim: int, kappa: float) -> float:
    """
    Log normalizing constant of the Power Spherical distribution.

    Equals -log of the integral of the kernel (1 + mu^T x)^kappa over S^{d-1},
    i.e. -log C(kappa, d) where

        C(kappa, d) = 2^{kappa+d-1} * pi^{(d-1)/2}
                      * Gamma(kappa + (d-1)/2) / Gamma(kappa + d - 1).
    """
    return float(
        -(kappa + dim - 1) * np.log(2.0)
        - ((dim - 1) / 2.0) * np.log(np.pi)
        - gammaln(kappa + (dim - 1) / 2.0)
        + gammaln(kappa + dim - 1)
    )


def power_spherical_logkernel(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate unnormalized log-density (log-kernel) of the Power Spherical
    distribution for all data points, given mode and concentration (mu, kappa).

    The kernel is (1 + mu^T x)^kappa.

    Arguments
    ---------
    X : array-like of shape (n_samples, n_features)
        Samples on the unit sphere.
    mu : array-like of shape (n_features,)
        Mean direction / mode, expected to have unit norm.
    kappa : float
        Concentration parameter >= 0.
    check_input : bool, default=True
        If True, validates shapes, kappa range, and approximate unit norms.
    eps : float, default=1e-12
        Numerical tolerance for norm checks and argument clipping.

    Returns
    -------
    logkernel : ndarray of shape (n_samples,)
        Unnormalized log-density values.
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if X.shape[1] != mu.shape[0]:
        raise ValueError("X and mu have incompatible shapes.")
    if kappa < 0.0:
        raise ValueError("kappa must satisfy kappa >= 0.")

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

    dot: np.ndarray = X @ mu
    return kappa * np.log(np.maximum(1.0 + dot, eps))


def power_spherical_logpdf(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate normalized log-density of the Power Spherical distribution for
    all data points, given mode and concentration (mu, kappa).

    Arguments
    ---------
    X : array-like of shape (n_samples, n_features)
        Samples on the unit sphere.
    mu : array-like of shape (n_features,)
        Mean direction / mode, expected to have unit norm.
    kappa : float
        Concentration parameter >= 0.
    check_input : bool, default=True
        If True, validates shapes, kappa range, and approximate unit norms.
    eps : float, default=1e-12
        Numerical tolerance for norm checks and argument clipping.

    Returns
    -------
    logpdf : ndarray of shape (n_samples,)
        Normalized log-density values.
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if X.shape[1] != mu.shape[0]:
        raise ValueError("X and mu have incompatible shapes.")
    if kappa < 0.0:
        raise ValueError("kappa must satisfy kappa >= 0.")

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
    log_constant: float = power_spherical_log_normalizing_constant(dim, kappa)
    logkernel: np.ndarray = power_spherical_logkernel(
        X=X,
        mu=mu,
        kappa=kappa,
        check_input=check_input,
        eps=eps,
    )
    return log_constant + logkernel


def power_spherical_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate the Power Spherical PDF for each row of X.

    Arguments
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples on the unit sphere.
    mu : array-like of shape (n_features,)
        Mean direction / mode, expected to have unit norm.
    kappa : float
        Concentration parameter >= 0.
    check_input : bool, default=True
        If True, validates shapes, kappa range, and approximate unit norms.
    eps : float, default=1e-12
        Numerical tolerance for norm checks and argument clipping.

    Returns
    -------
    pdf : ndarray of shape (n_samples,)
        Density value for each row of X.
    """
    log_pdf: np.ndarray = power_spherical_logpdf(
        X=X,
        mu=mu,
        kappa=kappa,
        check_input=check_input,
        eps=eps,
    )
    return np.exp(log_pdf)  # type: ignore[no-any-return]
