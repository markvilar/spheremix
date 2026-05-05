"""
Module for von Mises-Fisher distribution.
"""

import numpy as np

from dataclasses import dataclass
from scipy.special import gammaln, ive


@dataclass(slots=True, frozen=True)
class VonMisesFisherDistribution:
    """
    Von Mises-Fisher distribution on the unit sphere.

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
        return sample_von_mises_fisher(
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
        return von_mises_fisher_logkernel(X=X, mu=self.mu, kappa=self.kappa)

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
        return von_mises_fisher_logpdf(X=X, mu=self.mu, kappa=self.kappa)


def sample_von_mises_fisher(
    n_samples: int,
    mu: np.ndarray,
    kappa: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample from a von Mises-Fisher distribution on S^{d-1}.

    Uses the Wood (1994) / Ulrich (1984) rejection sampler. The projection
    t = mu^T x is sampled from its marginal via a Beta-based proposal; the
    remaining component is drawn uniformly from the (d-2)-sphere embedded in
    the hyperplane orthogonal to mu. This sampler is valid for all d >= 2 and
    kappa >= 0.

    Arguments
    ----------
    n_samples : int
        Number of samples.
    mu : array-like of shape (d,)
        Unit vector location / mode. Need not be pre-normalised.
    kappa : float
        Concentration parameter >= 0. kappa=0 yields the uniform distribution.
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

    if kappa == 0.0:
        Z: np.ndarray = rng.standard_normal((n_samples, d))
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)  # type: ignore[no-any-return]

    # Wood (1994) constants for the marginal t = mu^T x
    b: float = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1) ** 2)) / (d - 1)
    x0: float = (1.0 - b) / (1.0 + b)
    c: float = kappa * x0 + (d - 1) * np.log(1.0 - x0**2)

    # Batch rejection sampling for t = mu^T x
    t_samples: np.ndarray = np.empty(n_samples)
    count: int = 0
    while count < n_samples:
        batch: int = max(n_samples - count, 128)
        Z_beta: np.ndarray = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=batch)
        W: np.ndarray = (1.0 - (1.0 + b) * Z_beta) / (1.0 - (1.0 - b) * Z_beta)
        U: np.ndarray = rng.uniform(size=batch)
        log_ratio: np.ndarray = (
            kappa * W + (d - 1) * np.log(np.maximum(1.0 - x0 * W, 1e-300)) - c
        )
        accepted: np.ndarray = log_ratio >= np.log(U)
        accepted_W: np.ndarray = W[accepted]
        n_accept: int = min(len(accepted_W), n_samples - count)
        t_samples[count : count + n_accept] = accepted_W[:n_accept]
        count += n_accept

    # Sample uniformly on S^{d-2} in the hyperplane orthogonal to mu
    V: np.ndarray = rng.standard_normal((n_samples, d))
    V -= (V @ mu)[:, None] * mu
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # Combine radial and equatorial components
    t: np.ndarray = t_samples[:, None]
    X: np.ndarray = t * mu + np.sqrt(np.maximum(1.0 - t**2, 0.0)) * V
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X  # type: ignore[no-any-return]


def von_mises_fisher_log_normalizing_constant(dim: int, kappa: float) -> float:
    """
    Log normalizing constant of the von Mises-Fisher distribution.

    For kappa > 0:
        log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log(I_{d/2-1}(kappa))

    where I_v denotes the modified Bessel function of the first kind of order v,
    evaluated via ``scipy.special.ive`` (exponentially scaled) for numerical
    stability at large kappa.

    For kappa = 0, returns the log surface-area normalizer of the uniform distribution
    on S^{d-1}: log(Gamma(d/2) / (2 * pi^{d/2})).
    """
    v: float = dim / 2.0 - 1.0
    if kappa == 0.0:
        return float(gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(np.pi))
    # ive(v, kappa) = I_v(kappa) * exp(-kappa), so log I_v(kappa) = log(ive) + kappa
    log_bessel: float = float(np.log(ive(v, kappa)) + kappa)
    return float(v * np.log(kappa) - (dim / 2.0) * np.log(2.0 * np.pi) - log_bessel)


def von_mises_fisher_logkernel(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate unnormalized log-density (log-kernel) of the von Mises-Fisher
    distribution for all data points, given mode and concentration (mu, kappa).

    The kernel is exp(kappa * mu^T x), so the log-kernel is kappa * mu^T x.

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
        Numerical tolerance for norm checks.

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

    return kappa * (X @ mu)  # type: ignore[no-any-return]


def von_mises_fisher_logpdf(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate normalized log-density of the von Mises-Fisher distribution for
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
        Numerical tolerance for norm checks.

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
    log_constant: float = von_mises_fisher_log_normalizing_constant(dim, kappa)
    logkernel: np.ndarray = von_mises_fisher_logkernel(
        X=X,
        mu=mu,
        kappa=kappa,
        check_input=False,
        eps=eps,
    )
    return log_constant + logkernel  # type: ignore[no-any-return]


def von_mises_fisher_pdf(
    X: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    *,
    check_input: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate the von Mises-Fisher PDF for each row of X.

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
        Numerical tolerance for norm checks.

    Returns
    -------
    pdf : ndarray of shape (n_samples,)
        Density value for each row of X.
    """
    log_pdf: np.ndarray = von_mises_fisher_logpdf(
        X=X,
        mu=mu,
        kappa=kappa,
        check_input=check_input,
        eps=eps,
    )
    return np.exp(log_pdf)  # type: ignore[no-any-return]
