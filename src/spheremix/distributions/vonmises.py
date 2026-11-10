"""
Module for von Mises-Fisher distribution.

Von Mises-Fisher distribution on S^{d-1} with location mu and concentration
kappa >= 0. Provides sampling, log-kernel, log-pdf, pdf, and log normalizing
constant.

References
----------
Fisher, R. A. (1953).
Dispersion on a sphere.
Proceedings of the Royal Society of London. Series A, 217(1130), 295-305.
https://doi.org/10.1098/rspa.1953.0064

Mardia, K. V., & Jupp, P. E. (2000).
Directional Statistics.
John Wiley & Sons.
"""

import enum
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from enum import auto
from scipy.special import gammaln, ive


class VmfLogNormalizerMethod(enum.StrEnum):
    """
    Method for computing the vMF log normalizing constant.

    Attributes
    ----------
    EXACT:
        Exact computation via the modified Bessel function (scipy.special.ive).
        Accurate where finite, but the Bessel function underflows when kappa is
        small relative to the dimension, giving a non-finite result (e.g. for
        kappa below about 0.1 * dim at dim = 1024).
    AMOS_UPPER:
        Upper bound on the log normalizing constant. Closed-form, no Bessel
        evaluation, stable across all dimensions and concentrations. Relative
        error is largest, up to about 1e-2, when kappa is close to the
        dimension. The default.
    AMOS_LOWER:
        Lower bound on the log normalizing constant. Closed-form, no Bessel
        evaluation, stable across all dimensions and concentrations.
    """

    EXACT = auto()
    AMOS_UPPER = auto()
    AMOS_LOWER = auto()


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


def _log_normalizer_exact(dim: int, kappa: float) -> float:
    """
    Exact log normalizing constant via the modified Bessel function of the first kind.

    For kappa > 0:
        log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log(I_{d/2-1}(kappa))

    where I_v is evaluated via ``scipy.special.ive`` (exponentially scaled) for
    numerical stability at large kappa.
    """
    v: float = dim / 2.0 - 1.0
    if kappa == 0.0:
        return float(gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(np.pi))
    log_bessel: float = float(np.log(ive(v, kappa)) + kappa)
    return float(v * np.log(kappa) - (dim / 2.0) * np.log(2.0 * np.pi) - log_bessel)


def _log_normalizer_amos_upper(dim: int, kappa: float) -> float:
    """
    Upper bound on the vMF log normalizing constant via Amos-type Bessel bounds.

    Derived by bounding the Bessel function ratio I_{d/2}/I_{d/2-1} from below
    and integrating the identity d/dkappa log C_d = -I_{d/2}/I_{d/2-1} in closed
    form.

    References
    ----------
    Amos, D. E. (1974). Computation of modified Bessel functions and their ratios.
    Mathematics of Computation, 28(125), 239-251. https://doi.org/10.2307/2005574
    """
    log_c0: float = float(
        gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(np.pi)
    )
    a: float = (dim - 1) / 2.0
    b: float = a + 1.0
    sqrt_val: float = float(np.sqrt(kappa**2 + b**2))
    integral: float = sqrt_val - b - a * float(np.log((a + sqrt_val) / (2.0 * a + 1.0)))
    return float(log_c0 - integral)


def _log_normalizer_amos_lower(dim: int, kappa: float) -> float:
    """
    Lower bound on the vMF log normalizing constant via Amos-type Bessel bounds.

    Derived by bounding the Bessel function ratio I_{d/2}/I_{d/2-1} from above
    and integrating the identity d/dkappa log C_d = -I_{d/2}/I_{d/2-1} in closed
    form.

    References
    ----------
    Amos, D. E. (1974). Computation of modified Bessel functions and their ratios.
    Mathematics of Computation, 28(125), 239-251. https://doi.org/10.2307/2005574
    """
    log_c0: float = float(
        gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(np.pi)
    )
    a: float = (dim - 1) / 2.0
    sqrt_val: float = float(np.sqrt(kappa**2 + a**2))
    integral: float = sqrt_val - a - a * float(np.log((a + sqrt_val) / (2.0 * a)))
    return float(log_c0 - integral)


_LOG_NORM_CONSTANT_FN_REGISTRY: dict[
    VmfLogNormalizerMethod, Callable[[int, float], float]
] = {
    VmfLogNormalizerMethod.EXACT: _log_normalizer_exact,
    VmfLogNormalizerMethod.AMOS_UPPER: _log_normalizer_amos_upper,
    VmfLogNormalizerMethod.AMOS_LOWER: _log_normalizer_amos_lower,
}


def von_mises_fisher_log_normalizing_constant(
    dim: int,
    kappa: float,
    method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER,
) -> float:
    """
    Log normalizing constant of the von Mises-Fisher distribution.

    Arguments
    ---------
    dim : int
        Dimension of the ambient space (data lies on S^{dim-1}).
    kappa : float
        Concentration parameter >= 0.
    method : VmfLogNormalizerMethod
        Computation method. EXACT uses the modified Bessel function;
        AMOS_UPPER and AMOS_LOWER are closed-form bounds with better
        numerical stability. Default is AMOS_UPPER.

    Returns
    -------
    float
        Log normalizing constant log C_d(kappa).
    """
    return _LOG_NORM_CONSTANT_FN_REGISTRY[method](dim, kappa)


def make_log_normalizer(
    method: VmfLogNormalizerMethod,
) -> Callable[[int, float], float]:
    """
    Return the log normalizing constant function for the given method.

    Useful for binding the method once and passing a plain callable through
    the EM loop or scoring functions without threading the enum at every call.

    Arguments
    ---------
    method : VmfLogNormalizerMethod
        Computation method to retrieve.

    Returns
    -------
    Callable[[int, float], float]
        Function with signature ``(dim, kappa) -> float``.
    """
    return _LOG_NORM_CONSTANT_FN_REGISTRY[method]


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
    method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER,
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
    method : VmfLogNormalizerMethod
        Method for computing the log normalizing constant. Default is AMOS_UPPER.
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
    log_constant: float = von_mises_fisher_log_normalizing_constant(
        dim, kappa, method=method
    )
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
    method: VmfLogNormalizerMethod = VmfLogNormalizerMethod.AMOS_UPPER,
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
    method : VmfLogNormalizerMethod
        Method for computing the log normalizing constant. Default is AMOS_UPPER.
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
        method=method,
        check_input=check_input,
        eps=eps,
    )
    return np.exp(log_pdf)  # type: ignore[no-any-return]
