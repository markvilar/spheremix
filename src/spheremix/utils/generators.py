"""
Random sample generators.
"""

import numpy as np


def orthogonal_sphere_modes(
    n: int,
    d: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """
    Return n well-spread unit vectors on S^{d-1}.

    When n <= d, uses the rows of a random orthogonal matrix (QR decomposition
    of a Gaussian matrix), which are exactly orthonormal — the maximally
    spread configuration. When n > d, the first d modes are orthonormal and
    the remainder are random unit vectors (unavoidable overlap once n > d).

    Parameters
    ----------
    n : int
        Number of mode vectors.
    d : int
        Ambient dimension.
    rng : np.random.Generator

    Returns
    -------
    modes : ndarray of shape (n, d)
        Unit-normalised mode vectors.
    """
    # QR decomposition of a (d x d) Gaussian matrix gives a uniform random
    # orthogonal matrix Q. Its rows form an orthonormal basis of R^d.
    G: np.ndarray = rng.standard_normal((d, d))
    Q: np.ndarray
    Q, _ = np.linalg.qr(G)  # Q: (d, d), orthogonal

    if n <= d:
        return Q[:n]  # first n orthonormal rows

    # n > d: take all d orthonormal rows, then pad with random unit vectors
    extra: np.ndarray = rng.standard_normal((n - d, d))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.vstack([Q, extra])  # shape (n, d)
