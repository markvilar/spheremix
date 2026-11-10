"""
Mean initialization strategies for spherical mixture models.
"""

from __future__ import annotations

import numpy as np

from typing import Literal, Protocol

from spheremix.distributions import sample_uniform_sphere


class MeanInitStrategy(Protocol):
    """Callable that initializes K cluster means from data X."""

    def __call__(self, X: np.ndarray, K: int) -> np.ndarray:
        """
        Parameters
        ----------
        X : ndarray of shape (N, D)
            Unit-normalized data on the sphere.
        K : int
            Number of cluster means to initialize.

        Returns
        -------
        means : ndarray of shape (K, D)
            Unit-normalized initial cluster means.
        """
        ...


def _random_mean_initializer(
    rng: np.random.Generator | None = None,
) -> MeanInitStrategy:
    """
    Return an initializer that draws unit vectors uniformly on the sphere.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator. A fresh generator is created if not provided.
    """
    _rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def initialize(X: np.ndarray, K: int) -> np.ndarray:
        return sample_uniform_sphere(K, X.shape[1], rng=_rng)

    return initialize


def _kmeans_mean_initializer(
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> MeanInitStrategy:
    """
    Return an initializer that warm-starts from Spherical K-means centres.

    Parameters
    ----------
    n_init : int
        Number of K-means restarts; the run with the lowest inertia is kept.
    max_iter : int
        Maximum number of Lloyd iterations per restart.
    tol : float
        Convergence tolerance on centre shift.
    rng : np.random.Generator, optional
        Random number generator. A fresh generator is created if not provided.
    """
    _rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def initialize(X: np.ndarray, K: int) -> np.ndarray:
        from spheremix.cluster import SphericalKMeans

        model: SphericalKMeans = SphericalKMeans(
            n_clusters=K,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            rng=_rng,
        )
        model.fit(X)
        assert model.cluster_means is not None
        return model.cluster_means

    return initialize


def create_mean_initializer(method: Literal["random", "kmeans"]) -> MeanInitStrategy:
    """
    Return a mean initializer by name.

    Parameters
    ----------
    method : {"random", "kmeans"}
        ``"random"``  — unit vectors drawn uniformly on the sphere (fast, order-independent).
        ``"kmeans"``  — Spherical K-means warm start (slower, better starting point).

    Returns
    -------
    MeanInitStrategy
    """
    match method:
        case "random":
            return _random_mean_initializer()
        case "kmeans":
            return _kmeans_mean_initializer()
        case _:
            raise ValueError(
                f"Unknown init method '{method}'. Expected 'random' or 'kmeans'."
            )
