"""
Implementation of the Spherical K-means algorithm.

Standalone functions implement the algorithm; the SphericalKMeans class
provides a scikit-learn-compatible interface that delegates to them.

Reference
---------
Dhillon, I. S., & Modha, D. S. (2001).
Concept decompositions for large sparse text data using clustering.
Machine Learning, 42(1), 143-175.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Literal, Self


@dataclass(slots=True)
class SphericalKMeans:
    """
    Interface for the Spherical K-means algorithm.
    """

    n_clusters: int
    init: Literal["kmeans++", "random"] = "kmeans++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    verbose: bool = False

    # Set by fit()
    cluster_means: np.ndarray | None = field(default=None, init=False)
    inertia_: float | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: None = None) -> Self:
        """
        Fits the model to the data an

        Arguments
        ---------
        X: samples, np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        Self: SphericalKMeans instance
        """

        labels: np.ndarray
        centers: np.ndarray
        inertia: float
        n_iter: int
        labels, centers, inertia, n_iter = _spherical_kmeans_fit(
            X,
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=self.rng,
            verbose=self.verbose,
        )

        self.cluster_means = centers
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster labels for the samples.

        Arguments
        ---------
        X: samples, np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels: np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        assert self.cluster_means is not None
        return _spherical_kmeans_predict(X=X, centers=self.cluster_means)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model to the samples, and predicts the sample cluster labels.

        Arguments
        ---------
        X: samples, np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels: np.ndarray of shape (n_samples,)
        """
        return self.fit(X).predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms X to a cluster-distance space, returning an array of shape
        (n_samples, n_clusters) where each value is the distance to a cluster centre.

        Arguments
        ---------
        X: samples, np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        distances: np.ndarray of shape (n_samples, n_clusters)
        """
        self._check_is_fitted()
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if self.cluster_means is None:
            raise RuntimeError(
                "This SphericalKMeans instance is not fitted yet. "
                "Call fit() before using this method."
            )


# ---------------------------------------------------------------------------
# Standalone algorithm functions
# ---------------------------------------------------------------------------


def _spherical_kmeans_init_centers(
    X: np.ndarray,
    n_clusters: int,
    init: Literal["random", "kmeans++"] = "kmeans++",
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """
    Initialise cluster centres on the unit sphere.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Unit-normalised data.
    n_clusters : int
        Number of cluster centres to initialise.
    init : {"random", "kmeans++"}
        Initialisation strategy.
        - "random"   : draw k rows of X uniformly at random.
        - "kmeans++" : probabilistic seeding that spreads centres apart
                       (Arthur & Vassilvitskii, 2007), using cosine distance.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Unit-normalised initial centres.
    """
    match init:
        case "random":
            return _init_centers_random(X, n_clusters, rng)
        case "kmeans++":
            return _init_centers_kmeans(X, n_clusters, rng)
        case _:
            raise NotImplementedError(f"invalid init strategy: {init}")


def _init_centers_random(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initializes the cluster centers as random samples.

    Returns
    -------
    centers: np.ndarray of shape (n_clusters, n_features)
    """
    n_samples: int = X.shape[0]
    idx: np.ndarray = rng.choice(n_samples, size=n_clusters, replace=False)
    return X[idx].copy()  # type: ignore[no-any-return]


def _init_centers_kmeans(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initializes the cluster centers using the K-means++ algorithm.

    Returns
    -------
    centers: np.ndarray of shape (n_clusters, n_features)
    """
    n_samples, _ = X.shape
    # kmeans++ seeding using cosine similarity
    # First centre chosen uniformly at random.

    centers: list[np.ndarray] = [X[rng.integers(n_samples)].copy()]

    for _ in range(1, n_clusters):
        # Cosine similarities between each sample and the nearest current centre.
        # similarity shape: (n_samples, len(centers))
        sims: np.ndarray = X @ np.array(centers).T  # (n_samples, k)
        nearest_sim: np.ndarray = sims.max(axis=1)  # (n_samples,)

        # Distance proxy: 1 - cos_sim  (in [0, 2] for unit vectors).
        # Sample next centre with probability proportional to this distance.
        dist: np.ndarray = 1.0 - nearest_sim
        dist = np.maximum(dist, 0.0)  # numerical guard
        total: float = float(dist.sum())
        if total == 0.0:
            probs: np.ndarray = np.ones(n_samples) / n_samples
        else:
            probs = dist / total

        next_idx: int = int(rng.choice(n_samples, p=probs))
        centers.append(X[next_idx].copy())

    return np.array(centers)


def _spherical_kmeans_fit(
    X: np.ndarray,
    n_clusters: int,
    *,
    init: Literal["random", "kmeans++"] = "kmeans++",
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Run spherical K-means n_init times and return the best result.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Unit-normalised data. Every row must satisfy ‖x‖ = 1.
    n_clusters : int
        Number of clusters.
    init : {"random", "kmeans++"}, default "kmeans++"
        Centre initialisation strategy passed to each Lloyd run.
    n_init : int, default 10
        Number of independent restarts. The run with the lowest inertia
        (sum of cosine distances) is returned.
    max_iter : int, default 300
        Maximum Lloyd iterations per restart.
    tol : float, default 1e-4
        Convergence tolerance on centre shift per restart.
    rng : np.random.Generator
        Random number generator. Each restart derives a child seed from
        this generator so restarts are independent but reproducible.
    verbose : bool, default False
        If True, print convergence information for every restart.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for the best run.
    centers : ndarray of shape (n_clusters, n_features)
        Unit-normalised cluster centres for the best run.
    inertia : float
        Sum of cosine distances for the best run.
    n_iter : int
        Number of iterations used by the best run.

    Raises
    ------
    ValueError
        If n_samples < n_clusters, n_init < 1, or max_iter < 1.
    """
    n_samples: int = X.shape[0]

    if n_samples < n_clusters:
        raise ValueError(f"n_samples={n_samples} must be >= n_clusters={n_clusters}.")
    if n_init < 1:
        raise ValueError(f"n_init={n_init} must be >= 1.")
    if max_iter < 1:
        raise ValueError(f"max_iter={max_iter} must be >= 1.")

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia: float = np.inf
    best_n_iter: int = 0

    # Draw n_init independent seeds upfront for reproducibility.
    seeds: np.ndarray = rng.integers(0, np.iinfo(np.int64).max, size=n_init)

    for i, seed in enumerate(seeds):
        child_rng: np.random.Generator = np.random.default_rng(int(seed))

        if verbose:
            print(f"\n--- Restart {i + 1}/{n_init} ---")

        labels, centers, inertia, n_iter = _spherical_kmeans_lloyd(
            X,
            n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            rng=child_rng,
            verbose=verbose,
        )

        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            best_n_iter = n_iter

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia, best_n_iter


def _spherical_kmeans_lloyd(
    X: np.ndarray,
    n_clusters: int,
    *,
    init: Literal["random", "kmeans++"] = "kmeans++",
    max_iter: int = 300,
    tol: float = 1e-4,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Single run of the spherical K-means Lloyd algorithm.

    The only modification to standard Lloyd's algorithm is that cluster
    centres are L2-normalised onto S^{d-1} after each M-step, replacing
    the Euclidean mean update with a mean-then-normalise step.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Unit-normalised data. Every row must satisfy ‖x‖ = 1.
    n_clusters : int
        Number of clusters.
    init : {"random", "kmeans++"}, default "kmeans++"
        Centre initialisation strategy.
    max_iter : int, default 300
        Maximum number of Lloyd iterations.
    tol : float, default 1e-4
        Convergence tolerance on the maximum cosine-distance shift of any
        centre between two consecutive iterations.
    rng : np.random.Generator
        Random number generator (must be provided by the caller).
    verbose : bool, default False
        If True, print per-iteration inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each sample.
    centers : ndarray of shape (n_clusters, n_features)
        Unit-normalised cluster centres.
    inertia : float
        Sum of cosine distances (1 - cos_sim) from each sample to its centre.
    n_iter : int
        Number of iterations actually performed.
    """
    n_samples: int
    n_features: int
    n_samples, n_features = X.shape

    # --- Initialise centres ---
    centers: np.ndarray = _spherical_kmeans_init_centers(
        X, n_clusters, init=init, rng=rng
    )

    labels: np.ndarray = np.empty(n_samples, dtype=np.intp)
    inertia: float = np.inf
    n_iter: int = 0

    for n_iter in range(1, max_iter + 1):
        # --- E-step: assign each sample to the nearest centre by cosine sim ---
        cos_sims: np.ndarray = X @ centers.T  # (n_samples, n_clusters)
        new_labels: np.ndarray = cos_sims.argmax(axis=1)  # (n_samples,)
        inertia = float((1.0 - cos_sims[np.arange(n_samples), new_labels]).sum())

        if verbose:
            print(f"Iteration {n_iter:3d}  inertia {inertia:.6f}")

        # --- M-step: recompute centres as normalised cluster means ---
        centers_old: np.ndarray = centers.copy()
        new_centers: np.ndarray = np.zeros((n_clusters, n_features))

        for k in range(n_clusters):
            mask: np.ndarray = new_labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: re-seed with a random sample.
                new_centers[k] = X[rng.integers(n_samples)]

        # Normalise onto the sphere.
        norms: np.ndarray = np.linalg.norm(new_centers, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # guard against zero norm
        centers = new_centers / norms

        labels = new_labels

        # --- Convergence check: max cosine-distance shift across all centres ---
        shift: float = float((1.0 - (centers * centers_old).sum(axis=1)).max())
        if shift <= tol:
            if verbose:
                print(
                    f"Converged at iteration {n_iter} (shift {shift:.2e} <= {tol:.2e})"
                )
            break

    return labels, centers, inertia, n_iter


def _spherical_kmeans_predict(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Assign each sample in X to its nearest cluster centre by cosine similarity.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Unit-normalised samples.
    centers : ndarray of shape (n_clusters, n_features)
        Unit-normalised cluster centres.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster index of the nearest centre for each sample.
    """
    cos_sims: np.ndarray = X @ centers.T  # (n_samples, n_clusters)
    return cos_sims.argmax(axis=1).astype(np.intp)  # type: ignore[no-any-return]


def _spherical_kmeans_transform(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Transform X to a cluster cosine-distance space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Unit-normalised samples.
    centers : ndarray of shape (n_clusters, n_features)
        Unit-normalised cluster centres.

    Returns
    -------
    distances : ndarray of shape (n_samples, n_clusters)
        Cosine distance (1 - cosine similarity) from each sample to each centre.
    """
    cos_sims: np.ndarray = X @ centers.T  # (n_samples, n_clusters)
    return 1.0 - cos_sims
