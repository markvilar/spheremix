"""
fit_spherical_kmeans.py

Generate synthetic samples from orthogonal spherical Cauchy modes,
cluster them with Spherical K-means, and log per-cluster statistics.

Usage
-----
    python examples/fit_spherical_kmeans.py
    python examples/fit_spherical_kmeans.py --n-clusters 5 --n-features 10 --n-samples 300
"""

import sys

import click
import numpy as np
from loguru import logger

from spheremix.cluster import SphericalKMeans
from spheremix.distributions import SphericalCauchyDistribution
from spheremix.utils.generators import orthogonal_sphere_modes

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
)


@click.command()
@click.option(
    "--n-clusters",
    "-k",
    default=3,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of clusters.",
)
@click.option(
    "--n-samples",
    "-n",
    default=200,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of samples to draw from each mode.",
)
@click.option(
    "--n-features",
    "-d",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
    help="Ambient dimension of the sphere S^{d-1}.",
)
@click.option(
    "--rho",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    help="Concentration parameter used when generating samples (0=uniform, →1=peaked).",
)
@click.option(
    "--init",
    default="kmeans++",
    show_default=True,
    type=click.Choice(["kmeans++", "random"]),
    help="Centre initialisation strategy.",
)
@click.option(
    "--n-init",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of random restarts; the best result is kept.",
)
@click.option(
    "--max-iter",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of Lloyd iterations per restart.",
)
@click.option(
    "--tol",
    default=1e-4,
    show_default=True,
    type=float,
    help="Convergence tolerance on centre shift.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Log per-iteration inertia during fitting.",
)
def main(
    n_clusters: int,
    n_samples: int,
    n_features: int,
    rho: float,
    init: str,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
    verbose: bool,
) -> None:
    """Cluster orthogonally-seeded sphere samples with Spherical K-means."""
    rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Generate orthogonal modes and draw samples
    # ------------------------------------------------------------------
    logger.info(
        f"Generating {n_clusters} orthogonal modes "
        f"on S^{{{n_features - 1}}} with rho={rho}"
    )
    modes: np.ndarray = orthogonal_sphere_modes(n_clusters, n_features, rng=rng)

    parts: list[np.ndarray] = []
    for k in range(n_clusters):
        dist: SphericalCauchyDistribution = SphericalCauchyDistribution(
            mu=modes[k], rho=rho
        )
        parts.append(dist.sample(n_samples=n_samples))

    X: np.ndarray = np.vstack(parts)
    N: int = X.shape[0]
    logger.info(f"Generated {N} samples total  (shape {X.shape})")

    # Shuffle so K-means does not see perfectly ordered blocks.
    X = X[rng.permutation(N)]

    # ------------------------------------------------------------------
    # 2. Fit Spherical K-means
    # ------------------------------------------------------------------
    logger.info(
        f"Fitting SphericalKMeans — "
        f"K={n_clusters}, init={init}, n_init={n_init}, max_iter={max_iter}"
    )
    model: SphericalKMeans = SphericalKMeans(
        n_clusters=n_clusters,
        init=init,  # type: ignore[arg-type]
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        rng=rng,
        verbose=verbose,
    )
    labels: np.ndarray = model.fit_predict(X)

    assert model.cluster_means is not None
    assert model.inertia_ is not None
    assert model.n_iter_ is not None

    logger.info(
        f"Fitting done — iterations: {model.n_iter_}, "
        f"total inertia: {model.inertia_:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Log per-cluster statistics
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Per-cluster statistics")
    logger.info("-" * 60)

    for k in range(n_clusters):
        mask: np.ndarray = labels == k
        n_assigned: int = int(mask.sum())
        centre_k: np.ndarray = model.cluster_means[k]

        logger.info(f"Cluster {k}:")
        logger.info(f"  centre      : {centre_k.round(4)}")
        logger.info(f"  n_assigned  : {n_assigned} ({100 * n_assigned / N:.1f}%)")

        if n_assigned == 0:
            logger.warning(f"  Cluster {k} has no assigned samples — skipping stats.")
            continue

        X_k: np.ndarray = X[mask]
        cos_sim: np.ndarray = X_k @ centre_k  # cosine similarity (unit vectors)
        cos_dist: np.ndarray = 1.0 - cos_sim  # cosine distance
        inertia_k: float = float(cos_dist.sum())

        logger.info(
            f"  cos-similarity : mean={cos_sim.mean():.4f}  "
            f"std={cos_sim.std():.4f}  "
            f"min={cos_sim.min():.4f}  "
            f"max={cos_sim.max():.4f}"
        )
        logger.info(f"  inertia        : {inertia_k:.4f}")


if __name__ == "__main__":
    main()
