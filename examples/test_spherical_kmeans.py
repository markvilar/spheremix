"""
Example script to showcase Spherical K-means.
"""

import numpy as np

from loguru import logger
from spheremix.distributions import PKBDistribution
from spheremix.cluster import SphericalKMeans
from spheremix.utils.generators import orthogonal_sphere_modes


def run_spherical_kmeans() -> np.ndarray:
    N_CLUSTERS: int = 10
    N_FEATURES: int = 100
    SAMPLES_PER_CLUSTER: int = 100

    RHO: float = 0.30

    # Generate 10 orthogonal samples on the unit sphere
    means: np.ndarray = orthogonal_sphere_modes(n=N_CLUSTERS, d=N_FEATURES)

    # Create multiple PKB distributions
    distributions: list[PKBDistribution] = [
        PKBDistribution(mu=mean, rho=RHO) for mean in means
    ]

    # Generate samples from each of the distributions
    samples: np.ndarray = np.concat(
        [distribution.sample(SAMPLES_PER_CLUSTER) for distribution in distributions]
    )

    # Create spherical K-means clusterer and fit to samples
    kmeans: SphericalKMeans = SphericalKMeans(n_clusters=10, verbose=True)
    kmeans.fit(samples)

    labels: np.ndarray = kmeans.predict(samples)

    logger.info(samples)
    logger.info(labels)

    raise NotImplementedError


def main() -> None:
    """Entrypoint."""
    run_spherical_kmeans()


if __name__ == "__main__":
    main()
