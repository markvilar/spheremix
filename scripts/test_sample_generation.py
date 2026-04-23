"""
Example script to showcase random sample generation from spherical distributions.
"""

import numpy as np

from spheremix.distributions import PKBDistribution
from spheremix.distributions import SphericalCauchyDistribution


def generate_samples_pkbd() -> None:
    mu: np.ndarray = np.array([1.0, 1.0, 1.0])
    mu: np.ndarray = mu / np.linalg.norm(mu)

    distribution: PKBDistribution = PKBDistribution(mu=mu, rho=0.30)
    samples: np.ndarray = distribution.sample(n_samples=100)

    sample_norms: np.ndarray = np.linalg.norm(samples, axis=1, keepdims=True)
    assert np.allclose(sample_norms, 1.0)


def generate_samples_spherical_cauchy() -> None:
    mu: np.ndarray = np.array([1.0, 1.0, 1.0])
    mu: np.ndarray = mu / np.linalg.norm(mu)

    distribution: SphericalCauchyDistribution = SphericalCauchyDistribution(
        mu=mu, rho=0.30
    )
    samples: np.ndarray = distribution.sample(n_samples=100)

    sample_norms: np.ndarray = np.linalg.norm(samples, axis=1, keepdims=True)
    assert np.allclose(sample_norms, 1.0)


def main() -> None:
    generate_samples_pkbd()
    generate_samples_spherical_cauchy()


if __name__ == "__main__":
    main()
