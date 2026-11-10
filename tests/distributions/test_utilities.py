"""
Tests for the uniform sphere sampler.
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from spheremix.distributions import sample_uniform_sphere

# 2000 draws give a sample-mean norm of about 1/sqrt(2000) ~ 0.02 under uniform
# sampling, and at least ~0.85 for positive-orthant draws.
N_DRAWS: int = 2000
MEAN_NORM_THRESHOLD: float = 0.1


@pytest.mark.parametrize("n_samples", [1, 5])
@pytest.mark.parametrize("n_features", [2, 3, 16])
def test_sample_uniform_sphere_returns_unit_vectors(
    n_samples: int,
    n_features: int,
) -> None:
    rng: np.random.Generator = np.random.default_rng(0)

    samples: NDArray[np.float64] = sample_uniform_sphere(n_samples, n_features, rng=rng)

    assert samples.shape == (n_samples, n_features)
    assert_allclose(np.linalg.norm(samples, axis=1), 1.0)


@pytest.mark.parametrize("n_features", [2, 3, 16])
def test_sample_uniform_sphere_is_centred(n_features: int) -> None:
    rng: np.random.Generator = np.random.default_rng(0)

    samples: NDArray[np.float64] = sample_uniform_sphere(N_DRAWS, n_features, rng=rng)

    assert np.linalg.norm(samples.mean(axis=0)) < MEAN_NORM_THRESHOLD


def test_sample_uniform_sphere_is_reproducible() -> None:
    first: NDArray[np.float64] = sample_uniform_sphere(
        10, 4, rng=np.random.default_rng(7)
    )
    second: NDArray[np.float64] = sample_uniform_sphere(
        10, 4, rng=np.random.default_rng(7)
    )

    assert_array_equal(first, second)
