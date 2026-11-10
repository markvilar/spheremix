"""
Tests for the random mean initializer of the mixture models.
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from spheremix.distributions import sample_uniform_sphere
from spheremix.mixtures import create_mean_initializer
from spheremix.mixtures.initializer import (
    MeanInitStrategy,
    _random_mean_initializer,
)

# 2000 draws give a sample-mean norm of about 1/sqrt(2000) ~ 0.02 under uniform
# sampling, and at least ~0.85 for positive-orthant draws.
N_DRAWS: int = 2000
MEAN_NORM_THRESHOLD: float = 0.1


@pytest.mark.parametrize("n_components", [1, 4])
@pytest.mark.parametrize("n_features", [2, 3, 16])
def test_random_mean_initializer_returns_unit_vectors(
    n_components: int,
    n_features: int,
) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    data: NDArray[np.float64] = sample_uniform_sphere(10, n_features, rng=rng)
    initializer: MeanInitStrategy = create_mean_initializer("random")

    means: NDArray[np.float64] = initializer(data, n_components)

    assert means.shape == (n_components, n_features)
    assert_allclose(np.linalg.norm(means, axis=1), 1.0)


@pytest.mark.parametrize("n_features", [2, 3, 16])
def test_random_mean_initializer_is_centred(n_features: int) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    data: NDArray[np.float64] = sample_uniform_sphere(1, n_features, rng=rng)
    initializer: MeanInitStrategy = _random_mean_initializer(rng=rng)

    means: NDArray[np.float64] = initializer(data, N_DRAWS)

    assert np.linalg.norm(means.mean(axis=0)) < MEAN_NORM_THRESHOLD


def test_random_mean_initializer_is_reproducible() -> None:
    data: NDArray[np.float64] = sample_uniform_sphere(
        10, 4, rng=np.random.default_rng(0)
    )
    first: NDArray[np.float64] = _random_mean_initializer(rng=np.random.default_rng(7))(
        data, 3
    )
    second: NDArray[np.float64] = _random_mean_initializer(
        rng=np.random.default_rng(7)
    )(data, 3)

    assert_array_equal(first, second)
