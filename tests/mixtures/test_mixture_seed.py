"""
Tests for seeded, reproducible mixture fits.
"""

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from typing import Literal

from spheremix.distributions import sample_uniform_sphere
from spheremix.mixtures import (
    PoissonKernelMixture,
    PoissonKernelMixtureParams,
    PowerSphericalMixture,
    PowerSphericalMixtureParams,
    SphericalCauchyMixture,
    SphericalCauchyMixtureParams,
    VonMisesFisherMixture,
    VonMisesFisherMixtureParams,
    create_mean_initializer,
)

type Mixture = (
    SphericalCauchyMixture
    | PoissonKernelMixture
    | PowerSphericalMixture
    | VonMisesFisherMixture
)
type MixtureType = type[Mixture]

# (mixture type, params type)
MIXTURE_TYPES: list[tuple[MixtureType, type]] = [
    (SphericalCauchyMixture, SphericalCauchyMixtureParams),
    (PoissonKernelMixture, PoissonKernelMixtureParams),
    (PowerSphericalMixture, PowerSphericalMixtureParams),
    (VonMisesFisherMixture, VonMisesFisherMixtureParams),
]
INIT_METHODS: list[Literal["random", "kmeans"]] = ["random", "kmeans"]

# D = 2 keeps the vMF fit away from the D >= 3 M-step failure tracked in #69.
N_SAMPLES: int = 60
N_FEATURES: int = 2
MAX_ITER: int = 5


@pytest.fixture
def data() -> NDArray[np.float64]:
    rng: np.random.Generator = np.random.default_rng(0)
    centres: NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, 1.0]])
    points: NDArray[np.float64] = centres[rng.integers(0, 2, N_SAMPLES)]
    points = points + 0.2 * rng.standard_normal((N_SAMPLES, N_FEATURES))
    return points / np.linalg.norm(points, axis=1, keepdims=True)  # type: ignore[no-any-return]


def _assert_same_fit(first: Mixture, second: Mixture) -> None:
    assert first.state.components is not None
    assert second.state.components is not None
    assert_array_equal(first.state.components.weights, second.state.components.weights)
    assert_array_equal(first.state.components.means, second.state.components.means)
    assert_array_equal(
        first.state.components.concentrations, second.state.components.concentrations
    )
    assert first.state.convergence == second.state.convergence


@pytest.mark.parametrize("init", INIT_METHODS)
@pytest.mark.parametrize(("mixture_type", "params_type"), MIXTURE_TYPES)
def test_same_seed_gives_same_fit(
    mixture_type: MixtureType,
    params_type: type,
    init: Literal["random", "kmeans"],
    data: NDArray[np.float64],
) -> None:
    first: Mixture = mixture_type(
        params_type(n_components=2, init=init, max_iter=MAX_ITER, seed=3)
    ).fit(data)
    second: Mixture = mixture_type(
        params_type(n_components=2, init=init, max_iter=MAX_ITER, seed=3)
    ).fit(data)

    _assert_same_fit(first, second)


@pytest.mark.parametrize("init", INIT_METHODS)
@pytest.mark.parametrize(("mixture_type", "params_type"), MIXTURE_TYPES)
def test_refit_is_stable(
    mixture_type: MixtureType,
    params_type: type,
    init: Literal["random", "kmeans"],
    data: NDArray[np.float64],
) -> None:
    model: Mixture = mixture_type(
        params_type(n_components=2, init=init, max_iter=MAX_ITER, seed=3)
    )

    _assert_same_fit(model.fit(data), model.fit(data))


@pytest.mark.parametrize("init", INIT_METHODS)
def test_different_seeds_give_different_starts(
    init: Literal["random", "kmeans"],
) -> None:
    data: NDArray[np.float64] = sample_uniform_sphere(
        50, 4, rng=np.random.default_rng(0)
    )
    first: NDArray[np.float64] = create_mean_initializer(init, seed=1)(data, 3)
    second: NDArray[np.float64] = create_mean_initializer(init, seed=2)(data, 3)

    assert not np.array_equal(first, second)
