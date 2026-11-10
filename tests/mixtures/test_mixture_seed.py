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
    PKBMixture,
    PowerSphericalMixture,
    SphericalCauchyMixture,
    VonMisesFisherMixture,
    create_mean_initializer,
)

type Mixture = (
    SphericalCauchyMixture | PKBMixture | PowerSphericalMixture | VonMisesFisherMixture
)
type MixtureType = type[Mixture]

MIXTURE_TYPES: list[MixtureType] = [
    SphericalCauchyMixture,
    PKBMixture,
    PowerSphericalMixture,
    VonMisesFisherMixture,
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
    assert first.params_ is not None
    assert second.params_ is not None
    assert_array_equal(first.params_.weights, second.params_.weights)
    assert_array_equal(first.params_.means, second.params_.means)
    assert_array_equal(first.params_.concentrations, second.params_.concentrations)
    assert first.log_likelihood_ == second.log_likelihood_
    assert first.n_iter_ == second.n_iter_


@pytest.mark.parametrize("init", INIT_METHODS)
@pytest.mark.parametrize("mixture_type", MIXTURE_TYPES)
def test_same_seed_gives_same_fit(
    mixture_type: MixtureType,
    init: Literal["random", "kmeans"],
    data: NDArray[np.float64],
) -> None:
    first: Mixture = mixture_type(
        n_components=2, init=init, max_iter=MAX_ITER, seed=3
    ).fit(data)
    second: Mixture = mixture_type(
        n_components=2, init=init, max_iter=MAX_ITER, seed=3
    ).fit(data)

    _assert_same_fit(first, second)


@pytest.mark.parametrize("init", INIT_METHODS)
@pytest.mark.parametrize("mixture_type", MIXTURE_TYPES)
def test_refit_is_stable(
    mixture_type: MixtureType,
    init: Literal["random", "kmeans"],
    data: NDArray[np.float64],
) -> None:
    model: Mixture = mixture_type(n_components=2, init=init, max_iter=MAX_ITER, seed=3)
    first: Mixture = mixture_type.from_dict(model.fit(data).to_dict())  # type: ignore[attr-defined]
    second: Mixture = model.fit(data)

    _assert_same_fit(first, second)


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


@pytest.mark.parametrize("seed", [None, 0, 42])
@pytest.mark.parametrize("mixture_type", MIXTURE_TYPES)
def test_seed_serialises(mixture_type: MixtureType, seed: int | None) -> None:
    model: Mixture = mixture_type(n_components=2, seed=seed)

    restored: Mixture = mixture_type.from_dict(model.to_dict())  # type: ignore[attr-defined]

    assert model.to_dict()["seed"] == seed  # type: ignore[attr-defined]
    assert restored.seed == seed


@pytest.mark.parametrize("mixture_type", MIXTURE_TYPES)
def test_model_without_seed_key_loads(mixture_type: MixtureType) -> None:
    serialised: dict = mixture_type(n_components=2, seed=5).to_dict()  # type: ignore[attr-defined]
    del serialised["seed"]

    restored: Mixture = mixture_type.from_dict(serialised)  # type: ignore[attr-defined]

    assert restored.seed is None
