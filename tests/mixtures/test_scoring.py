"""
Tests for the shared prediction, scoring and model selection functions.
"""

import numpy as np
import pytest

from dataclasses import dataclass

from numpy.testing import assert_allclose
from numpy.typing import NDArray

from spheremix.mixtures import (
    SphericalMixtureComponents,
    compute_aic,
    compute_bic,
    compute_bici,
    compute_dic,
    compute_icl,
    compute_log_likelihoods,
    compute_responsibilities,
)
from spheremix.mixtures.scoring import _classification_entropy

N_SAMPLES: int = 20
N_COMPONENTS: int = 3
N_FEATURES: int = 4
# Deliberately not K * (D + 1) - 1, so the criteria must read the property.
N_FREE_PARAMETERS: int = 7


@dataclass(slots=True, frozen=True)
class _FixedComponents:
    """Components whose log joint densities are a fixed array of shape (N, K)."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    log_joint: NDArray[np.float64]
    free_parameter_count: int

    @property
    def component_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def log_joint_densities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.log_joint


@pytest.fixture
def X() -> NDArray[np.float64]:
    points: NDArray[np.float64] = np.random.default_rng(0).standard_normal(
        (N_SAMPLES, N_FEATURES)
    )
    return points / np.linalg.norm(points, axis=1, keepdims=True)  # type: ignore[no-any-return]


@pytest.fixture
def components() -> _FixedComponents:
    rng: np.random.Generator = np.random.default_rng(1)
    return _FixedComponents(
        weights=np.full(N_COMPONENTS, 1.0 / N_COMPONENTS),
        means=np.eye(N_FEATURES)[:N_COMPONENTS],
        log_joint=rng.normal(-2.0, 1.0, (N_SAMPLES, N_COMPONENTS)),
        free_parameter_count=N_FREE_PARAMETERS,
    )


def _expected_log_likelihood(components: _FixedComponents) -> float:
    return float(np.log(np.exp(components.log_joint).sum(axis=1)).sum())


def _expected_responsibilities(components: _FixedComponents) -> NDArray[np.float64]:
    joint: NDArray[np.float64] = np.exp(components.log_joint)
    return joint / joint.sum(axis=1, keepdims=True)  # type: ignore[no-any-return]


def _expected_entropy(components: _FixedComponents) -> float:
    responsibilities: NDArray[np.float64] = _expected_responsibilities(components)
    return float(-(responsibilities * np.log(responsibilities)).sum())


def test_fixed_components_satisfy_protocol(components: _FixedComponents) -> None:
    assert isinstance(components, SphericalMixtureComponents)


def test_responsibilities(X: NDArray[np.float64], components: _FixedComponents) -> None:
    responsibilities: NDArray[np.float64] = compute_responsibilities(X, components)

    assert responsibilities.shape == (N_SAMPLES, N_COMPONENTS)
    assert_allclose(responsibilities, _expected_responsibilities(components))
    assert_allclose(responsibilities.sum(axis=1), 1.0)


def test_log_likelihoods(X: NDArray[np.float64], components: _FixedComponents) -> None:
    log_likelihoods: NDArray[np.float64] = compute_log_likelihoods(X, components)

    assert log_likelihoods.shape == (N_SAMPLES,)
    assert log_likelihoods.sum() == pytest.approx(_expected_log_likelihood(components))


def test_log_likelihoods_are_stable_for_very_negative_log_joint(
    X: NDArray[np.float64],
) -> None:
    components: _FixedComponents = _FixedComponents(
        weights=np.full(2, 0.5),
        means=np.eye(N_FEATURES)[:2],
        log_joint=np.full((N_SAMPLES, 2), -1e4),
        free_parameter_count=N_FREE_PARAMETERS,
    )

    assert_allclose(compute_log_likelihoods(X, components), -1e4 + np.log(2.0))
    assert_allclose(compute_responsibilities(X, components), 0.5)


def test_aic(X: NDArray[np.float64], components: _FixedComponents) -> None:
    expected: float = (
        -2.0 * _expected_log_likelihood(components) + 2.0 * N_FREE_PARAMETERS
    )

    assert compute_aic(X, components) == pytest.approx(expected)


def test_bic(X: NDArray[np.float64], components: _FixedComponents) -> None:
    expected: float = -2.0 * _expected_log_likelihood(
        components
    ) + N_FREE_PARAMETERS * np.log(N_SAMPLES)

    assert compute_bic(X, components) == pytest.approx(expected)


def test_icl(X: NDArray[np.float64], components: _FixedComponents) -> None:
    expected: float = compute_bic(X, components) + 2.0 * _expected_entropy(components)

    assert compute_icl(X, components) == pytest.approx(expected)
    assert compute_icl(X, components) >= compute_bic(X, components)


def test_bici(X: NDArray[np.float64], components: _FixedComponents) -> None:
    responsibilities: NDArray[np.float64] = _expected_responsibilities(components)
    expected: float = compute_bic(X, components) - 2.0 * float(
        np.log(responsibilities.max(axis=1)).sum()
    )

    assert compute_bici(X, components) == pytest.approx(expected)
    assert compute_bici(X, components) >= compute_bic(X, components)


def test_dic(X: NDArray[np.float64], components: _FixedComponents) -> None:
    expected: float = (
        -2.0 * _expected_log_likelihood(components)
        + 2.0 * _expected_entropy(components)
        + 2.0 * N_FREE_PARAMETERS
    )

    assert compute_dic(X, components) == pytest.approx(expected)


def test_classification_entropy_is_zero_for_hard_assignments() -> None:
    responsibilities: NDArray[np.float64] = np.eye(3)

    assert _classification_entropy(responsibilities) == pytest.approx(0.0)


def test_classification_entropy_of_uniform_assignments() -> None:
    responsibilities: NDArray[np.float64] = np.full((5, 4), 0.25)

    assert _classification_entropy(responsibilities) == pytest.approx(5 * np.log(4))
