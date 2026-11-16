"""
Tests for the shared EM loop, with stub steps and components.
"""

import dataclasses

import numpy as np
import pytest

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from spheremix.mixtures import (
    EMSettings,
    FitProgress,
    PoissonKernelMixture,
    PoissonKernelMixtureParams,
    PowerSphericalMixture,
    PowerSphericalMixtureParams,
    SphericalCauchyMixture,
    SphericalCauchyMixtureParams,
    SphericalMixtureComponents,
    SphericalMixtureConvergence,
    VonMisesFisherMixture,
    VonMisesFisherMixtureParams,
    compute_log_likelihoods,
    fit_mixture_em,
)

N_SAMPLES: int = 10
N_FEATURES: int = 2


@dataclass(slots=True, frozen=True)
class _Settings:
    max_iter: int = 5
    tol: float = 0.0
    rtol: float = 0.0
    verbose: bool = False


@dataclass(slots=True, frozen=True)
class _StubComponents:
    """Components whose log joint densities are a constant `level` everywhere."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    level: float
    free_parameter_count: int = 1

    @property
    def component_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def log_joint_densities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full((X.shape[0], self.component_count), self.level)


class _RecordingMStep:
    """An M-step that raises the level by `steps[i]` on its i-th call."""

    def __init__(self, steps: list[float], calls: list[str]) -> None:
        self.steps: list[float] = steps
        self.calls: list[str] = calls

    def __call__(
        self,
        X: NDArray[np.float64],
        responsibilities: NDArray[np.float64],
        components: _StubComponents,
    ) -> _StubComponents:
        step: float = self.steps[min(self.calls.count("M"), len(self.steps) - 1)]
        self.calls.append("M")
        return dataclasses.replace(components, level=components.level + step)


@pytest.fixture
def X() -> NDArray[np.float64]:
    angles: NDArray[np.float64] = np.linspace(0.0, np.pi, N_SAMPLES)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


@pytest.fixture
def components() -> _StubComponents:
    return _StubComponents(
        weights=np.full(2, 0.5), means=np.eye(N_FEATURES), level=-1.0
    )


def _recording_e_step(calls: list[str]):  # type: ignore[no-untyped-def]
    def expectation(
        X: NDArray[np.float64], components: _StubComponents
    ) -> NDArray[np.float64]:
        calls.append("E")
        return np.full((X.shape[0], components.component_count), 0.5)

    return expectation


def test_steps_alternate_starting_with_expectation(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    calls: list[str] = []

    fit_mixture_em(
        X,
        components,
        _RecordingMStep([1.0], calls),
        _Settings(max_iter=3),
        expectation=_recording_e_step(calls),
    )

    assert calls == ["E", "M", "E", "M", "E", "M"]


def test_stops_at_max_iter_without_converging(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    _, convergence = fit_mixture_em(
        X, components, _RecordingMStep([1.0], []), _Settings(max_iter=4)
    )

    assert convergence.n_iter == 4
    assert convergence.converged is False


@pytest.mark.parametrize(
    "settings", [_Settings(tol=1e-3), _Settings(rtol=1e-3)], ids=["tol", "rtol"]
)
def test_stops_when_change_within_tolerance(
    X: NDArray[np.float64],
    components: _StubComponents,
    settings: _Settings,
) -> None:
    # The level stops changing from the second M-step on.
    _, convergence = fit_mixture_em(
        X, components, _RecordingMStep([1.0, 0.0], []), settings
    )

    assert convergence.n_iter == 2
    assert convergence.converged is True


def test_log_likelihood_belongs_to_returned_components(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    fitted, convergence = fit_mixture_em(
        X, components, _RecordingMStep([1.0], []), _Settings(max_iter=3)
    )

    assert fitted.level == pytest.approx(2.0)
    assert convergence.log_likelihood == pytest.approx(
        compute_log_likelihoods(X, fitted).sum()
    )
    assert isinstance(convergence, SphericalMixtureConvergence)


def test_rejects_dimension_mismatch(components: _StubComponents) -> None:
    X: NDArray[np.float64] = np.eye(3)

    with pytest.raises(ValueError, match=r"must have shape \(N, 2\)"):
        fit_mixture_em(X, components, _RecordingMStep([1.0], []), _Settings())


@pytest.mark.parametrize(
    "params_type",
    [
        SphericalCauchyMixtureParams,
        PoissonKernelMixtureParams,
        PowerSphericalMixtureParams,
        VonMisesFisherMixtureParams,
    ],
)
def test_mixture_params_satisfy_settings(params_type: type) -> None:
    assert isinstance(params_type(n_components=2), EMSettings)


def test_stub_components_satisfy_protocol(components: _StubComponents) -> None:
    assert isinstance(components, SphericalMixtureComponents)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

MIXTURE_TYPES: list[tuple[type, type]] = [
    (SphericalCauchyMixture, SphericalCauchyMixtureParams),
    (PoissonKernelMixture, PoissonKernelMixtureParams),
    (PowerSphericalMixture, PowerSphericalMixtureParams),
    (VonMisesFisherMixture, VonMisesFisherMixtureParams),
]


class _CallbackError(Exception):
    pass


@pytest.mark.parametrize(
    ("steps", "settings"),
    [([1.0, 0.0], _Settings(tol=1e-3)), ([1.0], _Settings())],
    ids=["converged", "max_iter"],
)
def test_progress_reports_each_iteration(
    X: NDArray[np.float64],
    components: _StubComponents,
    steps: list[float],
    settings: _Settings,
) -> None:
    events: list[FitProgress] = []

    _, convergence = fit_mixture_em(
        X,
        components,
        _RecordingMStep(steps, []),
        settings,
        progress_callback=events.append,
    )

    assert [event.n_iter for event in events] == list(range(1, convergence.n_iter + 1))
    assert all(event.max_iter == settings.max_iter for event in events)
    assert all(event.converged is False for event in events[:-1])
    assert events[-1].n_iter == convergence.n_iter
    assert events[-1].log_likelihood == convergence.log_likelihood
    assert events[-1].converged is convergence.converged


def test_progress_reports_log_likelihood_change(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    events: list[FitProgress] = []

    fit_mixture_em(
        X,
        components,
        _RecordingMStep([1.0, 2.0, 0.5], []),
        _Settings(max_iter=4),
        progress_callback=events.append,
    )

    log_likelihoods: NDArray[np.float64] = np.array(
        [event.log_likelihood for event in events]
    )
    changes: NDArray[np.float64] = np.array(
        [event.log_likelihood_change for event in events]
    )
    assert np.isinf(changes[0])
    np.testing.assert_allclose(changes[1:], np.abs(np.diff(log_likelihoods)))


def test_progress_callback_exception_ends_fit(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    calls: list[str] = []
    error: _CallbackError = _CallbackError("stop")

    def on_progress(event: FitProgress) -> None:
        if event.n_iter == 2:
            raise error

    with pytest.raises(_CallbackError) as raised:
        fit_mixture_em(
            X,
            components,
            _RecordingMStep([1.0], calls),
            _Settings(),
            progress_callback=on_progress,
        )

    assert raised.value is error
    assert calls.count("M") == 2


def test_progress_callback_leaves_result_unchanged(
    X: NDArray[np.float64], components: _StubComponents
) -> None:
    without: tuple[Any, SphericalMixtureConvergence] = fit_mixture_em(
        X, components, _RecordingMStep([1.0], []), _Settings()
    )
    with_callback: tuple[Any, SphericalMixtureConvergence] = fit_mixture_em(
        X,
        components,
        _RecordingMStep([1.0], []),
        _Settings(),
        progress_callback=lambda event: None,
    )

    assert with_callback == without


@pytest.mark.parametrize(
    ("mixture_type", "params_type"),
    MIXTURE_TYPES,
    ids=[mixture_type.__name__ for mixture_type, _ in MIXTURE_TYPES],
)
def test_mixture_fit_reports_progress(
    mixture_type: type,
    params_type: type,
) -> None:
    points: NDArray[np.float64] = np.random.default_rng(0).standard_normal((20, 3))
    sphere_points: NDArray[np.float64] = points / np.linalg.norm(
        points, axis=1, keepdims=True
    )
    events: list[FitProgress] = []

    fitted: Any = mixture_type(params_type(n_components=2, max_iter=3, seed=0)).fit(
        sphere_points, progress_callback=events.append
    )
    convergence: SphericalMixtureConvergence = fitted.state.convergence

    assert [event.n_iter for event in events] == list(range(1, len(events) + 1))
    assert events[-1].n_iter == convergence.n_iter
    assert events[-1].log_likelihood == convergence.log_likelihood
    assert events[-1].converged is convergence.converged
