"""
Tests that the mixture fits are correct: regression against recorded fits, a
log-likelihood that never decreases, and recovery of known parameters.
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose
from numpy.typing import NDArray

from collections.abc import Callable
from dataclasses import dataclass

from spheremix.distributions import (
    sample_pkbd,
    sample_power_spherical,
    sample_spherical_cauchy,
)
from spheremix.mixtures import (
    PoissonKernelMixture,
    PoissonKernelMixtureParams,
    PowerSphericalMixture,
    PowerSphericalMixtureParams,
    SphericalCauchyMixture,
    SphericalCauchyMixtureParams,
    SphericalMixtureConvergence,
    VonMisesFisherMixture,
    VonMisesFisherMixtureParams,
)

type Mixture = (
    SphericalCauchyMixture
    | PoissonKernelMixture
    | PowerSphericalMixture
    | VonMisesFisherMixture
)
type Sampler = Callable[..., NDArray[np.float64]]

N_SAMPLES: int = 60
N_FEATURES: int = 3
MAX_ITER: int = 10
MONOTONE_MAX_ITER: int = 5
# The non-vMF M-steps run a numerical optimiser, so allow round-off.
MONOTONE_RTOL: float = 1e-8


@dataclass(frozen=True, slots=True)
class _RecordedFit:
    """
    A seeded fit, recorded from the per-module EM loops before #82 and, for the
    Poisson kernel, after the PKBD normalisation fix in #85.
    """

    weights: list[float]
    means: list[list[float]]
    concentrations: list[float]
    log_likelihood: float
    n_iter: int
    converged: bool
    # An L-BFGS-B M-step differs by about 1e-8 across machines, so only the
    # closed-form vMF M-step is compared at round-off.
    rtol: float = 1e-6


# (mixture type, params type, recorded fit)
REGRESSION_CASES: list[tuple[type, type, _RecordedFit]] = [
    (
        SphericalCauchyMixture,
        SphericalCauchyMixtureParams,
        _RecordedFit(
            weights=[0.5723717450006366, 0.4276282549993634],
            means=[
                [-0.07807420393968706, 0.05862498415952955, 0.9952223519955122],
                [0.9954279172010272, -0.08580549102661403, 0.04196045002745657],
            ],
            concentrations=[0.6376241734930274, 0.6478420116281426],
            log_likelihood=-94.55330520138905,
            n_iter=10,
            converged=False,
        ),
    ),
    (
        PoissonKernelMixture,
        PoissonKernelMixtureParams,
        _RecordedFit(
            weights=[0.5777646912425346, 0.42223530875746546],
            means=[
                [-0.07464156584888919, 0.04940062942950491, 0.9959860513378662],
                [0.9937102309060932, -0.10377238562017205, 0.042086446455562386],
            ],
            concentrations=[0.7225400626604768, 0.7384614445826527],
            log_likelihood=-96.86895247277357,
            n_iter=10,
            converged=False,
        ),
    ),
    (
        PowerSphericalMixture,
        PowerSphericalMixtureParams,
        _RecordedFit(
            weights=[0.5471704441805602, 0.45282955581943996],
            means=[
                [-0.12827172849100943, 0.10427715412789262, 0.9862416736261536],
                [0.9975056900120647, -0.06346072708574474, 0.03090525054587431],
            ],
            concentrations=[12.80073700749194, 10.08703352238515],
            log_likelihood=-91.94366282640627,
            n_iter=10,
            converged=False,
        ),
    ),
    (
        VonMisesFisherMixture,
        VonMisesFisherMixtureParams,
        _RecordedFit(
            weights=[0.5617124231869839, 0.43828757681301606],
            means=[
                [-0.108687856708034, 0.09704925199297422, 0.9893272423681753],
                [0.99768426175791, -0.06773166378686822, 0.006207701767691746],
            ],
            concentrations=[7.804986893504853, 7.239376028140311],
            log_likelihood=-85.71095380014707,
            n_iter=10,
            converged=False,
            rtol=1e-10,
        ),
    ),
]
MIXTURE_TYPES: list[tuple[type, type]] = [case[:2] for case in REGRESSION_CASES]

# (mixture type, params type, sampler, true concentrations, concentration rtol)
RECOVERY_CASES: list[object] = [
    (
        SphericalCauchyMixture,
        SphericalCauchyMixtureParams,
        sample_spherical_cauchy,
        (0.7, 0.8),
        0.1,
    ),
    (
        PoissonKernelMixture,
        PoissonKernelMixtureParams,
        sample_pkbd,
        (0.7, 0.8),
        0.1,
    ),
    (
        PowerSphericalMixture,
        PowerSphericalMixtureParams,
        sample_power_spherical,
        (10.0, 20.0),
        # The power spherical concentration estimate is noisy at this sample size.
        0.25,
    ),
]
N_PER_COMPONENT: int = 100
RECOVERY_MAX_ITER: int = 15


@pytest.fixture
def data() -> NDArray[np.float64]:
    rng: np.random.Generator = np.random.default_rng(7)
    centres: NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    points: NDArray[np.float64] = centres[rng.integers(0, 2, N_SAMPLES)]
    points = points + 0.4 * rng.standard_normal((N_SAMPLES, N_FEATURES))
    return points / np.linalg.norm(points, axis=1, keepdims=True)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mixture_type", "params_type", "recorded"), REGRESSION_CASES)
def test_fit_matches_recorded_fit(
    mixture_type: type,
    params_type: type,
    recorded: _RecordedFit,
    data: NDArray[np.float64],
) -> None:
    fitted: Mixture = mixture_type(
        params_type(n_components=2, max_iter=MAX_ITER, seed=0)
    ).fit(data)
    convergence: SphericalMixtureConvergence | None = fitted.state.convergence

    assert fitted.state.components is not None
    assert convergence is not None
    assert_allclose(
        fitted.state.components.weights, recorded.weights, rtol=recorded.rtol
    )
    assert_allclose(fitted.state.components.means, recorded.means, rtol=recorded.rtol)
    assert_allclose(
        fitted.state.components.concentrations,
        recorded.concentrations,
        rtol=recorded.rtol,
    )
    assert convergence.log_likelihood == pytest.approx(
        recorded.log_likelihood, rel=recorded.rtol
    )
    assert convergence.n_iter == recorded.n_iter
    assert convergence.converged is recorded.converged


# ---------------------------------------------------------------------------
# Monotone log-likelihood
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mixture_type", "params_type"), MIXTURE_TYPES)
def test_log_likelihood_never_decreases(
    mixture_type: type,
    params_type: type,
    data: NDArray[np.float64],
) -> None:
    log_likelihoods: list[float] = []
    for max_iter in range(1, MONOTONE_MAX_ITER + 1):
        fitted: Mixture = mixture_type(
            params_type(n_components=2, max_iter=max_iter, tol=0.0, rtol=0.0, seed=0)
        ).fit(data)
        assert fitted.state.convergence is not None
        log_likelihoods.append(fitted.state.convergence.log_likelihood)

    changes: NDArray[np.float64] = np.diff(log_likelihoods)
    allowance: float = MONOTONE_RTOL * abs(log_likelihoods[-1])

    assert np.all(changes >= -allowance), log_likelihoods


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mixture_type", "params_type", "sampler", "concentrations", "rtol"),
    RECOVERY_CASES,
)
def test_fit_recovers_parameters(
    mixture_type: type,
    params_type: type,
    sampler: Sampler,
    concentrations: tuple[float, float],
    rtol: float,
) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    true_means: NDArray[np.float64] = np.eye(N_FEATURES)[:2]
    data: NDArray[np.float64] = np.concatenate(
        [
            sampler(N_PER_COMPONENT, mean, concentration, rng=rng)
            for mean, concentration in zip(true_means, concentrations, strict=True)
        ]
    )

    fitted: Mixture = mixture_type(
        params_type(n_components=2, init="kmeans", max_iter=RECOVERY_MAX_ITER, seed=0)
    ).fit(data)
    assert fitted.state.components is not None

    # Match each true component to the fitted component nearest to it.
    order: NDArray[np.intp] = np.argmax(
        true_means @ fitted.state.components.means.T, axis=1
    )
    means: NDArray[np.float64] = fitted.state.components.means[order]

    assert len(set(order.tolist())) == 2
    assert np.all(np.sum(means * true_means, axis=1) > 0.99)
    assert_allclose(fitted.state.components.weights[order], 0.5, atol=0.05)
    assert_allclose(
        fitted.state.components.concentrations[order], concentrations, rtol=rtol
    )
