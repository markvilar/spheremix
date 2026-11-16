"""
Tests that the mixture E-step and log-likelihood use the normalised log-pdf.
"""

from collections.abc import Callable

import numpy as np
import pytest

from numpy.testing import assert_allclose
from numpy.typing import NDArray

from spheremix.distributions import (
    VmfLogNormalizerMethod,
    poisson_kernel_logpdf,
    power_spherical_logpdf,
    sample_uniform_sphere,
    spherical_cauchy_logpdf,
    von_mises_fisher_logpdf,
)
from spheremix.mixtures import (
    PoissonKernelMixture,
    PoissonKernelMixtureComponents,
    PoissonKernelMixtureParams,
    PowerSphericalMixture,
    PowerSphericalMixtureComponents,
    PowerSphericalMixtureParams,
    SphericalCauchyMixture,
    SphericalCauchyMixtureComponents,
    SphericalCauchyMixtureParams,
    SphericalMixtureComponents,
    SphericalMixtureConvergence,
    VonMisesFisherMixture,
    VonMisesFisherMixtureComponents,
    VonMisesFisherMixtureParams,
    VonMisesFisherMixtureState,
    compute_responsibilities,
)

type Mixture = (
    SphericalCauchyMixture
    | PoissonKernelMixture
    | PowerSphericalMixture
    | VonMisesFisherMixture
)
type LogPdf = Callable[
    [NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]
]

N_SAMPLES: int = 200
N_FEATURES: int = 3

# (components type, log-pdf, two concentrations that differ strongly)
CASES: list[
    tuple[Callable[..., SphericalMixtureComponents], LogPdf, tuple[float, float]]
] = [
    (SphericalCauchyMixtureComponents, spherical_cauchy_logpdf, (0.2, 0.9)),
    (PoissonKernelMixtureComponents, poisson_kernel_logpdf, (0.2, 0.9)),
    (PowerSphericalMixtureComponents, power_spherical_logpdf, (5.0, 50.0)),
    (
        lambda **arrays: VonMisesFisherMixtureComponents(
            **arrays, log_normalizer_method=VmfLogNormalizerMethod.AMOS_UPPER
        ),
        von_mises_fisher_logpdf,
        (5.0, 50.0),
    ),
]
# (mixture type, params type)
MIXTURE_TYPES: list[tuple[type, type]] = [
    (SphericalCauchyMixture, SphericalCauchyMixtureParams),
    (PoissonKernelMixture, PoissonKernelMixtureParams),
    (PowerSphericalMixture, PowerSphericalMixtureParams),
    (VonMisesFisherMixture, VonMisesFisherMixtureParams),
]


@pytest.fixture
def data() -> NDArray[np.float64]:
    rng: np.random.Generator = np.random.default_rng(0)
    centres: NDArray[np.float64] = np.eye(N_FEATURES)[:2]
    points: NDArray[np.float64] = centres[rng.integers(0, 2, N_SAMPLES)]
    points = points + 0.3 * rng.standard_normal((N_SAMPLES, N_FEATURES))
    return points / np.linalg.norm(points, axis=1, keepdims=True)  # type: ignore[no-any-return]


@pytest.mark.parametrize(("mixture_type", "params_type"), MIXTURE_TYPES)
def test_log_likelihood_matches_score(
    mixture_type: type,
    params_type: type,
    data: NDArray[np.float64],
) -> None:
    model: Mixture = mixture_type(params_type(n_components=2, max_iter=10)).fit(data)
    convergence: SphericalMixtureConvergence | None = model.state.convergence

    assert convergence is not None
    assert convergence.log_likelihood == pytest.approx(
        model.score(data) * data.shape[0], rel=1e-10
    )


@pytest.mark.parametrize(("components_type", "logpdf", "concentrations"), CASES)
def test_responsibilities_use_logpdf(
    components_type: Callable[..., SphericalMixtureComponents],
    logpdf: LogPdf,
    concentrations: tuple[float, float],
) -> None:
    data: NDArray[np.float64] = sample_uniform_sphere(
        N_SAMPLES, N_FEATURES, rng=np.random.default_rng(1)
    )
    means: NDArray[np.float64] = np.eye(N_FEATURES)[:2]
    weights: NDArray[np.float64] = np.array([0.5, 0.5])
    components: SphericalMixtureComponents = components_type(
        weights=weights, means=means, concentrations=np.array(concentrations)
    )

    log_joint: NDArray[np.float64] = np.stack(
        [
            np.log(weights[k]) + logpdf(data, means[k], concentrations[k])
            for k in range(2)
        ],
        axis=1,
    )
    expected: NDArray[np.float64] = np.exp(
        log_joint - np.logaddexp.reduce(log_joint, axis=1, keepdims=True)
    )

    assert_allclose(
        compute_responsibilities(data, components),
        expected,
        rtol=1e-10,
    )


def test_vmf_estep_is_finite_at_high_dimension() -> None:
    dim: int = 1024
    data: NDArray[np.float64] = sample_uniform_sphere(
        50, dim, rng=np.random.default_rng(2)
    )
    params: VonMisesFisherMixtureParams = VonMisesFisherMixtureParams(n_components=2)
    model: VonMisesFisherMixture = VonMisesFisherMixture.from_state(
        VonMisesFisherMixtureState(
            params=params,
            components=VonMisesFisherMixtureComponents(
                weights=np.array([0.5, 0.5]),
                means=np.eye(dim)[:2],
                concentrations=np.array([10.0, 10.0]),
                log_normalizer_method=VmfLogNormalizerMethod.AMOS_UPPER,
            ),
            convergence=SphericalMixtureConvergence(
                log_likelihood=0.0,
                n_iter=1,
                converged=True,
                log_likelihood_history=(0.0,),
            ),
        )
    )

    assert np.all(np.isfinite(model.predict_proba(data)))
    assert np.all(np.isfinite(model.score_samples(data)))

    fitted: VonMisesFisherMixture = VonMisesFisherMixture(
        VonMisesFisherMixtureParams(n_components=2, max_iter=5)
    ).fit(data)
    assert fitted.state.convergence is not None
    assert np.isfinite(fitted.state.convergence.log_likelihood)
