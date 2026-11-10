"""
Tests for fitting VonMisesFisherMixture and its M-step concentration update.
"""

import warnings

import numpy as np
import pytest

from numpy.typing import NDArray

from spheremix.distributions import (
    VmfLogNormalizerMethod,
    sample_von_mises_fisher,
    von_mises_fisher_log_normalizing_constant,
)
from spheremix.mixtures import VonMisesFisherMixture
from spheremix.mixtures.initializer import _kmeans_mean_initializer
from spheremix.mixtures.von_mises_fisher import (
    _bessel_ratio,
    _solve_concentration,
)

N_PER_COMPONENT: int = 250
TRUE_WEIGHTS: NDArray[np.float64] = np.array([0.5, 0.5])

# Exact maximum-likelihood kappa for (dim, r_bar), from mpmath at 40 digits.
EXACT_REFERENCE: list[tuple[int, float, float]] = [
    (3, 1e-4, 0.00030000000180000002),
    (3, 0.3, 0.95314947285740595),
    (3, 0.95, 19.999999999999997),
    (3, 0.999, 1000.0),
    (256, 1e-4, 0.025600000254015506),
    (256, 0.3, 84.341435129172792),
    (256, 0.95, 2485.1148277165142),
    (256, 0.999, 127436.71811076605),
    (4096, 1e-4, 0.40960000409400102),
    (4096, 0.3, 1350.2752463548552),
    (4096, 0.95, 39900.499349202727),
    (4096, 0.999, 2046476.2378688064),
]


def _sample_two_components(
    dim: int,
    kappa: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    means: NDArray[np.float64] = np.eye(dim)[:2]
    data: NDArray[np.float64] = np.concatenate(
        [
            sample_von_mises_fisher(N_PER_COMPONENT, mean, kappa, rng=rng)
            for mean in means
        ]
    )
    return data, means


def _fit(
    data: NDArray[np.float64],
    method: VmfLogNormalizerMethod,
) -> VonMisesFisherMixture:
    model: VonMisesFisherMixture = VonMisesFisherMixture(
        n_components=2, max_iter=30, log_normalizer_method=method
    )
    # A seeded K-means start; the seed setting arrives with #67.
    model.init = _kmeans_mean_initializer(rng=np.random.default_rng(0))  # type: ignore[assignment]
    return model.fit(data)


@pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
@pytest.mark.parametrize("dim", [3, 16, 256])
def test_fit_recovers_parameters(method: VmfLogNormalizerMethod, dim: int) -> None:
    kappa: float = 10.0 * dim
    data, true_means = _sample_two_components(dim, kappa, np.random.default_rng(dim))

    model: VonMisesFisherMixture = _fit(data, method)

    params = model.get_params()
    order: NDArray[np.intp] = np.argmax(params.means @ true_means.T, axis=0)
    assert np.all(np.sum(params.means[order] * true_means, axis=1) > 0.99)
    np.testing.assert_allclose(params.weights[order], TRUE_WEIGHTS, atol=0.05)
    assert np.all(np.isfinite(params.concentrations))
    assert np.all(params.concentrations > 0.0)
    np.testing.assert_allclose(params.concentrations, kappa, rtol=0.3)


def test_fit_high_dimension_default() -> None:
    dim: int = 1024
    data, _ = _sample_two_components(dim, 10.0 * dim, np.random.default_rng(1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model: VonMisesFisherMixture = _fit(data, VmfLogNormalizerMethod.AMOS_UPPER)

    params = model.get_params()
    assert np.all(np.isfinite(params.means))
    assert np.all(np.isfinite(params.concentrations))
    assert np.isfinite(model.log_likelihood_)


@pytest.mark.parametrize(("dim", "r_bar", "expected"), EXACT_REFERENCE)
def test_exact_solver_matches_reference(
    dim: int, r_bar: float, expected: float
) -> None:
    kappa: float = _solve_concentration(r_bar, dim, VmfLogNormalizerMethod.EXACT)

    assert kappa == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "method",
    [VmfLogNormalizerMethod.AMOS_UPPER, VmfLogNormalizerMethod.AMOS_LOWER],
)
@pytest.mark.parametrize("dim", [3, 16, 256, 4096])
@pytest.mark.parametrize("r_bar", [0.1, 0.5, 0.95, 0.999])
def test_closed_form_maximises_bound_objective(
    method: VmfLogNormalizerMethod,
    dim: int,
    r_bar: float,
) -> None:
    def objective(kappa: float) -> float:
        return r_bar * kappa + von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=method
        )

    kappa: float = _solve_concentration(r_bar, dim, method)

    assert objective(kappa) > objective(kappa * (1.0 - 1e-3))
    assert objective(kappa) > objective(kappa * (1.0 + 1e-3))


@pytest.mark.parametrize("dim", [2, 3, 16, 256, 1024, 4096])
@pytest.mark.parametrize("r_bar", [1e-4, 0.3, 0.95, 0.999])
def test_exact_solver_lies_in_amos_bracket(dim: int, r_bar: float) -> None:
    lower: float = _solve_concentration(r_bar, dim, VmfLogNormalizerMethod.AMOS_LOWER)
    exact: float = _solve_concentration(r_bar, dim, VmfLogNormalizerMethod.EXACT)
    upper: float = _solve_concentration(r_bar, dim, VmfLogNormalizerMethod.AMOS_UPPER)

    assert lower <= exact <= upper


@pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
def test_zero_resultant_gives_zero_concentration(
    method: VmfLogNormalizerMethod,
) -> None:
    assert _solve_concentration(0.0, 16, method) == 0.0


def test_bessel_ratio_raises_when_not_converged() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        _bessel_ratio(1e5, 3, max_terms=10)
