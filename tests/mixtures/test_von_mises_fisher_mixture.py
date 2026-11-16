"""
Tests for VonMisesFisherMixture and its params, components and state.
"""

import dataclasses

import numpy as np
import pytest

from collections.abc import Callable
from typing import Any

from numpy.testing import assert_allclose
from numpy.typing import NDArray

from spheremix.distributions import (
    VmfLogNormalizerMethod,
    von_mises_fisher_logpdf,
)
from spheremix.mixtures import (
    VonMisesFisherMixture,
    VonMisesFisherMixtureComponents,
    VonMisesFisherMixtureParams,
    VonMisesFisherMixtureState,
    SphericalMixture,
    SphericalMixtureComponents,
    SphericalMixtureConvergence,
    SphericalMixtureParams,
    SphericalMixtureState,
    compute_aic,
    compute_bic,
    compute_bici,
    compute_dic,
    compute_icl,
    compute_log_likelihoods,
    compute_responsibilities,
)

N_COMPONENTS: int = 2
N_FEATURES: int = 3
N_SAMPLES: int = 20
CONCENTRATIONS: NDArray[np.float64] = np.array([2.0, 8.0])

SCORING_METHODS: list[str] = [
    "predict_proba",
    "predict",
    "score_samples",
    "score",
    "aic",
    "bic",
    "icl",
    "bici",
    "dic",
]
SHARED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "predict_proba": compute_responsibilities,
    "score_samples": compute_log_likelihoods,
    "aic": compute_aic,
    "bic": compute_bic,
    "icl": compute_icl,
    "bici": compute_bici,
    "dic": compute_dic,
}


@pytest.fixture
def X() -> NDArray[np.float64]:
    points: NDArray[np.float64] = np.random.default_rng(0).standard_normal(
        (N_SAMPLES, N_FEATURES)
    )
    return points / np.linalg.norm(points, axis=1, keepdims=True)  # type: ignore[no-any-return]


@pytest.fixture
def params() -> VonMisesFisherMixtureParams:
    return VonMisesFisherMixtureParams(n_components=N_COMPONENTS, max_iter=5, seed=0)


@pytest.fixture
def components() -> VonMisesFisherMixtureComponents:
    return VonMisesFisherMixtureComponents(
        weights=np.array([0.4, 0.6]),
        means=np.eye(N_FEATURES)[:N_COMPONENTS],
        concentrations=CONCENTRATIONS,
        log_normalizer_method=VmfLogNormalizerMethod.AMOS_UPPER,
    )


@pytest.fixture
def convergence() -> SphericalMixtureConvergence:
    return SphericalMixtureConvergence(log_likelihood=-12.5, n_iter=3, converged=True)


@pytest.fixture
def fitted_mixture(
    params: VonMisesFisherMixtureParams,
    components: VonMisesFisherMixtureComponents,
    convergence: SphericalMixtureConvergence,
) -> VonMisesFisherMixture:
    return VonMisesFisherMixture.from_state(
        VonMisesFisherMixtureState(
            params=params, components=components, convergence=convergence
        )
    )


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def test_params_satisfy_protocol(params: VonMisesFisherMixtureParams) -> None:
    assert isinstance(params, SphericalMixtureParams)


@pytest.mark.parametrize("n_components", [0, -1])
def test_params_reject_too_few_components(n_components: int) -> None:
    with pytest.raises(ValueError, match="n_components must be at least 1"):
        VonMisesFisherMixtureParams(n_components=n_components)


@pytest.mark.parametrize("max_iter", [0, -1])
def test_params_reject_too_few_iterations(max_iter: int) -> None:
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        VonMisesFisherMixtureParams(n_components=N_COMPONENTS, max_iter=max_iter)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_components_satisfy_protocol(
    components: VonMisesFisherMixtureComponents,
) -> None:
    assert isinstance(components, SphericalMixtureComponents)


def test_components_count_and_dimension(
    components: VonMisesFisherMixtureComponents,
) -> None:
    assert components.component_count == N_COMPONENTS
    assert components.dimension == N_FEATURES
    assert components.free_parameter_count == N_COMPONENTS * (N_FEATURES + 1) - 1


def test_log_joint_densities_use_logpdf(
    X: NDArray[np.float64],
    components: VonMisesFisherMixtureComponents,
) -> None:
    expected: NDArray[np.float64] = np.stack(
        [
            np.log(components.weights[k])
            + von_mises_fisher_logpdf(
                X,
                components.means[k],
                CONCENTRATIONS[k],
                method=VmfLogNormalizerMethod.AMOS_UPPER,
            )
            for k in range(N_COMPONENTS)
        ],
        axis=1,
    )

    assert_allclose(components.log_joint_densities(X), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_state_satisfies_protocol(params: VonMisesFisherMixtureParams) -> None:
    assert isinstance(VonMisesFisherMixtureState(params=params), SphericalMixtureState)


def test_state_is_not_fitted_without_components(
    params: VonMisesFisherMixtureParams,
) -> None:
    assert VonMisesFisherMixtureState(params=params).is_fitted is False


def test_state_rejects_components_without_convergence(
    params: VonMisesFisherMixtureParams,
    components: VonMisesFisherMixtureComponents,
) -> None:
    with pytest.raises(ValueError, match="both be set or both be None"):
        VonMisesFisherMixtureState(params=params, components=components)


def test_state_rejects_convergence_without_components(
    params: VonMisesFisherMixtureParams,
    convergence: SphericalMixtureConvergence,
) -> None:
    with pytest.raises(ValueError, match="both be set or both be None"):
        VonMisesFisherMixtureState(params=params, convergence=convergence)


def test_state_rejects_wrong_component_count(
    components: VonMisesFisherMixtureComponents,
    convergence: SphericalMixtureConvergence,
) -> None:
    with pytest.raises(ValueError, match="params.n_components is 3"):
        VonMisesFisherMixtureState(
            params=VonMisesFisherMixtureParams(n_components=3),
            components=components,
            convergence=convergence,
        )


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------


def test_mixture_satisfies_protocol(params: VonMisesFisherMixtureParams) -> None:
    assert isinstance(VonMisesFisherMixture(params), SphericalMixture)


def test_new_mixture_holds_only_params(params: VonMisesFisherMixtureParams) -> None:
    mixture: VonMisesFisherMixture = VonMisesFisherMixture(params)

    assert mixture.params is params
    assert mixture.is_fitted is False
    assert mixture.state.components is None
    assert mixture.state.convergence is None


def test_fit_returns_new_fitted_mixture(
    X: NDArray[np.float64],
    params: VonMisesFisherMixtureParams,
) -> None:
    mixture: VonMisesFisherMixture = VonMisesFisherMixture(params)

    fitted: VonMisesFisherMixture = mixture.fit(X)

    assert fitted is not mixture
    assert fitted.is_fitted is True
    assert fitted.params is params
    assert mixture.is_fitted is False


def test_fit_records_convergence(
    X: NDArray[np.float64],
    params: VonMisesFisherMixtureParams,
) -> None:
    fitted: VonMisesFisherMixture = VonMisesFisherMixture(params).fit(X)
    convergence: SphericalMixtureConvergence | None = fitted.state.convergence

    assert convergence is not None
    assert 1 <= convergence.n_iter <= params.max_iter
    assert convergence.log_likelihood == pytest.approx(
        fitted.score_samples(X).sum(), rel=1e-10
    )


def test_fit_stopped_at_max_iter_is_not_converged(X: NDArray[np.float64]) -> None:
    params: VonMisesFisherMixtureParams = VonMisesFisherMixtureParams(
        n_components=N_COMPONENTS, max_iter=1, tol=0.0, rtol=0.0, seed=0
    )

    convergence: SphericalMixtureConvergence | None = (
        VonMisesFisherMixture(params).fit(X).state.convergence
    )

    assert convergence is not None
    assert convergence.n_iter == 1
    assert convergence.converged is False


def test_fit_with_loose_tolerance_is_converged(X: NDArray[np.float64]) -> None:
    params: VonMisesFisherMixtureParams = VonMisesFisherMixtureParams(
        n_components=N_COMPONENTS, max_iter=5, tol=np.inf, seed=0
    )

    convergence: SphericalMixtureConvergence | None = (
        VonMisesFisherMixture(params).fit(X).state.convergence
    )

    # The first iteration compares with a log-likelihood of -inf, so the
    # earliest a fit can converge is the second iteration.
    assert convergence is not None
    assert convergence.n_iter == 2
    assert convergence.converged is True


def test_from_state_keeps_state(fitted_mixture: VonMisesFisherMixture) -> None:
    restored: VonMisesFisherMixture = VonMisesFisherMixture.from_state(
        fitted_mixture.state
    )

    assert restored.state is fitted_mixture.state
    assert restored.is_fitted is True


@pytest.mark.parametrize("method", SCORING_METHODS)
def test_unfitted_mixture_raises(
    X: NDArray[np.float64],
    params: VonMisesFisherMixtureParams,
    method: str,
) -> None:
    with pytest.raises(RuntimeError, match="not fitted; call fit first"):
        getattr(VonMisesFisherMixture(params), method)(X)


@pytest.mark.parametrize("method", list(SHARED_FUNCTIONS))
def test_methods_call_shared_functions(
    X: NDArray[np.float64],
    fitted_mixture: VonMisesFisherMixture,
    components: VonMisesFisherMixtureComponents,
    method: str,
) -> None:
    expected: Any = SHARED_FUNCTIONS[method](X, components)

    assert_allclose(getattr(fitted_mixture, method)(X), expected)


def test_predict_and_score(
    X: NDArray[np.float64],
    fitted_mixture: VonMisesFisherMixture,
    components: VonMisesFisherMixtureComponents,
) -> None:
    assert_allclose(
        fitted_mixture.predict(X),
        compute_responsibilities(X, components).argmax(axis=1),
    )
    assert fitted_mixture.score(X) == pytest.approx(
        compute_log_likelihoods(X, components).mean()
    )


# ---------------------------------------------------------------------------
# Log normaliser method
# ---------------------------------------------------------------------------


def test_default_log_normalizer_method_is_amos_upper() -> None:
    params: VonMisesFisherMixtureParams = VonMisesFisherMixtureParams(n_components=2)

    assert params.log_normalizer_method == VmfLogNormalizerMethod.AMOS_UPPER


@pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
def test_fit_copies_log_normalizer_method_to_components(
    X: NDArray[np.float64],
    method: VmfLogNormalizerMethod,
) -> None:
    params: VonMisesFisherMixtureParams = VonMisesFisherMixtureParams(
        n_components=N_COMPONENTS, max_iter=3, log_normalizer_method=method, seed=0
    )

    components: VonMisesFisherMixtureComponents | None = (
        VonMisesFisherMixture(params).fit(X).state.components
    )

    assert components is not None
    assert components.log_normalizer_method == method


@pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
def test_log_joint_densities_use_log_normalizer_method(
    X: NDArray[np.float64],
    components: VonMisesFisherMixtureComponents,
    method: VmfLogNormalizerMethod,
) -> None:
    with_method: VonMisesFisherMixtureComponents = dataclasses.replace(
        components, log_normalizer_method=method
    )
    expected: NDArray[np.float64] = np.log(components.weights[0]) + (
        von_mises_fisher_logpdf(
            X, components.means[0], CONCENTRATIONS[0], method=method
        )
    )

    log_joint: NDArray[np.float64] = with_method.log_joint_densities(X)

    assert_allclose(log_joint[:, 0], expected, rtol=1e-12)
    assert np.all(np.isfinite(log_joint))
