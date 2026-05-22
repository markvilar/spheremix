"""
Tests for VonMisesFisherMixture.to_dict / from_dict serialization, scoring,
and information criteria.
"""

import json

import numpy as np
import pytest

from numpy.testing import assert_allclose
from spheremix.mixtures.von_mises_fisher import (
    VonMisesFisherMixture,
    VonMisesFisherMixtureParams,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_COMPONENTS: int = 2
N_FEATURES: int = 3

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def components(rng: np.random.Generator) -> VonMisesFisherMixtureParams:
    means = rng.standard_normal((N_COMPONENTS, N_FEATURES))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    return VonMisesFisherMixtureParams(
        weights=np.array([0.4, 0.6]),
        means=means,
        concentrations=np.array([1.0, 3.0]),
    )


@pytest.fixture
def fitted_model(
    components: VonMisesFisherMixtureParams,
) -> VonMisesFisherMixture:
    model = VonMisesFisherMixture(n_components=N_COMPONENTS)
    model.params_ = components
    model.log_likelihood_ = -123.45
    model.n_iter_ = 10
    return model


@pytest.fixture
def unfitted_model() -> VonMisesFisherMixture:
    return VonMisesFisherMixture(n_components=N_COMPONENTS)


# ---------------------------------------------------------------------------
# VonMisesFisherMixtureParams.to_dict
# ---------------------------------------------------------------------------


def test_components_to_dict_keys(components: VonMisesFisherMixtureParams) -> None:
    assert set(components.to_dict().keys()) == {"weights", "means", "concentrations"}


def test_components_to_dict_values_are_lists(
    components: VonMisesFisherMixtureParams,
) -> None:
    d = components.to_dict()
    assert isinstance(d["weights"], list)
    assert isinstance(d["means"], list)
    assert isinstance(d["concentrations"], list)


def test_components_to_dict_shapes(
    components: VonMisesFisherMixtureParams,
) -> None:
    d = components.to_dict()
    assert len(d["weights"]) == N_COMPONENTS
    assert len(d["means"]) == N_COMPONENTS
    assert len(d["means"][0]) == N_FEATURES
    assert len(d["concentrations"]) == N_COMPONENTS


def test_components_to_dict_values_match(
    components: VonMisesFisherMixtureParams,
) -> None:
    d = components.to_dict()
    assert_allclose(d["weights"], components.weights)
    assert_allclose(d["means"], components.means)
    assert_allclose(d["concentrations"], components.concentrations)


def test_components_to_dict_json_serializable(
    components: VonMisesFisherMixtureParams,
) -> None:
    json.dumps(components.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# VonMisesFisherMixture.to_dict
# ---------------------------------------------------------------------------


def test_to_dict_type_tag(unfitted_model: VonMisesFisherMixture) -> None:
    assert unfitted_model.to_dict()["type"] == "VonMisesFisherMixture"


def test_to_dict_version(unfitted_model: VonMisesFisherMixture) -> None:
    assert unfitted_model.to_dict()["version"] == 1


def test_to_dict_spheremix_version(unfitted_model: VonMisesFisherMixture) -> None:
    assert "spheremix_version" in unfitted_model.to_dict()


def test_to_dict_hyperparameters() -> None:
    model = VonMisesFisherMixture(
        n_components=3, init="kmeans", max_iter=50, tol=1e-3, rtol=1e-5, verbose=True
    )
    d = model.to_dict()
    assert d["n_components"] == 3
    assert d["init"] == "kmeans"
    assert d["max_iter"] == 50
    assert d["tol"] == pytest.approx(1e-3)
    assert d["rtol"] == pytest.approx(1e-5)
    assert d["verbose"] is True


def test_to_dict_unfitted_params_is_none(
    unfitted_model: VonMisesFisherMixture,
) -> None:
    assert unfitted_model.to_dict()["params_"] is None


def test_to_dict_fitted_state(fitted_model: VonMisesFisherMixture) -> None:
    d = fitted_model.to_dict()
    assert d["log_likelihood_"] == pytest.approx(-123.45)
    assert d["n_iter_"] == 10
    assert d["params_"] is not None
    assert set(d["params_"].keys()) == {"weights", "means", "concentrations"}


def test_to_dict_json_serializable(fitted_model: VonMisesFisherMixture) -> None:
    json.dumps(fitted_model.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# VonMisesFisherMixture.from_dict
# ---------------------------------------------------------------------------


def test_from_dict_restores_hyperparameters() -> None:
    model = VonMisesFisherMixture(
        n_components=3, init="kmeans", max_iter=50, tol=1e-3, rtol=1e-5, verbose=True
    )
    restored = VonMisesFisherMixture.from_dict(model.to_dict())
    assert restored.n_components == 3
    assert restored.init == "kmeans"
    assert restored.max_iter == 50
    assert restored.tol == pytest.approx(1e-3)
    assert restored.rtol == pytest.approx(1e-5)
    assert restored.verbose is True


def test_from_dict_restores_unfitted_state(
    unfitted_model: VonMisesFisherMixture,
) -> None:
    restored = VonMisesFisherMixture.from_dict(unfitted_model.to_dict())
    assert restored.params_ is None
    assert restored.n_iter_ == 0


def test_from_dict_restores_fitted_state(fitted_model: VonMisesFisherMixture) -> None:
    restored = VonMisesFisherMixture.from_dict(fitted_model.to_dict())
    assert restored.log_likelihood_ == pytest.approx(-123.45)
    assert restored.n_iter_ == 10
    assert restored.params_ is not None


def test_from_dict_restores_arrays_as_ndarray(
    fitted_model: VonMisesFisherMixture,
) -> None:
    restored = VonMisesFisherMixture.from_dict(fitted_model.to_dict())
    assert restored.params_ is not None
    assert isinstance(restored.params_.weights, np.ndarray)
    assert isinstance(restored.params_.means, np.ndarray)
    assert isinstance(restored.params_.concentrations, np.ndarray)


def test_from_dict_restored_arrays_match(fitted_model: VonMisesFisherMixture) -> None:
    restored = VonMisesFisherMixture.from_dict(fitted_model.to_dict())
    assert restored.params_ is not None
    assert fitted_model.params_ is not None
    assert_allclose(restored.params_.weights, fitted_model.params_.weights)
    assert_allclose(restored.params_.means, fitted_model.params_.means)
    assert_allclose(
        restored.params_.concentrations, fitted_model.params_.concentrations
    )


def test_from_dict_raises_on_wrong_type(fitted_model: VonMisesFisherMixture) -> None:
    d = fitted_model.to_dict()
    d["type"] = "SomeOtherModel"
    with pytest.raises(ValueError, match="type"):
        VonMisesFisherMixture.from_dict(d)


def test_from_dict_raises_on_wrong_version(
    fitted_model: VonMisesFisherMixture,
) -> None:
    d = fitted_model.to_dict()
    d["version"] = 99
    with pytest.raises(ValueError, match="version"):
        VonMisesFisherMixture.from_dict(d)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_dict_roundtrip_fitted(fitted_model: VonMisesFisherMixture) -> None:
    restored = VonMisesFisherMixture.from_dict(fitted_model.to_dict())
    assert restored.to_dict() == fitted_model.to_dict()


def test_dict_roundtrip_unfitted(unfitted_model: VonMisesFisherMixture) -> None:
    restored = VonMisesFisherMixture.from_dict(unfitted_model.to_dict())
    assert restored.to_dict() == unfitted_model.to_dict()


def test_json_roundtrip_fitted(fitted_model: VonMisesFisherMixture) -> None:
    restored = VonMisesFisherMixture.from_dict(
        json.loads(json.dumps(fitted_model.to_dict()))
    )
    assert restored.params_ is not None
    assert fitted_model.params_ is not None
    assert_allclose(restored.params_.weights, fitted_model.params_.weights)
    assert_allclose(restored.params_.means, fitted_model.params_.means)
    assert_allclose(
        restored.params_.concentrations, fitted_model.params_.concentrations
    )
    assert restored.log_likelihood_ == pytest.approx(fitted_model.log_likelihood_)
    assert restored.n_iter_ == fitted_model.n_iter_


# ---------------------------------------------------------------------------
# Shared fixture for scoring / information-criterion tests
# ---------------------------------------------------------------------------

N_SAMPLES: int = 20


@pytest.fixture
def X(rng: np.random.Generator) -> np.ndarray:
    """Unit vectors on S^2, shape (N_SAMPLES, N_FEATURES)."""
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    return X / np.linalg.norm(X, axis=1, keepdims=True)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# score_samples
# ---------------------------------------------------------------------------


def test_score_samples_returns_ndarray(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert isinstance(fitted_model.score_samples(X), np.ndarray)


def test_score_samples_shape(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert fitted_model.score_samples(X).shape == (N_SAMPLES,)


def test_score_samples_finite(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert np.all(np.isfinite(fitted_model.score_samples(X)))


def test_score_samples_raises_if_unfitted(
    unfitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        unfitted_model.score_samples(X)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_returns_float(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert isinstance(fitted_model.score(X), float)


def test_score_equals_mean_of_score_samples(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert fitted_model.score(X) == pytest.approx(fitted_model.score_samples(X).mean())


def test_score_accepts_y_none(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    fitted_model.score(X, y=None)  # must not raise


def test_score_raises_if_unfitted(
    unfitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        unfitted_model.score(X)


# ---------------------------------------------------------------------------
# aic
# ---------------------------------------------------------------------------


def test_aic_returns_float(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    assert isinstance(fitted_model.aic(X), float)


def test_aic_formula(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    n_params = N_COMPONENTS * (N_FEATURES + 1) - 1
    expected = -2.0 * fitted_model.score_samples(X).sum() + 2.0 * n_params
    assert fitted_model.aic(X) == pytest.approx(expected)


def test_aic_finite(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    assert np.isfinite(fitted_model.aic(X))


def test_aic_raises_if_unfitted(
    unfitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        unfitted_model.aic(X)


# ---------------------------------------------------------------------------
# bic
# ---------------------------------------------------------------------------


def test_bic_returns_float(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    assert isinstance(fitted_model.bic(X), float)


def test_bic_formula(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    n_params = N_COMPONENTS * (N_FEATURES + 1) - 1
    expected = -2.0 * fitted_model.score_samples(X).sum() + n_params * np.log(N_SAMPLES)
    assert fitted_model.bic(X) == pytest.approx(expected)


def test_bic_finite(fitted_model: VonMisesFisherMixture, X: np.ndarray) -> None:
    assert np.isfinite(fitted_model.bic(X))


def test_bic_raises_if_unfitted(
    unfitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        unfitted_model.bic(X)


def test_bic_penalises_more_than_aic(
    fitted_model: VonMisesFisherMixture, X: np.ndarray
) -> None:
    assert fitted_model.bic(X) > fitted_model.aic(X)
