"""
Tests for SphericalCauchyMixture.to_dict / from_dict serialization.
"""

import json

import numpy as np
import pytest

from numpy.testing import assert_allclose
from spheremix.mixtures.spherical_cauchy import (
    SphericalCauchyMixture,
    SphericalCauchyMixtureParams,
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
def components(rng: np.random.Generator) -> SphericalCauchyMixtureParams:
    means = rng.standard_normal((N_COMPONENTS, N_FEATURES))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    return SphericalCauchyMixtureParams(
        weights=np.array([0.4, 0.6]),
        means=means,
        concentrations=np.array([0.3, 0.7]),
    )


@pytest.fixture
def fitted_model(
    components: SphericalCauchyMixtureParams,
) -> SphericalCauchyMixture:
    model = SphericalCauchyMixture(n_components=N_COMPONENTS)
    model.params_ = components
    model.log_likelihood_ = -123.45
    model.n_iter_ = 10
    return model


@pytest.fixture
def unfitted_model() -> SphericalCauchyMixture:
    return SphericalCauchyMixture(n_components=N_COMPONENTS)


# ---------------------------------------------------------------------------
# SphericalCauchyMixtureParams.to_dict
# ---------------------------------------------------------------------------


def test_components_to_dict_keys(components: SphericalCauchyMixtureParams) -> None:
    assert set(components.to_dict().keys()) == {"weights", "means", "concentrations"}


def test_components_to_dict_values_are_lists(
    components: SphericalCauchyMixtureParams,
) -> None:
    d = components.to_dict()
    assert isinstance(d["weights"], list)
    assert isinstance(d["means"], list)
    assert isinstance(d["concentrations"], list)


def test_components_to_dict_shapes(
    components: SphericalCauchyMixtureParams,
) -> None:
    d = components.to_dict()
    assert len(d["weights"]) == N_COMPONENTS
    assert len(d["means"]) == N_COMPONENTS
    assert len(d["means"][0]) == N_FEATURES
    assert len(d["concentrations"]) == N_COMPONENTS


def test_components_to_dict_values_match(
    components: SphericalCauchyMixtureParams,
) -> None:
    d = components.to_dict()
    assert_allclose(d["weights"], components.weights)
    assert_allclose(d["means"], components.means)
    assert_allclose(d["concentrations"], components.concentrations)


def test_components_to_dict_json_serializable(
    components: SphericalCauchyMixtureParams,
) -> None:
    json.dumps(components.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# SphericalCauchyMixture.to_dict
# ---------------------------------------------------------------------------


def test_to_dict_type_tag(unfitted_model: SphericalCauchyMixture) -> None:
    assert unfitted_model.to_dict()["type"] == "SphericalCauchyMixture"


def test_to_dict_version(unfitted_model: SphericalCauchyMixture) -> None:
    assert unfitted_model.to_dict()["version"] == 1


def test_to_dict_spheremix_version(unfitted_model: SphericalCauchyMixture) -> None:
    assert "spheremix_version" in unfitted_model.to_dict()


def test_to_dict_hyperparameters() -> None:
    model = SphericalCauchyMixture(n_components=3, init="kmeans")
    d = model.to_dict()
    assert d["n_components"] == 3
    assert d["init"] == "kmeans"


def test_to_dict_unfitted_params_is_none(
    unfitted_model: SphericalCauchyMixture,
) -> None:
    assert unfitted_model.to_dict()["params_"] is None


def test_to_dict_fitted_state(fitted_model: SphericalCauchyMixture) -> None:
    d = fitted_model.to_dict()
    assert d["log_likelihood_"] == pytest.approx(-123.45)
    assert d["n_iter_"] == 10
    assert d["params_"] is not None
    assert set(d["params_"].keys()) == {"weights", "means", "concentrations"}


def test_to_dict_json_serializable(fitted_model: SphericalCauchyMixture) -> None:
    json.dumps(fitted_model.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# SphericalCauchyMixture.from_dict
# ---------------------------------------------------------------------------


def test_from_dict_restores_hyperparameters() -> None:
    model = SphericalCauchyMixture(n_components=3, init="kmeans")
    restored = SphericalCauchyMixture.from_dict(model.to_dict())
    assert restored.n_components == 3
    assert restored.init == "kmeans"


def test_from_dict_restores_unfitted_state(
    unfitted_model: SphericalCauchyMixture,
) -> None:
    restored = SphericalCauchyMixture.from_dict(unfitted_model.to_dict())
    assert restored.params_ is None
    assert restored.n_iter_ == 0


def test_from_dict_restores_fitted_state(fitted_model: SphericalCauchyMixture) -> None:
    restored = SphericalCauchyMixture.from_dict(fitted_model.to_dict())
    assert restored.log_likelihood_ == pytest.approx(-123.45)
    assert restored.n_iter_ == 10
    assert restored.params_ is not None


def test_from_dict_restores_arrays_as_ndarray(
    fitted_model: SphericalCauchyMixture,
) -> None:
    restored = SphericalCauchyMixture.from_dict(fitted_model.to_dict())
    assert isinstance(restored.params_.weights, np.ndarray)
    assert isinstance(restored.params_.means, np.ndarray)
    assert isinstance(restored.params_.concentrations, np.ndarray)


def test_from_dict_restored_arrays_match(fitted_model: SphericalCauchyMixture) -> None:
    restored = SphericalCauchyMixture.from_dict(fitted_model.to_dict())
    assert_allclose(restored.params_.weights, fitted_model.params_.weights)
    assert_allclose(restored.params_.means, fitted_model.params_.means)
    assert_allclose(
        restored.params_.concentrations, fitted_model.params_.concentrations
    )


def test_from_dict_raises_on_wrong_type(fitted_model: SphericalCauchyMixture) -> None:
    d = fitted_model.to_dict()
    d["type"] = "SomeOtherModel"
    with pytest.raises(ValueError, match="type"):
        SphericalCauchyMixture.from_dict(d)


def test_from_dict_raises_on_wrong_version(
    fitted_model: SphericalCauchyMixture,
) -> None:
    d = fitted_model.to_dict()
    d["version"] = 99
    with pytest.raises(ValueError, match="version"):
        SphericalCauchyMixture.from_dict(d)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_dict_roundtrip_fitted(fitted_model: SphericalCauchyMixture) -> None:
    restored = SphericalCauchyMixture.from_dict(fitted_model.to_dict())
    assert restored.to_dict() == fitted_model.to_dict()


def test_dict_roundtrip_unfitted(unfitted_model: SphericalCauchyMixture) -> None:
    restored = SphericalCauchyMixture.from_dict(unfitted_model.to_dict())
    assert restored.to_dict() == unfitted_model.to_dict()


def test_json_roundtrip_fitted(fitted_model: SphericalCauchyMixture) -> None:
    restored = SphericalCauchyMixture.from_dict(
        json.loads(json.dumps(fitted_model.to_dict()))
    )
    assert_allclose(restored.params_.weights, fitted_model.params_.weights)
    assert_allclose(restored.params_.means, fitted_model.params_.means)
    assert_allclose(
        restored.params_.concentrations, fitted_model.params_.concentrations
    )
    assert restored.log_likelihood_ == pytest.approx(fitted_model.log_likelihood_)
    assert restored.n_iter_ == fitted_model.n_iter_
