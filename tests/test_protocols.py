"""
Conformance tests: verify concrete models satisfy their declared protocols.

isinstance() on @runtime_checkable Protocols checks method presence by name,
not signatures — these tests catch accidental method removals.
"""

import numpy as np
import pytest

from spheremix import SphericalClusterer, SphericalMixtureModel
from spheremix.cluster import SphericalKMeans
from spheremix.mixtures import SphericalCauchyMixture
from spheremix.mixtures import PowerSphericalMixture
from spheremix.mixtures import VonMisesFisherMixture


# ---------------------------------------------------------------------------
# SphericalMixtureModel
# ---------------------------------------------------------------------------


def test_spherical_cauchy_mixture_satisfies_mixture_model() -> None:
    assert isinstance(SphericalCauchyMixture(n_components=2), SphericalMixtureModel)


@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "score",
        "score_samples",
        "aic",
        "bic",
    ],
)
def test_spherical_cauchy_mixture_has_method(method: str) -> None:
    assert hasattr(SphericalCauchyMixture(n_components=2), method)

@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "score",
        "score_samples",
        "aic",
        "bic",
    ],
)
def test_power_spherical_mixture_satisfies_mixture_model() -> None:
    assert isinstance(PowerSphericalMixture(n_components=2), SphericalMixtureModel)
    
@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "score",
        "score_samples",
        "aic",
        "bic",
    ],
)
def test_von_mises_fisher_mixture_satisfies_mixture_model() -> None:
    assert isinstance(VonMisesFisherMixture(n_components=2), SphericalMixtureModel)


@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "score",
        "score_samples",
        "aic",
        "bic",
    ],
)
def test_power_spherical_mixture_has_method(method: str) -> None:
    assert hasattr(PowerSphericalMixture(n_components=2), method)
   
@pytest.mark.parametrize(
    "method",
    [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "score",
        "score_samples",
        "aic",
        "bic",
    ],
)
def test_von_mises_fisher_mixture_has_method(method: str) -> None:
    assert hasattr(VonMisesFisherMixture(n_components=2), method)


# ---------------------------------------------------------------------------
# SphericalClusterer
# ---------------------------------------------------------------------------


def test_spherical_kmeans_satisfies_clusterer() -> None:
    rng = np.random.default_rng(0)
    assert isinstance(SphericalKMeans(n_clusters=2, rng=rng), SphericalClusterer)


@pytest.mark.parametrize("method", ["fit", "predict", "fit_predict"])
def test_spherical_kmeans_has_method(method: str) -> None:
    rng = np.random.default_rng(0)
    assert hasattr(SphericalKMeans(n_clusters=2, rng=rng), method)
