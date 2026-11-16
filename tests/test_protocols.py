"""
Conformance tests: verify concrete models satisfy their declared protocols.

isinstance() on @runtime_checkable Protocols checks method presence by name,
not signatures — these tests catch accidental method removals.
"""

import numpy as np
import pytest

from spheremix import SphericalClusterer, SphericalMixture
from spheremix.cluster import SphericalKMeans
from spheremix.mixtures import (
    PoissonKernelMixture,
    PoissonKernelMixtureParams,
    PowerSphericalMixture,
    PowerSphericalMixtureParams,
    SphericalCauchyMixture,
    SphericalCauchyMixtureParams,
    VonMisesFisherMixture,
    VonMisesFisherMixtureParams,
)


# ---------------------------------------------------------------------------
# SphericalMixture
# ---------------------------------------------------------------------------

MIXTURES: list[object] = [
    SphericalCauchyMixture(SphericalCauchyMixtureParams(n_components=2)),
    PoissonKernelMixture(PoissonKernelMixtureParams(n_components=2)),
    PowerSphericalMixture(PowerSphericalMixtureParams(n_components=2)),
    VonMisesFisherMixture(VonMisesFisherMixtureParams(n_components=2)),
]


@pytest.mark.parametrize("mixture", MIXTURES)
def test_mixture_satisfies_spherical_mixture(mixture: object) -> None:
    assert isinstance(mixture, SphericalMixture)


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
