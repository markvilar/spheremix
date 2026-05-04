"""
Tests for the spheremix.distributions.power_spherical public API.
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose
from spheremix.distributions import (
    PowerSphericalDistribution,
    power_spherical_logkernel,
    power_spherical_logpdf,
    power_spherical_pdf,
    sample_power_spherical,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

RNG_SEED: int = 42
N_SAMPLES: int = 10_000


def make_rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def random_unit_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    v: np.ndarray = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def random_sphere_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    Z: np.ndarray = rng.standard_normal((n, d))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. power_spherical_logkernel
# ---------------------------------------------------------------------------


class TestLogKernel:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_output_dtype_is_float(self) -> None:
        out: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=2.0)
        assert np.issubdtype(out.dtype, np.floating)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=2.0)
        assert np.all(np.isfinite(out))

    # --- Mathematical properties ---

    def test_kappa_zero_gives_constant_kernel(self) -> None:
        """At kappa=0 the kernel (1 + mu^T x)^0 = 1, so log-kernel must be 0."""
        out: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=0.0)
        assert_allclose(out, 0.0, atol=1e-12)

    def test_mode_is_global_maximum(self) -> None:
        """mu is the mode, so its log-kernel must be >= every other point's."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lk_mode: np.ndarray = power_spherical_logkernel(mu_row, self.mu, kappa=3.0)
        lk_rest: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=3.0)
        assert np.all(lk_mode >= lk_rest)

    def test_antipode_is_global_minimum(self) -> None:
        """
        -mu maximises (1 + mu^T x)^kappa at its minimum value of 0,
        so its log-kernel must be <= every other point's (clamped by eps).
        """
        antipode: np.ndarray = (-self.mu)[np.newaxis, :]
        lk_antipode: np.ndarray = power_spherical_logkernel(
            antipode, self.mu, kappa=3.0
        )
        lk_rest: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=3.0)
        assert np.all(lk_antipode <= lk_rest)

    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_higher_kappa_increases_mode_kernel(self, kappa: float) -> None:
        """The kernel value at the mode is log(2)^kappa, which grows with kappa."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        values: list[float] = [
            float(power_spherical_logkernel(mu_row, self.mu, kappa=k)[0])
            for k in [0.5, kappa]
        ]
        assert values[1] > values[0]

    def test_check_input_false_normalises_mu_internally(self) -> None:
        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(5, rng)
        X: np.ndarray = random_sphere_samples(20, 5, rng)
        out_unit: np.ndarray = power_spherical_logkernel(
            X, mu, kappa=2.0, check_input=False
        )
        out_scaled: np.ndarray = power_spherical_logkernel(
            X, 4.2 * mu, kappa=2.0, check_input=False
        )
        assert_allclose(out_unit, out_scaled, rtol=1e-9)

    # --- Input validation ---

    def test_raises_on_1d_X(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            power_spherical_logkernel(self.X[0], self.mu, kappa=1.0)

    def test_raises_on_2d_mu(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            power_spherical_logkernel(self.X, self.mu[np.newaxis, :], kappa=1.0)

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            power_spherical_logkernel(self.X, self.mu[:-1], kappa=1.0)

    def test_raises_when_kappa_negative(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            power_spherical_logkernel(self.X, self.mu, kappa=-0.1)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            power_spherical_logkernel(self.X, np.zeros(self.d), kappa=1.0)

    def test_raises_on_non_unit_mu_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            power_spherical_logkernel(
                self.X, 2.0 * self.mu, kappa=1.0, check_input=True
            )

    def test_raises_on_non_unit_X_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            power_spherical_logkernel(
                2.0 * self.X, self.mu, kappa=1.0, check_input=True
            )


# ---------------------------------------------------------------------------
# 2. power_spherical_logpdf
# ---------------------------------------------------------------------------


class TestLogPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=2.0)
        assert np.all(np.isfinite(out))

    # --- Relationship to logkernel ---

    @pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0, 10.0])
    def test_is_constant_shift_of_logkernel(self, kappa: float) -> None:
        """logpdf and logkernel must differ by a constant (the log-normalising constant)."""
        lp: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=kappa)
        lk: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=kappa)
        shift: np.ndarray = lp - lk
        assert_allclose(shift, shift[0], atol=1e-12)

    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_shift_is_negative(self, kappa: float) -> None:
        """The normalising constant C(kappa, d) > 1, so logpdf < logkernel everywhere."""
        lp: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=kappa)
        lk: np.ndarray = power_spherical_logkernel(self.X, self.mu, kappa=kappa)
        assert np.all(lp <= lk)

    # --- Mathematical properties ---

    def test_kappa_zero_is_constant(self) -> None:
        """At kappa=0 the distribution is uniform, so logpdf must be constant."""
        lp: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_kappa_zero_logpdf_equals_log_inverse_surface_area(self, d: int) -> None:
        """
        At kappa=0 the distribution is uniform on S^{d-1}. The constant log-pdf
        must equal -log(surface_area(S^{d-1})) = -log(2 pi^{d/2} / Gamma(d/2)).
        Verified by Monte Carlo: mean(exp(logpdf)) ≈ 1 (all values are equal
        to 1/|Omega|, so their mean equals the inverse surface area — we instead
        verify the exact analytical value matches the known formula).
        """
        from scipy.special import gammaln

        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(d, rng)
        X: np.ndarray = random_sphere_samples(N_SAMPLES, d, rng)
        lp: np.ndarray = power_spherical_logpdf(X, mu, kappa=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)
        log_surface: float = (
            np.log(2.0) + (d / 2.0) * np.log(np.pi) - float(gammaln(d / 2.0))
        )
        assert_allclose(lp[0], -log_surface, atol=1e-9)

    def test_mode_has_maximum_logpdf(self) -> None:
        """mu is the mode, so logpdf(mu) >= logpdf(x) for all x."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lp_mode: float = float(power_spherical_logpdf(mu_row, self.mu, kappa=3.0)[0])
        lp_rest: np.ndarray = power_spherical_logpdf(self.X, self.mu, kappa=3.0)
        assert np.all(lp_mode >= lp_rest)

    # --- Input validation ---

    def test_raises_on_kappa_negative(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            power_spherical_logpdf(self.X, self.mu, kappa=-1.0)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            power_spherical_logpdf(self.X, np.zeros(self.d), kappa=1.0)


# ---------------------------------------------------------------------------
# 3. power_spherical_pdf
# ---------------------------------------------------------------------------


class TestPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    def test_output_shape(self) -> None:
        out: np.ndarray = power_spherical_pdf(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_equals_exp_logpdf(self) -> None:
        assert_allclose(
            power_spherical_pdf(self.X, self.mu, kappa=2.0),
            np.exp(power_spherical_logpdf(self.X, self.mu, kappa=2.0)),
            rtol=1e-12,
        )

    def test_non_negative(self) -> None:
        out: np.ndarray = power_spherical_pdf(self.X, self.mu, kappa=2.0)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# 4. sample_power_spherical
# ---------------------------------------------------------------------------


class TestSamplePowerSpherical:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()

    # --- Output shape and geometry ---

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_output_shape(self, d: int) -> None:
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=2.0, rng=self.rng)
        assert X.shape == (N_SAMPLES, d)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_samples_lie_on_sphere(self, d: int) -> None:
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=2.0, rng=self.rng)
        norms: np.ndarray = np.linalg.norm(X, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)

    # --- Edge cases ---

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_kappa_zero_produces_uniform_samples(self, d: int) -> None:
        """
        At kappa=0 the distribution is uniform on S^{d-1}.
        Coordinate means should be near zero (within 6 standard errors).
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=0.0, rng=self.rng)
        se: float = 1.0 / np.sqrt(N_SAMPLES)
        assert_allclose(X.mean(axis=0), 0.0, atol=6.0 * se)

    def test_unnormalised_mu_is_accepted(self) -> None:
        mu_unnorm: np.ndarray = np.array([3.0, 0.0, 0.0])
        X: np.ndarray = sample_power_spherical(10, mu_unnorm, kappa=2.0, rng=self.rng)
        assert X.shape == (10, 3)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    def test_deterministic_with_fixed_seed(self) -> None:
        mu: np.ndarray = random_unit_vector(3, make_rng())
        X1: np.ndarray = sample_power_spherical(
            20, mu, kappa=2.0, rng=np.random.default_rng(0)
        )
        X2: np.ndarray = sample_power_spherical(
            20, mu, kappa=2.0, rng=np.random.default_rng(0)
        )
        assert_allclose(X1, X2)

    def test_default_rng_does_not_raise(self) -> None:
        mu: np.ndarray = np.array([0.0, 0.0, 1.0])
        sample_power_spherical(5, mu, kappa=1.0)

    # --- Statistical tests ---

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_samples_score_higher_log_likelihood_than_uniform(
        self, d: int, kappa: float
    ) -> None:
        """
        Samples from PowerSpherical(mu, kappa) must have strictly higher mean
        log-likelihood under the correct model than under the uniform (kappa=0).
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=kappa, rng=self.rng)
        mean_llk: float = float(power_spherical_logpdf(X, mu, kappa=kappa).mean())
        uniform_llk: float = float(power_spherical_logpdf(X[:1], mu, kappa=0.0)[0])
        assert mean_llk > uniform_llk, (
            f"d={d}, kappa={kappa}: PS log-likelihood ({mean_llk:.4f}) "
            f"<= uniform ({uniform_llk:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("kappa", [1.0, 3.0])
    def test_empirical_mean_direction_aligns_with_mu(
        self, d: int, kappa: float
    ) -> None:
        """The mean resultant vector of the sample should point toward mu."""
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=kappa, rng=self.rng)
        mean_vec: np.ndarray = X.mean(axis=0)
        cos_sim: float = float(
            (mean_vec @ mu) / (np.linalg.norm(mean_vec) * np.linalg.norm(mu))
        )
        assert (
            cos_sim > 0.99
        ), f"d={d}, kappa={kappa}: cosine similarity = {cos_sim:.4f}"

    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_concentration_increases_with_kappa(self, kappa: float) -> None:
        """Larger kappa → tighter clustering around mu → higher mean cosine similarity."""
        d: int = 3
        mu: np.ndarray = random_unit_vector(d, self.rng)
        mean_cos_low: float = (
            sample_power_spherical(N_SAMPLES, mu, kappa=0.5, rng=make_rng()) @ mu
        ).mean()
        mean_cos_high: float = (
            sample_power_spherical(N_SAMPLES, mu, kappa=kappa, rng=make_rng()) @ mu
        ).mean()
        assert mean_cos_high > mean_cos_low, (
            f"kappa={kappa}: mean cos at kappa ({mean_cos_high:.4f}) "
            f"<= mean cos at kappa=0.5 ({mean_cos_low:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_sample_log_pdfs_are_finite(self, d: int, kappa: float) -> None:
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_power_spherical(N_SAMPLES, mu, kappa=kappa, rng=self.rng)
        lp: np.ndarray = power_spherical_logpdf(X, mu, kappa=kappa)
        assert np.all(
            np.isfinite(lp)
        ), f"d={d}, kappa={kappa}: {(~np.isfinite(lp)).sum()} non-finite log-pdf values"


# ---------------------------------------------------------------------------
# 5. PowerSphericalDistribution dataclass
# ---------------------------------------------------------------------------


class TestPowerSphericalDistribution:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.mu: np.ndarray = random_unit_vector(3, self.rng)
        self.dist: PowerSphericalDistribution = PowerSphericalDistribution(
            mu=self.mu, kappa=2.0
        )

    # --- Immutability ---

    def test_instance_is_frozen(self) -> None:
        with pytest.raises(Exception):
            self.dist.kappa = 5.0  # type: ignore[misc]

    def test_mu_is_stored_correctly(self) -> None:
        assert_allclose(self.dist.mu, self.mu)

    def test_kappa_is_stored_correctly(self) -> None:
        assert self.dist.kappa == 2.0

    # --- sample ---

    def test_sample_output_shape(self) -> None:
        X: np.ndarray = self.dist.sample(50)
        assert X.shape == (50, 3)

    def test_sample_points_lie_on_sphere(self) -> None:
        X: np.ndarray = self.dist.sample(50)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    # --- logkernel ---

    def test_logkernel_output_shape(self) -> None:
        X: np.ndarray = random_sphere_samples(20, 3, self.rng)
        assert self.dist.logkernel(X).shape == (20,)

    def test_logkernel_agrees_with_standalone(self) -> None:
        X: np.ndarray = random_sphere_samples(20, 3, self.rng)
        assert_allclose(
            self.dist.logkernel(X),
            power_spherical_logkernel(X, self.mu, kappa=2.0),
            rtol=1e-12,
        )

    # --- logpdf ---

    def test_logpdf_output_shape(self) -> None:
        X: np.ndarray = random_sphere_samples(20, 3, self.rng)
        assert self.dist.logpdf(X).shape == (20,)

    def test_logpdf_agrees_with_standalone(self) -> None:
        X: np.ndarray = random_sphere_samples(20, 3, self.rng)
        assert_allclose(
            self.dist.logpdf(X),
            power_spherical_logpdf(X, self.mu, kappa=2.0),
            rtol=1e-12,
        )

    # --- Internal consistency ---

    def test_logpdf_is_constant_shift_of_logkernel(self) -> None:
        X: np.ndarray = random_sphere_samples(30, 3, self.rng)
        shift: np.ndarray = self.dist.logpdf(X) - self.dist.logkernel(X)
        assert_allclose(shift, shift[0], atol=1e-12)
