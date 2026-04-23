"""
Tests for the spheremix Spherical Cauchy public API.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spheremix.distributions import (
    SphericalCauchyDistribution,
    sample_spherical_cauchy,
    spherical_cauchy_logkernel,
    spherical_cauchy_logpdf,
    spherical_cauchy_pdf,
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
# 1. spherical_cauchy_logkernel
# ---------------------------------------------------------------------------


class TestSphericalCauchyLogKernel:

    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_output_dtype_is_float(self) -> None:
        out: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.5)
        assert np.issubdtype(out.dtype, np.floating)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.5)
        assert np.all(np.isfinite(out))

    # --- Mathematical properties ---

    def test_rho_zero_gives_constant_kernel(self) -> None:
        """
        At rho=0 the kernel is (1-0)^(d-1) / 1^(d-1) = 1 everywhere,
        so the log-kernel must be identically 0.
        """
        out: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.0)
        assert_allclose(out, 0.0, atol=1e-12)

    def test_mode_is_global_maximum(self) -> None:
        """mu is the mode: its log-kernel must be >= every other point's."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lk_mode: np.ndarray = spherical_cauchy_logkernel(mu_row, self.mu, rho=0.7)
        lk_rest: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.7)
        assert np.all(lk_mode >= lk_rest)

    def test_antipode_is_global_minimum(self) -> None:
        """-mu is the anti-mode: its log-kernel must be <= every other point's."""
        antipode: np.ndarray = (-self.mu)[np.newaxis, :]
        lk_antipode: np.ndarray = spherical_cauchy_logkernel(
            antipode, self.mu, rho=0.7
        )
        lk_rest: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=0.7)
        assert np.all(lk_antipode <= lk_rest)

    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_higher_rho_increases_mode_kernel(self, rho: float) -> None:
        """The log-kernel at the mode must be strictly increasing with rho."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        rhos: list[float] = [0.1, rho]
        values: list[float] = [
            float(spherical_cauchy_logkernel(mu_row, self.mu, rho=r).item())
            for r in rhos
        ]
        assert values[1] > values[0]

    def test_check_input_false_normalises_mu_internally(self) -> None:
        """
        With check_input=False a scaled mu must give the same result as a
        unit mu, because the function normalises it internally.
        """
        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(5, rng)
        X: np.ndarray = random_sphere_samples(20, 5, rng)
        out_unit: np.ndarray = spherical_cauchy_logkernel(
            X, mu, rho=0.5, check_input=False
        )
        out_scaled: np.ndarray = spherical_cauchy_logkernel(
            X, 4.2 * mu, rho=0.5, check_input=False
        )
        assert_allclose(out_unit, out_scaled, rtol=1e-9)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_kernel_exponent_scales_with_dimension(self, d: int) -> None:
        """
        The spherical Cauchy kernel exponent is (d-1), not d/2 as in the PKBD.
        Verify by checking the mode-vs-antipode log-kernel gap increases with d,
        which holds when the exponent is (d-1) and the denominator at the
        antipode is (1 + rho)^2 > (1 - rho)^2 at the mode.
        """
        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(d, rng)
        mu_row: np.ndarray = mu[np.newaxis, :]
        antipode: np.ndarray = (-mu)[np.newaxis, :]
        rho: float = 0.7
        gap: float = float(
            spherical_cauchy_logkernel(mu_row, mu, rho=rho).item()
            - spherical_cauchy_logkernel(antipode, mu, rho=rho).item()
        )
        # Analytically: gap = (d-1) * log((1+rho)^2 / (1-rho)^2)
        expected_gap: float = (d - 1.0) * 2.0 * np.log((1.0 + rho) / (1.0 - rho))
        assert_allclose(gap, expected_gap, rtol=1e-9)

    # --- Input validation ---

    def test_raises_on_1d_X(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            spherical_cauchy_logkernel(self.X[0], self.mu, rho=0.5)

    def test_raises_on_2d_mu(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            spherical_cauchy_logkernel(self.X, self.mu[np.newaxis, :], rho=0.5)

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            spherical_cauchy_logkernel(self.X, self.mu[:-1], rho=0.5)

    def test_raises_when_rho_equals_one(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            spherical_cauchy_logkernel(self.X, self.mu, rho=1.0)

    def test_raises_when_rho_negative(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            spherical_cauchy_logkernel(self.X, self.mu, rho=-0.1)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            spherical_cauchy_logkernel(self.X, np.zeros(self.d), rho=0.5)

    def test_raises_on_non_unit_mu_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            spherical_cauchy_logkernel(
                self.X, 2.0 * self.mu, rho=0.5, check_input=True
            )

    def test_raises_on_non_unit_X_when_check_input_true(self) -> None:
        X_bad: np.ndarray = 2.0 * self.X
        with pytest.raises(ValueError, match="unit norm"):
            spherical_cauchy_logkernel(X_bad, self.mu, rho=0.5, check_input=True)


# ---------------------------------------------------------------------------
# 2. spherical_cauchy_logpdf
# ---------------------------------------------------------------------------


class TestSphericalCauchyLogPDF:

    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=0.5)
        assert np.all(np.isfinite(out))

    # --- Relationship to logkernel ---

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
    def test_is_constant_shift_of_logkernel(self, rho: float) -> None:
        """
        logpdf and logkernel must differ by a constant (the log-normalising
        constant), so their pairwise differences across samples must all be equal.
        """
        lp: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=rho)
        lk: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=rho)
        shift: np.ndarray = lp - lk
        assert_allclose(shift, shift[0], atol=1e-12)

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
    def test_shift_is_negative(self, rho: float) -> None:
        """
        The normalising constant is < 1 on the sphere, so log C_d < 0,
        meaning logpdf < logkernel everywhere.
        """
        lp: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=rho)
        lk: np.ndarray = spherical_cauchy_logkernel(self.X, self.mu, rho=rho)
        assert np.all(lp < lk)

    # --- Mathematical properties ---

    def test_rho_zero_is_constant(self) -> None:
        """At rho=0 the distribution is uniform, so logpdf must be constant."""
        lp: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_rho_zero_logpdf_equals_log_inverse_surface_area(self, d: int) -> None:
        """
        At rho=0 the spherical Cauchy is uniform on S^{d-1}.
        The log-pdf must equal -log(surface_area(S^{d-1}))
        = -log(2 * pi^{d/2} / Gamma(d/2)).
        """
        from scipy.special import gammaln

        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(d, rng)
        X: np.ndarray = random_sphere_samples(N_SAMPLES, d, rng)
        lp: np.ndarray = spherical_cauchy_logpdf(X, mu, rho=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)
        log_surface: float = (
            np.log(2.0) + (d / 2.0) * np.log(np.pi) - float(gammaln(d / 2.0))
        )
        assert_allclose(lp[0], -log_surface, atol=1e-9)

    def test_mode_has_maximum_logpdf(self) -> None:
        """mu is the mode, so logpdf(mu) >= logpdf(x) for all x."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lp_mode: float = float(
            spherical_cauchy_logpdf(mu_row, self.mu, rho=0.7).item()
        )
        lp_rest: np.ndarray = spherical_cauchy_logpdf(self.X, self.mu, rho=0.7)
        assert np.all(lp_mode >= lp_rest)

    # --- Input validation ---

    def test_raises_on_rho_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            spherical_cauchy_logpdf(self.X, self.mu, rho=1.5)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            spherical_cauchy_logpdf(self.X, np.zeros(self.d), rho=0.5)


# ---------------------------------------------------------------------------
# 3. spherical_cauchy_pdf
# ---------------------------------------------------------------------------


class TestSphericalCauchyPDF:

    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    def test_output_shape(self) -> None:
        out: np.ndarray = spherical_cauchy_pdf(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_equals_exp_logpdf(self) -> None:
        rho: float = 0.6
        assert_allclose(
            spherical_cauchy_pdf(self.X, self.mu, rho=rho),
            np.exp(spherical_cauchy_logpdf(self.X, self.mu, rho=rho)),
            rtol=1e-12,
        )

    def test_non_negative(self) -> None:
        out: np.ndarray = spherical_cauchy_pdf(self.X, self.mu, rho=0.5)
        assert np.all(out >= 0.0)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_monte_carlo_integral_is_one(self, d: int) -> None:
        """
        Monte Carlo estimate of the integral over S^{d-1} must be ~1.
        For uniform proposals q(x), E_q[p(x)] * surface_area ≈ 1.
        """
        from scipy.special import gammaln

        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(d, rng)
        X: np.ndarray = random_sphere_samples(N_SAMPLES, d, rng)
        pdf_vals: np.ndarray = spherical_cauchy_pdf(X, mu, rho=0.5)
        log_surface: float = (
            np.log(2.0) + (d / 2.0) * np.log(np.pi) - float(gammaln(d / 2.0))
        )
        surface_area: float = float(np.exp(log_surface))
        mc_integral: float = float(pdf_vals.mean()) * surface_area
        assert_allclose(mc_integral, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# 4. sample_spherical_cauchy
# ---------------------------------------------------------------------------


class TestSampleSphericalCauchy:

    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()

    # --- Output shape and geometry ---

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_output_shape(self, d: int) -> None:
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(N_SAMPLES, mu, rho=0.5, rng=self.rng)
        assert X.shape == (N_SAMPLES, d)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_samples_lie_on_sphere(self, d: int) -> None:
        """Every returned point must satisfy ‖x‖ = 1 to machine precision."""
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(N_SAMPLES, mu, rho=0.5, rng=self.rng)
        norms: np.ndarray = np.linalg.norm(X, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)

    # --- Edge cases ---

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_rho_zero_produces_uniform_samples(self, d: int) -> None:
        """
        At rho=0 the Möbius transform is the identity, so samples should be
        uniform on S^{d-1}. Coordinate means should be near zero.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(
            N_SAMPLES, mu, rho=0.0, rng=self.rng
        )
        se: float = 1.0 / np.sqrt(N_SAMPLES)
        assert_allclose(X.mean(axis=0), 0.0, atol=6.0 * se)

    def test_unnormalised_mu_is_accepted(self) -> None:
        """
        The sampler normalises mu internally, so un-normalised input must
        still produce valid unit-sphere samples.
        """
        mu_unnorm: np.ndarray = np.array([3.0, 0.0, 0.0])
        X: np.ndarray = sample_spherical_cauchy(10, mu_unnorm, rho=0.5, rng=self.rng)
        assert X.shape == (10, 3)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            sample_spherical_cauchy(10, np.zeros(3), rho=0.5, rng=self.rng)

    def test_raises_when_rho_equals_one(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            sample_spherical_cauchy(10, np.array([0.0, 0.0, 1.0]), rho=1.0)

    def test_raises_when_rho_negative(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            sample_spherical_cauchy(10, np.array([0.0, 0.0, 1.0]), rho=-0.1)

    def test_deterministic_with_fixed_seed(self) -> None:
        """Same seed must produce bit-identical output."""
        mu: np.ndarray = random_unit_vector(3, make_rng())
        X1: np.ndarray = sample_spherical_cauchy(
            20, mu, rho=0.5, rng=np.random.default_rng(0)
        )
        X2: np.ndarray = sample_spherical_cauchy(
            20, mu, rho=0.5, rng=np.random.default_rng(0)
        )
        assert_allclose(X1, X2)

    def test_default_rng_does_not_raise(self) -> None:
        """Passing rng=None must not raise."""
        mu: np.ndarray = np.array([0.0, 0.0, 1.0])
        sample_spherical_cauchy(5, mu, rho=0.3)

    # --- Statistical tests ---

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_samples_score_higher_log_likelihood_than_uniform(
        self, d: int, rho: float
    ) -> None:
        """
        Samples drawn from SphericalCauchy(mu, rho) should have strictly
        higher mean log-likelihood under the correct model than under rho=0.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(
            N_SAMPLES, mu, rho=rho, rng=self.rng
        )
        mean_llk: float = float(spherical_cauchy_logpdf(X, mu, rho=rho).mean())
        uniform_llk: float = float(spherical_cauchy_logpdf(X[:1], mu, rho=0.0).item())
        assert mean_llk > uniform_llk, (
            f"d={d}, rho={rho}: SpCauchy log-likelihood ({mean_llk:.4f}) "
            f"<= uniform ({uniform_llk:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7])
    def test_empirical_mean_direction_aligns_with_mu(
        self, d: int, rho: float
    ) -> None:
        """
        The mean resultant vector of the sample should point in the direction
        of mu, verified via cosine similarity.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(
            N_SAMPLES, mu, rho=rho, rng=self.rng
        )
        mean_vec: np.ndarray = X.mean(axis=0)
        cos_sim: float = float(
            (mean_vec @ mu) / (np.linalg.norm(mean_vec) * np.linalg.norm(mu))
        )
        assert cos_sim > 0.99, (
            f"d={d}, rho={rho}: cosine similarity to mu = {cos_sim:.4f}"
        )

    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_concentration_increases_with_rho(self, rho: float) -> None:
        """
        Larger rho → tighter clustering around mu → higher mean cos similarity.
        """
        d: int = 3
        mu: np.ndarray = random_unit_vector(d, self.rng)
        mean_cos_low: float = float(
            (sample_spherical_cauchy(N_SAMPLES, mu, rho=0.1, rng=make_rng()) @ mu).mean()
        )
        mean_cos_high: float = float(
            (sample_spherical_cauchy(N_SAMPLES, mu, rho=rho, rng=make_rng()) @ mu).mean()
        )
        assert mean_cos_high > mean_cos_low, (
            f"rho={rho}: mean cos at rho ({mean_cos_high:.4f}) "
            f"<= mean cos at rho=0.1 ({mean_cos_low:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_sample_log_pdfs_are_finite(self, d: int, rho: float) -> None:
        """
        logpdf evaluated at the sampled points must always be finite —
        a cross-function integration check for both sampler and density.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_spherical_cauchy(
            N_SAMPLES, mu, rho=rho, rng=self.rng
        )
        lp: np.ndarray = spherical_cauchy_logpdf(X, mu, rho=rho)
        assert np.all(np.isfinite(lp)), (
            f"d={d}, rho={rho}: {(~np.isfinite(lp)).sum()} non-finite log-pdf values"
        )


# ---------------------------------------------------------------------------
# 5. SphericalCauchyDistribution dataclass
# ---------------------------------------------------------------------------


class TestSphericalCauchyDistribution:

    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.mu: np.ndarray = random_unit_vector(3, self.rng)
        self.dist: SphericalCauchyDistribution = SphericalCauchyDistribution(
            mu=self.mu, rho=0.5
        )

    # --- Immutability ---

    def test_instance_is_frozen(self) -> None:
        with pytest.raises(Exception):
            self.dist.rho = 0.9  # type: ignore[misc]

    def test_mu_is_stored_correctly(self) -> None:
        assert_allclose(self.dist.mu, self.mu)

    def test_rho_is_stored_correctly(self) -> None:
        assert self.dist.rho == 0.5

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
            spherical_cauchy_logkernel(X, self.mu, rho=0.5),
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
            spherical_cauchy_logpdf(X, self.mu, rho=0.5),
            rtol=1e-12,
        )

    # --- Internal consistency ---

    def test_logpdf_is_constant_shift_of_logkernel(self) -> None:
        """The shift between logpdf and logkernel must be constant across X."""
        X: np.ndarray = random_sphere_samples(30, 3, self.rng)
        shift: np.ndarray = self.dist.logpdf(X) - self.dist.logkernel(X)
        assert_allclose(shift, shift[0], atol=1e-12)
