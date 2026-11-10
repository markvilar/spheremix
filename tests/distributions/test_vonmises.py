"""
Tests for the spheremix.distributions.vonmises public API.

Reference comparisons
---------------------
d=2 : scipy.stats.vonmises
    The circular von Mises distribution is the vMF on S^1. scipy's logpdf at
    angular deviation δ = arccos(mu^T x) with concentration kappa must agree
    with our von_mises_fisher_logpdf.

d=3 analytical :
    The normalising constant has the closed form
        C_3(κ) = κ / (4π · sinh(κ))
    so log C_3(κ) = log(κ) − log(4π) − log(sinh(κ)).

Any d — Monte Carlo normalisation :
    For uniform x on S^{d-1},  E[pdf(x)] = 1 / |S^{d-1}|,
    so  E[pdf(x)] · |S^{d-1}| ≈ 1.  Surface area |S^{d-1}| = 2π^{d/2}/Γ(d/2).
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose
from scipy.special import gammaln
from scipy.stats import vonmises as scipy_vonmises

from spheremix.distributions import (
    VmfLogNormalizerMethod,
    VonMisesFisherDistribution,
    make_log_normalizer,
    sample_von_mises_fisher,
    von_mises_fisher_log_normalizing_constant,
    von_mises_fisher_logkernel,
    von_mises_fisher_logpdf,
    von_mises_fisher_pdf,
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
    return v / np.linalg.norm(v)  # type: ignore[no-any-return]


def random_sphere_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    Z: np.ndarray = rng.standard_normal((n, d))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)  # type: ignore[no-any-return]


def sphere_surface_area(d: int) -> float:
    """Surface area of S^{d-1} = 2 π^{d/2} / Γ(d/2)."""
    return float(2.0 * np.pi ** (d / 2.0) / np.exp(gammaln(d / 2.0)))


def vmf_logpdf_d3_analytical(X: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    """Ground-truth logpdf for d=3: log C_3(κ) + κ μᵀx."""
    log_c3 = np.log(kappa) - np.log(4.0 * np.pi) - np.log(np.sinh(kappa))
    return log_c3 + kappa * (X @ mu)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. von_mises_fisher_logkernel
# ---------------------------------------------------------------------------


class TestLogKernel:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out = von_mises_fisher_logkernel(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_output_dtype_is_float(self) -> None:
        out = von_mises_fisher_logkernel(self.X, self.mu, kappa=2.0)
        assert np.issubdtype(out.dtype, np.floating)

    def test_output_is_finite(self) -> None:
        out = von_mises_fisher_logkernel(self.X, self.mu, kappa=2.0)
        assert np.all(np.isfinite(out))

    # --- Mathematical properties ---

    def test_kappa_zero_gives_zero_logkernel(self) -> None:
        """At kappa=0 the kernel is exp(0 * mu^T x) = 1, so log-kernel = 0."""
        out = von_mises_fisher_logkernel(self.X, self.mu, kappa=0.0)
        assert_allclose(out, 0.0, atol=1e-12)

    def test_equals_kappa_times_dot_product(self) -> None:
        """log-kernel = kappa * mu^T x by definition."""
        kappa = 4.0
        expected = kappa * (self.X @ self.mu)
        out = von_mises_fisher_logkernel(self.X, self.mu, kappa=kappa)
        assert_allclose(out, expected, rtol=1e-12)

    def test_mode_has_maximum_logkernel(self) -> None:
        mu_row = self.mu[np.newaxis, :]
        lk_mode = von_mises_fisher_logkernel(mu_row, self.mu, kappa=3.0)
        lk_rest = von_mises_fisher_logkernel(self.X, self.mu, kappa=3.0)
        assert np.all(lk_mode >= lk_rest)

    def test_antipode_has_minimum_logkernel(self) -> None:
        antipode = (-self.mu)[np.newaxis, :]
        lk_antipode = von_mises_fisher_logkernel(antipode, self.mu, kappa=3.0)
        lk_rest = von_mises_fisher_logkernel(self.X, self.mu, kappa=3.0)
        assert np.all(lk_antipode <= lk_rest)

    def test_check_input_false_normalises_mu_internally(self) -> None:
        rng = make_rng()
        mu = random_unit_vector(5, rng)
        X = random_sphere_samples(20, 5, rng)
        out_unit = von_mises_fisher_logkernel(X, mu, kappa=2.0, check_input=False)
        out_scaled = von_mises_fisher_logkernel(
            X, 4.2 * mu, kappa=2.0, check_input=False
        )
        assert_allclose(out_unit, out_scaled, rtol=1e-9)

    # --- Input validation ---

    def test_raises_on_1d_X(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            von_mises_fisher_logkernel(self.X[0], self.mu, kappa=1.0)

    def test_raises_on_2d_mu(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            von_mises_fisher_logkernel(self.X, self.mu[np.newaxis, :], kappa=1.0)

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            von_mises_fisher_logkernel(self.X, self.mu[:-1], kappa=1.0)

    def test_raises_when_kappa_negative(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            von_mises_fisher_logkernel(self.X, self.mu, kappa=-0.1)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            von_mises_fisher_logkernel(self.X, np.zeros(self.d), kappa=1.0)

    def test_raises_on_non_unit_mu_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            von_mises_fisher_logkernel(
                self.X, 2.0 * self.mu, kappa=1.0, check_input=True
            )

    def test_raises_on_non_unit_X_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            von_mises_fisher_logkernel(
                2.0 * self.X, self.mu, kappa=1.0, check_input=True
            )


# ---------------------------------------------------------------------------
# 2. von_mises_fisher_logpdf
# ---------------------------------------------------------------------------


class TestLogPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out = von_mises_fisher_logpdf(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_output_is_finite(self) -> None:
        out = von_mises_fisher_logpdf(self.X, self.mu, kappa=2.0)
        assert np.all(np.isfinite(out))

    # --- Relationship to logkernel ---

    @pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0, 10.0])
    def test_is_constant_shift_of_logkernel(self, kappa: float) -> None:
        """logpdf and logkernel must differ by a constant (the log-normalising constant)."""
        lp = von_mises_fisher_logpdf(self.X, self.mu, kappa=kappa)
        lk = von_mises_fisher_logkernel(self.X, self.mu, kappa=kappa)
        shift = lp - lk
        assert_allclose(shift, shift[0], atol=1e-12)

    @pytest.mark.parametrize("kappa", [1.0, 3.0, 10.0])
    def test_shift_is_negative(self, kappa: float) -> None:
        """The normalising constant C_d(kappa) > 1 for kappa > 0, so logpdf < logkernel."""
        lp = von_mises_fisher_logpdf(self.X, self.mu, kappa=kappa)
        lk = von_mises_fisher_logkernel(self.X, self.mu, kappa=kappa)
        assert np.all(lp <= lk)

    # --- Mathematical properties ---

    def test_kappa_zero_is_constant(self) -> None:
        """At kappa=0 the distribution is uniform on S^{d-1}, so logpdf is constant."""
        lp = von_mises_fisher_logpdf(self.X, self.mu, kappa=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_kappa_zero_logpdf_equals_log_inverse_surface_area(self, d: int) -> None:
        rng = make_rng()
        mu = random_unit_vector(d, rng)
        X = random_sphere_samples(100, d, rng)
        lp = von_mises_fisher_logpdf(X, mu, kappa=0.0)
        expected = -np.log(sphere_surface_area(d))
        assert_allclose(lp[0], expected, atol=1e-9)

    def test_mode_has_maximum_logpdf(self) -> None:
        mu_row = self.mu[np.newaxis, :]
        lp_mode = float(von_mises_fisher_logpdf(mu_row, self.mu, kappa=3.0)[0])
        lp_rest = von_mises_fisher_logpdf(self.X, self.mu, kappa=3.0)
        assert np.all(lp_mode >= lp_rest)

    # --- Reference: d=2 vs scipy.stats.vonmises ---

    @pytest.mark.parametrize("kappa", [0.5, 1.0, 3.0, 10.0])
    def test_d2_agrees_with_scipy(self, kappa: float) -> None:
        """
        For d=2, the vMF reduces to the circular von Mises distribution.
        scipy.stats.vonmises.logpdf(delta, kappa) where delta = signed angle
        between x and mu must match our logpdf.
        """
        rng = make_rng()
        theta_mu = rng.uniform(0, 2 * np.pi)
        mu = np.array([np.cos(theta_mu), np.sin(theta_mu)])

        angles = rng.uniform(0, 2 * np.pi, size=50)
        X = np.column_stack([np.cos(angles), np.sin(angles)])

        our_logpdf = von_mises_fisher_logpdf(
            X, mu, kappa=kappa, method=VmfLogNormalizerMethod.EXACT
        )

        # scipy uses the signed angular deviation from the mean
        delta = angles - theta_mu
        scipy_logpdf = scipy_vonmises.logpdf(delta, kappa)

        assert_allclose(our_logpdf, scipy_logpdf, atol=1e-9)

    # --- Reference: d=3 analytical ---

    @pytest.mark.parametrize("kappa", [0.5, 1.0, 3.0, 10.0, 50.0])
    def test_d3_agrees_with_analytical_formula(self, kappa: float) -> None:
        """
        For d=3 the normalising constant is C_3(κ) = κ/(4π sinh(κ)), giving
        logpdf = log(κ) − log(4π) − log(sinh(κ)) + κ μᵀx.
        """
        rng = make_rng()
        mu = random_unit_vector(3, rng)
        X = random_sphere_samples(100, 3, rng)
        our_logpdf = von_mises_fisher_logpdf(
            X, mu, kappa=kappa, method=VmfLogNormalizerMethod.EXACT
        )
        ref_logpdf = vmf_logpdf_d3_analytical(X, mu, kappa)
        assert_allclose(our_logpdf, ref_logpdf, rtol=1e-7)

    # --- Monte Carlo normalisation ---

    @pytest.mark.parametrize("d", [2, 3, 5])
    @pytest.mark.parametrize("kappa", [0.0, 1.0, 5.0])
    def test_pdf_integrates_to_one(self, d: int, kappa: float) -> None:
        """
        E_{x~uniform}[pdf(x)] · |S^{d-1}| ≈ 1.
        Tolerance is 3 Monte Carlo standard errors.
        """
        rng = make_rng()
        mu = random_unit_vector(d, rng)
        X_unif = random_sphere_samples(N_SAMPLES, d, rng)
        pdf_vals = von_mises_fisher_pdf(
            X_unif, mu, kappa=kappa, method=VmfLogNormalizerMethod.EXACT
        )
        mc_integral = pdf_vals.mean() * sphere_surface_area(d)
        # std of the estimator ≈ std(pdf) / sqrt(N) * surface_area
        se = pdf_vals.std() / np.sqrt(N_SAMPLES) * sphere_surface_area(d)
        assert_allclose(mc_integral, 1.0, atol=4.0 * se)

    # --- Input validation ---

    def test_raises_on_kappa_negative(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            von_mises_fisher_logpdf(self.X, self.mu, kappa=-1.0)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            von_mises_fisher_logpdf(self.X, np.zeros(self.d), kappa=1.0)


# ---------------------------------------------------------------------------
# 3. von_mises_fisher_pdf
# ---------------------------------------------------------------------------


class TestPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    def test_output_shape(self) -> None:
        out = von_mises_fisher_pdf(self.X, self.mu, kappa=2.0)
        assert out.shape == (50,)

    def test_equals_exp_logpdf(self) -> None:
        assert_allclose(
            von_mises_fisher_pdf(self.X, self.mu, kappa=2.0),
            np.exp(von_mises_fisher_logpdf(self.X, self.mu, kappa=2.0)),
            rtol=1e-12,
        )

    def test_non_negative(self) -> None:
        out = von_mises_fisher_pdf(self.X, self.mu, kappa=2.0)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# 4. sample_von_mises_fisher
# ---------------------------------------------------------------------------


class TestSampleVonMisesFisher:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_output_shape(self, d: int) -> None:
        mu = random_unit_vector(d, self.rng)
        X = sample_von_mises_fisher(N_SAMPLES, mu, kappa=2.0, rng=self.rng)
        assert X.shape == (N_SAMPLES, d)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_samples_lie_on_sphere(self, d: int) -> None:
        mu = random_unit_vector(d, self.rng)
        X = sample_von_mises_fisher(N_SAMPLES, mu, kappa=2.0, rng=self.rng)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_kappa_zero_produces_uniform_samples(self, d: int) -> None:
        """At kappa=0 coordinate means should be near zero (within 6 standard errors)."""
        mu = random_unit_vector(d, self.rng)
        X = sample_von_mises_fisher(N_SAMPLES, mu, kappa=0.0, rng=self.rng)
        se = 1.0 / np.sqrt(N_SAMPLES)
        assert_allclose(X.mean(axis=0), 0.0, atol=6.0 * se)

    def test_unnormalised_mu_is_accepted(self) -> None:
        mu_unnorm = np.array([3.0, 0.0, 0.0])
        X = sample_von_mises_fisher(10, mu_unnorm, kappa=2.0, rng=self.rng)
        assert X.shape == (10, 3)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    def test_deterministic_with_fixed_seed(self) -> None:
        mu = random_unit_vector(3, make_rng())
        X1 = sample_von_mises_fisher(20, mu, kappa=2.0, rng=np.random.default_rng(0))
        X2 = sample_von_mises_fisher(20, mu, kappa=2.0, rng=np.random.default_rng(0))
        assert_allclose(X1, X2)

    def test_default_rng_does_not_raise(self) -> None:
        mu = np.array([0.0, 0.0, 1.0])
        sample_von_mises_fisher(5, mu, kappa=1.0)

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("kappa", [1.0, 5.0])
    def test_empirical_mean_direction_aligns_with_mu(
        self, d: int, kappa: float
    ) -> None:
        """The mean resultant vector of the sample should point toward mu."""
        mu = random_unit_vector(d, self.rng)
        X = sample_von_mises_fisher(N_SAMPLES, mu, kappa=kappa, rng=self.rng)
        mean_vec = X.mean(axis=0)
        cos_sim = float(
            (mean_vec @ mu) / (np.linalg.norm(mean_vec) * np.linalg.norm(mu))
        )
        assert (
            cos_sim > 0.99
        ), f"d={d}, kappa={kappa}: cosine similarity = {cos_sim:.4f}"

    @pytest.mark.parametrize("kappa", [1.0, 5.0, 20.0])
    def test_concentration_increases_with_kappa(self, kappa: float) -> None:
        """Larger kappa → tighter clustering around mu → higher mean projection."""
        d = 3
        mu = random_unit_vector(d, self.rng)
        mean_cos_low = (
            sample_von_mises_fisher(N_SAMPLES, mu, kappa=0.5, rng=make_rng()) @ mu
        ).mean()
        mean_cos_high = (
            sample_von_mises_fisher(N_SAMPLES, mu, kappa=kappa, rng=make_rng()) @ mu
        ).mean()
        assert mean_cos_high > mean_cos_low

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("kappa", [1.0, 5.0, 20.0])
    def test_samples_score_higher_log_likelihood_than_uniform(
        self, d: int, kappa: float
    ) -> None:
        """Samples from vMF(mu, kappa) must have higher mean logpdf than uniform."""
        mu = random_unit_vector(d, self.rng)
        X = sample_von_mises_fisher(N_SAMPLES, mu, kappa=kappa, rng=self.rng)
        mean_llk = float(von_mises_fisher_logpdf(X, mu, kappa=kappa).mean())
        uniform_llk = float(von_mises_fisher_logpdf(X[:1], mu, kappa=0.0)[0])
        assert mean_llk > uniform_llk

    # --- Input validation ---

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            sample_von_mises_fisher(10, np.zeros(3), kappa=1.0)

    def test_raises_when_kappa_negative(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            sample_von_mises_fisher(10, np.array([0.0, 0.0, 1.0]), kappa=-1.0)


# ---------------------------------------------------------------------------
# 5. VonMisesFisherDistribution dataclass
# ---------------------------------------------------------------------------


class TestVonMisesFisherDistribution:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.mu: np.ndarray = random_unit_vector(3, self.rng)
        self.dist: VonMisesFisherDistribution = VonMisesFisherDistribution(
            mu=self.mu, kappa=2.0
        )

    def test_instance_is_frozen(self) -> None:
        with pytest.raises(Exception):
            self.dist.kappa = 5.0  # type: ignore[misc]

    def test_mu_is_stored_correctly(self) -> None:
        assert_allclose(self.dist.mu, self.mu)

    def test_kappa_is_stored_correctly(self) -> None:
        assert self.dist.kappa == 2.0

    def test_sample_output_shape(self) -> None:
        X = self.dist.sample(50)
        assert X.shape == (50, 3)

    def test_sample_points_lie_on_sphere(self) -> None:
        X = self.dist.sample(50)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    def test_logkernel_output_shape(self) -> None:
        X = random_sphere_samples(20, 3, self.rng)
        assert self.dist.logkernel(X).shape == (20,)

    def test_logkernel_agrees_with_standalone(self) -> None:
        X = random_sphere_samples(20, 3, self.rng)
        assert_allclose(
            self.dist.logkernel(X),
            von_mises_fisher_logkernel(X, self.mu, kappa=2.0),
            rtol=1e-12,
        )

    def test_logpdf_output_shape(self) -> None:
        X = random_sphere_samples(20, 3, self.rng)
        assert self.dist.logpdf(X).shape == (20,)

    def test_logpdf_agrees_with_standalone(self) -> None:
        X = random_sphere_samples(20, 3, self.rng)
        assert_allclose(
            self.dist.logpdf(X),
            von_mises_fisher_logpdf(X, self.mu, kappa=2.0),
            rtol=1e-12,
        )

    def test_logpdf_is_constant_shift_of_logkernel(self) -> None:
        X = random_sphere_samples(30, 3, self.rng)
        shift = self.dist.logpdf(X) - self.dist.logkernel(X)
        assert_allclose(shift, shift[0], atol=1e-12)


# ---------------------------------------------------------------------------
# 6. VmfLogNormalizerMethod — enum, registry, and Amos bounds
# ---------------------------------------------------------------------------


class TestVmfLogNormalizerMethod:
    @pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
    @pytest.mark.parametrize("dim", [2, 3, 5, 10, 50])
    @pytest.mark.parametrize("kappa", [0.0, 0.1, 1.0, 10.0, 100.0])
    def test_all_methods_return_finite(
        self, method: VmfLogNormalizerMethod, dim: int, kappa: float
    ) -> None:
        result = von_mises_fisher_log_normalizing_constant(dim, kappa, method=method)
        assert np.isfinite(result), f"method={method}, dim={dim}, kappa={kappa}"

    @pytest.mark.parametrize("dim", [2, 3, 5, 10, 50])
    @pytest.mark.parametrize("kappa", [0.0, 0.1, 1.0, 10.0, 100.0])
    def test_amos_bounds_bracket_exact(self, dim: int, kappa: float) -> None:
        exact = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.EXACT
        )
        upper = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.AMOS_UPPER
        )
        lower = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.AMOS_LOWER
        )
        assert (
            lower <= exact + 1e-9
        ), f"dim={dim}, kappa={kappa}: lower={lower} > exact={exact}"
        assert (
            upper >= exact - 1e-9
        ), f"dim={dim}, kappa={kappa}: upper={upper} < exact={exact}"

    @pytest.mark.parametrize("dim", [2, 3, 5])
    @pytest.mark.parametrize("kappa", [0.0, 1.0, 10.0])
    def test_all_methods_agree_at_kappa_zero(self, dim: int, kappa: float) -> None:
        if kappa != 0.0:
            return
        exact = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.EXACT
        )
        upper = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.AMOS_UPPER
        )
        lower = von_mises_fisher_log_normalizing_constant(
            dim, kappa, method=VmfLogNormalizerMethod.AMOS_LOWER
        )
        assert_allclose(upper, exact, atol=1e-10)
        assert_allclose(lower, exact, atol=1e-10)

    def test_default_method_is_amos_upper(self) -> None:
        result_default = von_mises_fisher_log_normalizing_constant(3, 2.0)
        result_upper = von_mises_fisher_log_normalizing_constant(
            3, 2.0, method=VmfLogNormalizerMethod.AMOS_UPPER
        )
        assert result_default == result_upper

    def test_default_logpdf_is_finite_where_exact_underflows(self) -> None:
        """At dim=1024 and kappa=10, the exact Bessel evaluation underflows."""
        dim: int = 1024
        kappa: float = 10.0
        rng = make_rng()
        mu = random_unit_vector(dim, rng)
        X = random_sphere_samples(5, dim, rng)

        with pytest.warns(RuntimeWarning, match="divide by zero"):
            exact = von_mises_fisher_log_normalizing_constant(
                dim, kappa, method=VmfLogNormalizerMethod.EXACT
            )
        logpdf = von_mises_fisher_logpdf(X, mu, kappa=kappa)

        assert not np.isfinite(exact)
        assert np.all(np.isfinite(logpdf))

    def test_make_log_normalizer_returns_correct_function(self) -> None:
        for method in VmfLogNormalizerMethod:
            fn = make_log_normalizer(method)
            direct = von_mises_fisher_log_normalizing_constant(3, 2.0, method=method)
            assert fn(3, 2.0) == direct

    @pytest.mark.parametrize("method", list(VmfLogNormalizerMethod))
    def test_logpdf_method_parameter_propagates(
        self, method: VmfLogNormalizerMethod
    ) -> None:
        rng = make_rng()
        mu = random_unit_vector(3, rng)
        X = random_sphere_samples(10, 3, rng)
        lp = von_mises_fisher_logpdf(X, mu, kappa=2.0, method=method)
        expected_shift = von_mises_fisher_log_normalizing_constant(
            3, 2.0, method=method
        )
        lk = von_mises_fisher_logkernel(X, mu, kappa=2.0)
        assert_allclose(lp, expected_shift + lk, rtol=1e-12)

    def test_str_enum_values(self) -> None:
        assert VmfLogNormalizerMethod.EXACT == "exact"
        assert VmfLogNormalizerMethod.AMOS_UPPER == "amos_upper"
        assert VmfLogNormalizerMethod.AMOS_LOWER == "amos_lower"
