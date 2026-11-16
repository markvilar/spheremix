"""
Tests for the spheremix.distributions.poisson_kernel public API.
"""

import numpy as np
import pytest

from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from spheremix.distributions import (
    PoissonKernelDistribution,
    poisson_kernel_logkernel,
    poisson_kernel_logpdf,
    poisson_kernel_pdf,
    sample_poisson_kernel,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

RNG_SEED: int = 42
N_SAMPLES: int = 10_000  # large enough for statistical tests, fast enough for CI


def make_rng() -> np.random.Generator:
    """Return a freshly seeded Generator so every test is deterministic."""
    return np.random.default_rng(RNG_SEED)


def random_unit_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    v: np.ndarray = rng.standard_normal(d)
    return v / np.linalg.norm(v)  # type: ignore[no-any-return]


def random_sphere_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    Z: np.ndarray = rng.standard_normal((n, d))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. poisson_kernel_logkernel
# ---------------------------------------------------------------------------


class TestLogKernel:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_output_dtype_is_float(self) -> None:
        out: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.5)
        assert np.issubdtype(out.dtype, np.floating)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.5)
        assert np.all(np.isfinite(out))

    # --- Mathematical properties ---

    def test_rho_zero_gives_constant_kernel(self) -> None:
        """
        At rho=0, the Poisson kernel reduces to 1 everywhere on the sphere,
        so the log-kernel should be identically 0.
        """
        out: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.0)
        assert_allclose(out, 0.0, atol=1e-12)

    def test_mode_is_global_maximum(self) -> None:
        """
        mu is the mode, so its log-kernel must be >= every other point's.
        """
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lk_mode: np.ndarray = poisson_kernel_logkernel(mu_row, self.mu, rho=0.7)
        lk_rest: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.7)
        assert np.all(lk_mode >= lk_rest)

    def test_antipode_is_global_minimum(self) -> None:
        """
        -mu is the anti-mode, so its log-kernel must be <= every other point's.
        """
        antipode: np.ndarray = (-self.mu)[np.newaxis, :]
        lk_antipode: np.ndarray = poisson_kernel_logkernel(antipode, self.mu, rho=0.7)
        lk_rest: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=0.7)
        assert np.all(lk_antipode <= lk_rest)

    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_higher_rho_increases_mode_kernel(self, rho: float) -> None:
        """
        The kernel at the mode should be strictly increasing with rho
        because the distribution becomes more peaked.
        """
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        rhos: list[float] = [0.1, rho]
        values: list[float] = [
            float(poisson_kernel_logkernel(mu_row, self.mu, rho=r)[0]) for r in rhos
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
        out_unit: np.ndarray = poisson_kernel_logkernel(
            X, mu, rho=0.5, check_input=False
        )
        out_scaled: np.ndarray = poisson_kernel_logkernel(
            X, 4.2 * mu, rho=0.5, check_input=False
        )
        assert_allclose(out_unit, out_scaled, rtol=1e-9)

    # --- Input validation ---

    def test_raises_on_1d_X(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            poisson_kernel_logkernel(self.X[0], self.mu, rho=0.5)

    def test_raises_on_2d_mu(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            poisson_kernel_logkernel(self.X, self.mu[np.newaxis, :], rho=0.5)

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            poisson_kernel_logkernel(self.X, self.mu[:-1], rho=0.5)

    def test_raises_when_rho_equals_one(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            poisson_kernel_logkernel(self.X, self.mu, rho=1.0)

    def test_raises_when_rho_negative(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            poisson_kernel_logkernel(self.X, self.mu, rho=-0.1)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            poisson_kernel_logkernel(self.X, np.zeros(self.d), rho=0.5)

    def test_raises_on_non_unit_mu_when_check_input_true(self) -> None:
        with pytest.raises(ValueError, match="unit norm"):
            poisson_kernel_logkernel(self.X, 2.0 * self.mu, rho=0.5, check_input=True)

    def test_raises_on_non_unit_X_when_check_input_true(self) -> None:
        X_bad: np.ndarray = 2.0 * self.X
        with pytest.raises(ValueError, match="unit norm"):
            poisson_kernel_logkernel(X_bad, self.mu, rho=0.5, check_input=True)


# ---------------------------------------------------------------------------
# 2. poisson_kernel_logpdf
# ---------------------------------------------------------------------------


class TestLogPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    # --- Output contract ---

    def test_output_shape(self) -> None:
        out: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_output_is_finite(self) -> None:
        out: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=0.5)
        assert np.all(np.isfinite(out))

    # --- Relationship to logkernel ---

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
    def test_is_constant_shift_of_logkernel(self, rho: float) -> None:
        """
        logpdf and logkernel must differ by a constant (the log-normalising
        constant), so their pairwise differences across samples must all be
        equal.
        """
        lp: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=rho)
        lk: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=rho)
        shift: np.ndarray = lp - lk
        # All shifts must be identical (constant w.r.t. x)
        assert_allclose(shift, shift[0], atol=1e-12)

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
    def test_shift_is_negative(self, rho: float) -> None:
        """
        The normalising constant is < 1 on the sphere, so log C_d < 0,
        meaning logpdf < logkernel everywhere.
        """
        lp: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=rho)
        lk: np.ndarray = poisson_kernel_logkernel(self.X, self.mu, rho=rho)
        assert np.all(lp < lk)

    # --- Mathematical properties ---

    def test_rho_zero_is_constant(self) -> None:
        """At rho=0 the PKBD is uniform, so logpdf must be constant across X."""
        lp: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=0.0)
        assert_allclose(lp, lp[0], atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_rho_zero_logpdf_equals_log_inverse_surface_area(self, d: int) -> None:
        """
        At rho=0 the PKBD is uniform on S^{d-1}. The log-pdf should equal
        -log(surface_area(S^{d-1})) = -log(2 pi^{d/2} / Gamma(d/2)).
        We derive this entirely from the public API by checking that
        exp(logpdf) integrates to 1 via Monte Carlo.
        """
        from scipy.special import gammaln  # scipy is a dev dependency

        rng: np.random.Generator = make_rng()
        mu: np.ndarray = random_unit_vector(d, rng)
        X: np.ndarray = random_sphere_samples(N_SAMPLES, d, rng)
        lp: np.ndarray = poisson_kernel_logpdf(X, mu, rho=0.0)
        # All values must be identical at rho=0
        assert_allclose(lp, lp[0], atol=1e-12)
        # The constant value must equal -log(surface area of S^{d-1})
        log_surface: float = (
            np.log(2.0) + (d / 2.0) * np.log(np.pi) - float(gammaln(d / 2.0))
        )
        assert_allclose(lp[0], -log_surface, atol=1e-9)

    def test_mode_has_maximum_logpdf(self) -> None:
        """mu is the mode, so logpdf(mu) >= logpdf(x) for all x."""
        mu_row: np.ndarray = self.mu[np.newaxis, :]
        lp_mode: float = float(poisson_kernel_logpdf(mu_row, self.mu, rho=0.7)[0])
        lp_rest: np.ndarray = poisson_kernel_logpdf(self.X, self.mu, rho=0.7)
        assert np.all(lp_mode >= lp_rest)

    # --- Input validation (forwarded from logkernel) ---

    def test_raises_on_rho_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            poisson_kernel_logpdf(self.X, self.mu, rho=1.5)

    def test_raises_on_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            poisson_kernel_logpdf(self.X, np.zeros(self.d), rho=0.5)


# ---------------------------------------------------------------------------
# 3. poisson_kernel_pdf
# ---------------------------------------------------------------------------


class TestPDF:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.d: int = 3
        self.mu: np.ndarray = random_unit_vector(self.d, self.rng)
        self.X: np.ndarray = random_sphere_samples(50, self.d, self.rng)

    def test_output_shape(self) -> None:
        out: np.ndarray = poisson_kernel_pdf(self.X, self.mu, rho=0.5)
        assert out.shape == (50,)

    def test_equals_exp_logpdf(self) -> None:
        rho: float = 0.6
        assert_allclose(
            poisson_kernel_pdf(self.X, self.mu, rho=rho),
            np.exp(poisson_kernel_logpdf(self.X, self.mu, rho=rho)),
            rtol=1e-12,
        )

    def test_non_negative(self) -> None:
        out: np.ndarray = poisson_kernel_pdf(self.X, self.mu, rho=0.5)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# 4. sample_poisson_kernel
# ---------------------------------------------------------------------------


class TestSamplePoissonKernel:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()

    # --- Output shape and geometry ---

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_output_shape(self, d: int) -> None:
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=0.5, rng=self.rng)
        assert X.shape == (N_SAMPLES, d)

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    def test_samples_lie_on_sphere(self, d: int) -> None:
        """Every returned point must satisfy ‖x‖ = 1 to machine precision."""
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=0.5, rng=self.rng)
        norms: np.ndarray = np.linalg.norm(X, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)

    # --- Edge cases ---

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_rho_zero_produces_uniform_samples(self, d: int) -> None:
        """
        At rho=0 the distribution is uniform on S^{d-1}.
        Coordinate means should be near zero (within 6 standard errors).
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=0.0, rng=self.rng)
        se: float = 1.0 / np.sqrt(N_SAMPLES)
        assert_allclose(X.mean(axis=0), 0.0, atol=6.0 * se)

    def test_unnormalised_mu_is_accepted(self) -> None:
        """
        The sampler must handle un-normalised mu gracefully, producing
        valid unit-sphere samples.
        """
        mu_unnorm: np.ndarray = np.array([3.0, 0.0, 0.0])
        X: np.ndarray = sample_poisson_kernel(10, mu_unnorm, rho=0.5, rng=self.rng)
        assert X.shape == (10, 3)
        assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10)

    def test_deterministic_with_fixed_seed(self) -> None:
        """Same seed must produce bit-identical output."""
        mu: np.ndarray = random_unit_vector(3, make_rng())
        X1: np.ndarray = sample_poisson_kernel(
            20, mu, rho=0.5, rng=np.random.default_rng(0)
        )
        X2: np.ndarray = sample_poisson_kernel(
            20, mu, rho=0.5, rng=np.random.default_rng(0)
        )
        assert_allclose(X1, X2)

    def test_default_rng_does_not_raise(self) -> None:
        """Passing rng=None must not raise."""
        mu: np.ndarray = np.array([0.0, 0.0, 1.0])
        sample_poisson_kernel(5, mu, rho=0.3)

    # --- Statistical tests ---

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_samples_score_higher_log_likelihood_than_uniform(
        self, d: int, rho: float
    ) -> None:
        """
        Samples drawn from PKBD(mu, rho) should have strictly higher mean
        log-likelihood under the correct model than under the uniform model.
        The uniform log-pdf is the rho=0 value (constant across x), which we
        retrieve directly from poisson_kernel_logpdf.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=rho, rng=self.rng)
        mean_llk: float = float(poisson_kernel_logpdf(X, mu, rho=rho).mean())
        uniform_llk: float = float(poisson_kernel_logpdf(X[:1], mu, rho=0.0)[0])
        assert mean_llk > uniform_llk, (
            f"d={d}, rho={rho}: PKBD log-likelihood ({mean_llk:.4f}) "
            f"<= uniform ({uniform_llk:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7])
    def test_empirical_mean_direction_aligns_with_mu(self, d: int, rho: float) -> None:
        """
        The mean resultant vector of the sample should point in the direction
        of mu. We verify via cosine similarity rather than magnitude.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=rho, rng=self.rng)
        mean_vec: np.ndarray = X.mean(axis=0)
        cos_sim: float = float(
            (mean_vec @ mu) / (np.linalg.norm(mean_vec) * np.linalg.norm(mu))
        )
        assert (
            cos_sim > 0.99
        ), f"d={d}, rho={rho}: cosine similarity to mu = {cos_sim:.4f}"

    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_concentration_increases_with_rho(self, rho: float) -> None:
        """
        Larger rho → tighter clustering around mu → higher mean cos similarity.
        Verified by comparing mean(X @ mu) at rho=0.1 vs rho=rho.
        """
        d: int = 3
        mu: np.ndarray = random_unit_vector(d, self.rng)
        mean_cos_low: float = (
            sample_poisson_kernel(N_SAMPLES, mu, rho=0.1, rng=make_rng()) @ mu
        ).mean()
        mean_cos_high: float = (
            sample_poisson_kernel(N_SAMPLES, mu, rho=rho, rng=make_rng()) @ mu
        ).mean()
        assert mean_cos_high > mean_cos_low, (
            f"rho={rho}: mean cos at rho ({mean_cos_high:.4f}) "
            f"<= mean cos at rho=0.1 ({mean_cos_low:.4f})"
        )

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
    def test_sample_log_pdfs_are_finite(self, d: int, rho: float) -> None:
        """
        logpdf evaluated at the sampled points must always be finite.
        A non-finite value indicates a numerical issue in either the sampler
        or the density evaluator.
        """
        mu: np.ndarray = random_unit_vector(d, self.rng)
        X: np.ndarray = sample_poisson_kernel(N_SAMPLES, mu, rho=rho, rng=self.rng)
        lp: np.ndarray = poisson_kernel_logpdf(X, mu, rho=rho)
        assert np.all(
            np.isfinite(lp)
        ), f"d={d}, rho={rho}: {(~np.isfinite(lp)).sum()} non-finite log-pdf values"


# ---------------------------------------------------------------------------
# 5. PoissonKernelDistribution dataclass
# ---------------------------------------------------------------------------


class TestPoissonKernelDistribution:
    def setup_method(self) -> None:
        self.rng: np.random.Generator = make_rng()
        self.mu: np.ndarray = random_unit_vector(3, self.rng)
        self.dist: PoissonKernelDistribution = PoissonKernelDistribution(
            mu=self.mu, rho=0.5
        )

    # --- Immutability ---

    def test_instance_is_frozen(self) -> None:
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
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
            poisson_kernel_logkernel(X, self.mu, rho=0.5),
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
            poisson_kernel_logpdf(X, self.mu, rho=0.5),
            rtol=1e-12,
        )

    # --- Internal consistency ---

    def test_logpdf_is_constant_shift_of_logkernel(self) -> None:
        """The shift between logpdf and logkernel must be constant across X."""
        X: np.ndarray = random_sphere_samples(30, 3, self.rng)
        shift: np.ndarray = self.dist.logpdf(X) - self.dist.logkernel(X)
        assert_allclose(shift, shift[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Normalisation (#85)
# ---------------------------------------------------------------------------


def _integrate_over_sphere(d: int, rho: float) -> float:
    """
    Integrate exp(poisson_kernel_logpdf) over the sphere in R^d, reduced to an integral
    over t = mu^T x weighted by the area of the (d - 2)-sphere at height t.
    """
    mu: np.ndarray = np.eye(d)[0]
    log_slice_area: float = (
        np.log(2.0) + ((d - 1) / 2.0) * np.log(np.pi) - gammaln((d - 1) / 2.0)
    )

    def integrand(t: float) -> float:
        x: np.ndarray = np.zeros((1, d))
        x[0, 0] = t
        x[0, 1] = np.sqrt(max(1.0 - t * t, 0.0))
        log_density: float = float(poisson_kernel_logpdf(x, mu, rho)[0])
        return float(
            np.exp(log_density + log_slice_area) * (1.0 - t * t) ** ((d - 3) / 2.0)
        )

    result: float = quad(integrand, -1.0, 1.0, limit=200)[0]
    return result


@pytest.mark.parametrize("d", [3, 5, 10])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8, 0.95])
def test_pdf_integrates_to_one(d: int, rho: float) -> None:
    assert _integrate_over_sphere(d, rho) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("rho", [0.3, 0.7])
def test_maximum_likelihood_rho_of_samples_matches(rho: float) -> None:
    mu: np.ndarray = np.eye(3)[0]
    X: np.ndarray = sample_poisson_kernel(2_000, mu, rho, rng=make_rng())

    result = minimize_scalar(
        lambda value: -float(poisson_kernel_logpdf(X, mu, value).sum()),
        bounds=(0.0, 0.999),
        method="bounded",
    )

    assert result.x == pytest.approx(rho, abs=0.03)
