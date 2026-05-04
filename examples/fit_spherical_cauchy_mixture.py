"""
fit_spherical_cauchy_mixture.py

Generate synthetic samples from orthogonal spherical Cauchy modes,
fit a Spherical Cauchy mixture model with EM, and log per-cluster statistics.

Usage
-----
    python examples/fit_spherical_cauchy_mixture.py
    python examples/fit_spherical_cauchy_mixture.py --n-components 5 --n-features 10 --n-samples 300
"""

import sys
import time

import click
import numpy as np
from loguru import logger

from spheremix.distributions import SphericalCauchyDistribution
from spheremix.distributions import spherical_cauchy_logpdf
from spheremix.mixtures import SphericalCauchyMixture
from spheremix.utils.generators import orthogonal_sphere_modes

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
)


@click.command()
@click.option(
    "--n-components",
    "-k",
    default=3,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of mixture components (modes).",
)
@click.option(
    "--n-samples",
    "-n",
    default=200,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of samples to draw from each mode.",
)
@click.option(
    "--n-features",
    "-d",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
    help="Ambient dimension of the sphere S^{d-1}.",
)
@click.option(
    "--rho",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    help="Concentration parameter used when generating samples (0=uniform, →1=peaked).",
)
@click.option(
    "--init",
    default="random",
    show_default=True,
    type=click.Choice(["random", "kmeans"]),
    help="Mean initialization strategy for EM.",
)
@click.option(
    "--max-iter",
    default=100,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of EM iterations.",
)
@click.option(
    "--tol",
    default=1e-4,
    show_default=True,
    type=float,
    help="Absolute log-likelihood convergence tolerance for EM.",
)
@click.option(
    "--rtol",
    default=1e-6,
    show_default=True,
    type=float,
    help="Relative log-likelihood convergence tolerance for EM.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Log EM iteration progress.",
)
def main(
    n_components: int,
    n_samples: int,
    n_features: int,
    rho: float,
    init: str,
    max_iter: int,
    tol: float,
    rtol: float,
    seed: int,
    verbose: bool,
) -> None:
    """Fit a Spherical Cauchy mixture model to orthogonally-seeded samples."""
    rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Generate orthogonal modes and draw samples
    # ------------------------------------------------------------------
    logger.info(
        f"Generating {n_components} orthogonal modes "
        f"on S^{{{n_features - 1}}} with rho={rho}"
    )
    modes: np.ndarray = orthogonal_sphere_modes(n_components, n_features, rng=rng)

    parts: list[np.ndarray] = []
    for k in range(n_components):
        dist: SphericalCauchyDistribution = SphericalCauchyDistribution(
            mu=modes[k], rho=rho
        )
        parts.append(dist.sample(n_samples=n_samples))

    X: np.ndarray = np.vstack(parts)
    N: int = X.shape[0]
    logger.info(f"Generated {N} samples total  (shape {X.shape})")

    # Shuffle so EM does not see perfectly ordered blocks.
    X = X[rng.permutation(N)]

    # ------------------------------------------------------------------
    # 2. Fit the mixture model
    # ------------------------------------------------------------------
    logger.info(
        f"Fitting SphericalCauchyMixture — "
        f"K={n_components}, init={init}, max_iter={max_iter}, tol={tol}, rtol={rtol}"
    )
    model: SphericalCauchyMixture = SphericalCauchyMixture(
        n_components=n_components,
        init=init,  # type: ignore[arg-type]
    )
    t0: float = time.perf_counter()
    model.fit(X, max_iter=max_iter, tol=tol, rtol=rtol, verbose=verbose)
    fit_time: float = time.perf_counter() - t0
    logger.info(
        f"EM finished — iterations: {model.n_iter}, "
        f"log-likelihood: {model.log_likelihood:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Assign samples to clusters via the E-step responsibilities
    # ------------------------------------------------------------------
    assert model.weights is not None
    assert model.means is not None
    assert model.concentrations is not None

    resp: np.ndarray = model.expectation_step(X)  # (N, K)
    assignments: np.ndarray = resp.argmax(axis=1)  # (N,)

    # ------------------------------------------------------------------
    # 4. Log per-cluster statistics
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Per-cluster statistics")
    logger.info("-" * 60)

    for k in range(n_components):
        mask: np.ndarray = assignments == k
        n_assigned: int = int(mask.sum())

        weight_k: float = float(model.weights[k])
        mu_k: np.ndarray = model.means[k]
        rho_k: float = float(model.concentrations[k])

        logger.info(f"Component {k}:")
        logger.info(f"  weight      : {weight_k:.4f}")
        logger.info(f"  rho         : {rho_k:.4f}")
        logger.info(f"  mu          : {mu_k.round(4)}")
        logger.info(f"  n_assigned  : {n_assigned} ({100 * n_assigned / N:.1f}%)")

        if n_assigned == 0:
            logger.warning(
                f"  Component {k} has no assigned samples — skipping density stats."
            )
            continue

        log_dens: np.ndarray = spherical_cauchy_logpdf(X[mask], mu_k, rho_k)
        logger.info(
            f"  log-density : mean={log_dens.mean():.4f}  "
            f"std={log_dens.std():.4f}  "
            f"min={log_dens.min():.4f}  "
            f"max={log_dens.max():.4f}"
        )

    logger.info("-" * 60)
    logger.info(f"Fit time: {fit_time:.3f}s")


if __name__ == "__main__":
    main()
