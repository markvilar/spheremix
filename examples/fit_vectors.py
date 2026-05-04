"""
fit_vectors.py

Load unit vectors from a .npy file (shape N×D) and fit either a Spherical
Cauchy mixture model or Spherical K-means to them.

Usage
-----
    python examples/fit_vectors.py cauchy --input vectors.npy --n-components 5
    python examples/fit_vectors.py kmeans --input vectors.npy --n-components 10
"""

import sys
import time

import click
import numpy as np
from loguru import logger

from spheremix.cluster import SphericalKMeans
from spheremix.distributions import spherical_cauchy_logpdf
from spheremix.mixtures import SphericalCauchyMixture

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
)


def _load_and_validate(input_path: str, normalize: bool) -> np.ndarray:
    logger.info(f"Loading vectors from '{input_path}'")
    X: np.ndarray = np.load(input_path)

    if X.ndim != 2:
        logger.error(f"Expected a 2-D array (N×D), got shape {X.shape}.")
        sys.exit(1)

    N, D = X.shape
    logger.info(f"Loaded array — N={N} samples, D={D} features")

    if normalize:
        logger.info("L2-normalizing rows to unit sphere.")
        norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)  # type: ignore[assignment]

    return X


def _log_cauchy_stats(
    X: np.ndarray,
    model: SphericalCauchyMixture,
    assignments: np.ndarray,
) -> None:
    assert model.weights is not None
    assert model.means is not None
    assert model.concentrations is not None

    N: int = X.shape[0]
    K: int = model.n_components

    logger.info("-" * 60)
    logger.info("Per-component statistics")
    logger.info("-" * 60)

    for k in range(K):
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
            logger.warning(f"  Component {k} has no assigned samples — skipping density stats.")
            continue

        log_dens: np.ndarray = spherical_cauchy_logpdf(X[mask], mu_k, rho_k)
        logger.info(
            f"  log-density : mean={log_dens.mean():.4f}  "
            f"std={log_dens.std():.4f}  "
            f"min={log_dens.min():.4f}  "
            f"max={log_dens.max():.4f}"
        )


def _log_kmeans_stats(
    X: np.ndarray,
    model: SphericalKMeans,
    labels: np.ndarray,
) -> None:
    assert model.cluster_means is not None

    N: int = X.shape[0]
    K: int = model.n_clusters

    logger.info("-" * 60)
    logger.info("Per-cluster statistics")
    logger.info("-" * 60)

    for k in range(K):
        mask: np.ndarray = labels == k
        n_assigned: int = int(mask.sum())
        centre_k: np.ndarray = model.cluster_means[k]

        logger.info(f"Cluster {k}:")
        logger.info(f"  centre      : {centre_k.round(4)}")
        logger.info(f"  n_assigned  : {n_assigned} ({100 * n_assigned / N:.1f}%)")

        if n_assigned == 0:
            logger.warning(f"  Cluster {k} has no assigned samples — skipping stats.")
            continue

        X_k: np.ndarray = X[mask]
        cos_sim: np.ndarray = X_k @ centre_k
        inertia_k: float = float((1.0 - cos_sim).sum())

        logger.info(
            f"  cos-similarity : mean={cos_sim.mean():.4f}  "
            f"std={cos_sim.std():.4f}  "
            f"min={cos_sim.min():.4f}  "
            f"max={cos_sim.max():.4f}"
        )
        logger.info(f"  inertia        : {inertia_k:.4f}")


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_input_option = click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the input .npy file containing an N×D array.",
)
_n_components_option = click.option(
    "--n-components", "-k",
    required=True,
    type=click.IntRange(min=1),
    help="Number of components / clusters.",
)
_normalize_option = click.option(
    "--normalize/--no-normalize",
    default=False,
    show_default=True,
    help="L2-normalize each row to unit length before fitting.",
)
_seed_option = click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
_verbose_option = click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Log fitting progress.",
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Fit a spherical model to unit vectors loaded from a .npy file."""


# ---------------------------------------------------------------------------
# cauchy subcommand
# ---------------------------------------------------------------------------

@cli.command()
@_input_option
@_n_components_option
@_normalize_option
@click.option(
    "--init",
    default="random",
    show_default=True,
    type=click.Choice(["random", "kmeans"]),
    help="Mean initialization strategy.",
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
    help="Absolute log-likelihood convergence tolerance.",
)
@click.option(
    "--rtol",
    default=1e-6,
    show_default=True,
    type=float,
    help="Relative log-likelihood convergence tolerance.",
)
@_seed_option
@_verbose_option
def cauchy(
    input_path: str,
    n_components: int,
    normalize: bool,
    init: str,
    max_iter: int,
    tol: float,
    rtol: float,
    seed: int,
    verbose: bool,
) -> None:
    """Fit a Spherical Cauchy mixture model."""
    X: np.ndarray = _load_and_validate(input_path, normalize)

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

    resp: np.ndarray = model.expectation_step(X)
    assignments: np.ndarray = resp.argmax(axis=1)

    _log_cauchy_stats(X, model, assignments)

    logger.info("-" * 60)
    logger.info(f"Fit time: {fit_time:.3f}s")


# ---------------------------------------------------------------------------
# kmeans subcommand
# ---------------------------------------------------------------------------

@cli.command()
@_input_option
@_n_components_option
@_normalize_option
@click.option(
    "--init",
    default="kmeans++",
    show_default=True,
    type=click.Choice(["kmeans++", "random"]),
    help="Centre initialisation strategy.",
)
@click.option(
    "--n-init",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of random restarts; the best result is kept.",
)
@click.option(
    "--max-iter",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of Lloyd iterations per restart.",
)
@click.option(
    "--tol",
    default=1e-4,
    show_default=True,
    type=float,
    help="Convergence tolerance on centre shift.",
)
@_seed_option
@_verbose_option
def kmeans(
    input_path: str,
    n_components: int,
    normalize: bool,
    init: str,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
    verbose: bool,
) -> None:
    """Fit Spherical K-means."""
    X: np.ndarray = _load_and_validate(input_path, normalize)
    rng: np.random.Generator = np.random.default_rng(seed)

    logger.info(
        f"Fitting SphericalKMeans — "
        f"K={n_components}, init={init}, n_init={n_init}, max_iter={max_iter}"
    )
    model: SphericalKMeans = SphericalKMeans(
        n_clusters=n_components,
        init=init,  # type: ignore[arg-type]
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        rng=rng,
        verbose=verbose,
    )

    t0: float = time.perf_counter()
    labels: np.ndarray = model.fit_predict(X)
    fit_time: float = time.perf_counter() - t0

    assert model.inertia_ is not None
    assert model.n_iter_ is not None

    logger.info(
        f"Fitting done — iterations: {model.n_iter_}, "
        f"total inertia: {model.inertia_:.4f}"
    )

    _log_kmeans_stats(X, model, labels)

    logger.info("-" * 60)
    logger.info(f"Fit time: {fit_time:.3f}s")


if __name__ == "__main__":
    cli()
