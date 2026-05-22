#!/usr/bin/env python3
"""
benchmark_mixture_models.py

Benchmark spherical mixture model families on real image vector data.
For each family, sweeps the component count over a range, selects the best
model by BIC, and produces a combined figure with three columns per family row:
  - PCA scatter of cluster assignments for the best model      (col 0)
  - Violin plot of per-point cosine similarity to cluster mean (col 1)
  - AIC / BIC line plot over the component range              (col 2)

Best model per family is selected by lowest BIC.

Usage
-----
    uv run scripts/benchmark_mixture_models.py
    uv run scripts/benchmark_mixture_models.py --k-min 3 --k-max 3
"""

import sys
import time

from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from spheremix.mixtures import (
    PKBMixture,
    PowerSphericalMixture,
    SphericalCauchyMixture,
)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
)

type AnyMixture = SphericalCauchyMixture | PKBMixture | PowerSphericalMixture

FAMILIES: list[tuple[str, type]] = [
    ("SphericalCauchy", SphericalCauchyMixture),
    ("PKBD", PKBMixture),
    ("PowerSpherical", PowerSphericalMixture),
]


@dataclass
class FamilyResult:
    """Aggregated fitting results for one model family across all K values."""

    family_name: str
    k_values: list[int]
    aics: list[float]
    bics: list[float]
    models: list[AnyMixture]

    @property
    def best_idx(self) -> int:
        return int(np.argmin(self.bics))

    @property
    def best_k(self) -> int:
        return self.k_values[self.best_idx]

    @property
    def best_model(self) -> AnyMixture:
        return self.models[self.best_idx]


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Return X with each row scaled to unit length."""
    norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)  # type: ignore[no-any-return]


def fit_family(
    family_name: str,
    model_class: type,
    X: np.ndarray,
    k_values: list[int],
    init: str,
    max_iter: int,
    tol: float,
    rtol: float,
) -> FamilyResult:
    """
    Fit a mixture model family for each K in k_values.

    Arguments
    ---------
    family_name: Display name of the model family.
    model_class: Mixture model class to instantiate (must follow SphericalMixtureModel).
    X: Unit-vector data, shape (N, D).
    k_values: Component counts to sweep.
    init: Mean initialization strategy ("random" or "kmeans").
    max_iter: Maximum number of EM iterations.
    tol: Absolute log-likelihood convergence tolerance.
    rtol: Relative log-likelihood convergence tolerance.

    Returns
    -------
    FamilyResult with k_values, AIC, BIC, and fitted models in the same order.
    """
    aics: list[float] = []
    bics: list[float] = []
    models: list[AnyMixture] = []

    for k in k_values:
        logger.info(f"[{family_name}] Fitting K={k} ...")
        model: AnyMixture = model_class(
            n_components=k,
            init=init,
            max_iter=max_iter,
            tol=tol,
            rtol=rtol,
        )
        t0: float = time.perf_counter()
        model.fit(X)
        elapsed: float = time.perf_counter() - t0
        aic: float = model.aic(X)
        bic: float = model.bic(X)
        logger.info(
            f"[{family_name}] K={k}  loglik={model.log_likelihood_:.2f}  "
            f"AIC={aic:.2f}  BIC={bic:.2f}  "
            f"iter={model.n_iter_}  time={elapsed:.1f}s"
        )
        aics.append(aic)
        bics.append(bic)
        models.append(model)

    return FamilyResult(
        family_name=family_name,
        k_values=k_values,
        aics=aics,
        bics=bics,
        models=models,
    )


def cluster_cosine_similarities(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    n_components: int,
) -> list[np.ndarray]:
    """
    For each cluster, return per-point cosine similarity to the fitted cluster mean.

    Arguments
    ---------
    X: Unit-vector data, shape (N, D).
    labels: Cluster assignment per sample, shape (N,).
    means: Fitted cluster mean directions, shape (K, D).
    n_components: Number of components K.

    Returns
    -------
    List of K arrays; element k contains cosine similarities for cluster-k members.
    Empty clusters are represented by a two-element array of zeros (violinplot minimum).
    """
    similarities: list[np.ndarray] = []
    for k in range(n_components):
        mask: np.ndarray = labels == k
        n_members: int = int(mask.sum())
        if n_members == 0:
            similarities.append(np.array([0.0, 0.0]))
        elif n_members == 1:
            sim_val: float = float(X[mask] @ means[k])
            similarities.append(np.array([sim_val, sim_val]))
        else:
            similarities.append(X[mask] @ means[k])
    return similarities


def plot_pca_scatter(
    ax: Axes,
    X_pca: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    family_name: str,
    best_k: int,
) -> None:
    """Scatter PC1 vs PC2 colored by cluster assignment."""
    cmap = plt.colormaps["tab10"]
    for k in range(n_components):
        mask: np.ndarray = labels == k
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=5,
            alpha=0.4,
            color=cmap(k % 10),
            label=f"C{k}",
            rasterized=True,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{family_name}  K={best_k}  PCA")
    ax.legend(markerscale=2, fontsize="small", loc="best")


def plot_cosine_violin(
    ax: Axes,
    similarities: list[np.ndarray],
    family_name: str,
    best_k: int,
) -> None:
    """Violin of per-point cosine similarity to cluster mean, one violin per cluster."""
    cmap = plt.colormaps["tab10"]
    positions: list[int] = list(range(best_k))
    parts = ax.violinplot(similarities, positions=positions, showmedians=True)
    for idx, body in enumerate(parts["bodies"]):  # type: ignore[index]
        body.set_facecolor(cmap(idx % 10))
        body.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{k}" for k in range(best_k)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Cosine similarity to μₖ")
    ax.set_title(f"{family_name}  K={best_k}  Within-cluster cosine sim")


def plot_aic_bic(
    ax: Axes,
    k_values: list[int],
    aics: list[float],
    bics: list[float],
    best_k: int,
    family_name: str,
) -> None:
    """Line plot of AIC and BIC over K, with the best-BIC K marked."""
    ax.plot(k_values, aics, marker="o", linestyle="-", label="AIC", color="steelblue")
    ax.plot(k_values, bics, marker="s", linestyle="--", label="BIC", color="coral")
    ax.axvline(
        best_k,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"best K={best_k}",
    )
    ax.set_xlabel("K (components)")
    ax.set_ylabel("Criterion value")
    ax.set_title(f"{family_name}  AIC / BIC")
    ax.set_xticks(k_values)
    ax.legend()


@click.command()
@click.option(
    "--data",
    "data_path",
    default="data/image_vectors_mono_d128_site2.npy",
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the .npy file of image vectors.",
)
@click.option(
    "--k-min",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Minimum number of mixture components (inclusive).",
)
@click.option(
    "--k-max",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of mixture components (inclusive).",
)
@click.option(
    "--init",
    default="kmeans",
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
    help="Absolute log-likelihood convergence tolerance.",
)
@click.option(
    "--rtol",
    default=1e-6,
    show_default=True,
    type=float,
    help="Relative log-likelihood convergence tolerance.",
)
@click.option(
    "--output-dir",
    default="output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to write output figures.",
)
def main(
    data_path: Path,
    k_min: int,
    k_max: int,
    init: str,
    max_iter: int,
    tol: float,
    rtol: float,
    output_dir: Path,
) -> None:
    """Benchmark spherical mixture models on image vector data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_path}")
    X_raw: np.ndarray = np.load(data_path)
    X: np.ndarray = normalize_rows(X_raw)
    logger.info(f"Data shape: {X.shape}  (L2-normalized to unit sphere)")

    k_values: list[int] = list(range(k_min, k_max + 1))
    logger.info(f"Sweeping K ∈ {k_values} across {len(FAMILIES)} model families")

    pca: PCA = PCA(n_components=2)
    X_pca: np.ndarray = pca.fit_transform(X)
    logger.info(
        f"PCA explained variance: "
        f"PC1={pca.explained_variance_ratio_[0]:.3f}  "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}"
    )

    n_families: int = len(FAMILIES)
    fig, axes = plt.subplots(
        n_families,
        3,
        figsize=(18, 5 * n_families),
        squeeze=False,
    )
    fig.suptitle(
        f"Mixture model benchmark — {data_path.name}  K∈{k_values}",
        fontsize=14,
        y=1.01,
    )

    results: list[FamilyResult] = []

    for row_idx, (family_name, model_class) in enumerate(FAMILIES):
        logger.info(f"=== Family: {family_name} ===")
        result: FamilyResult = fit_family(
            family_name=family_name,
            model_class=model_class,
            X=X,
            k_values=k_values,
            init=init,
            max_iter=max_iter,
            tol=tol,
            rtol=rtol,
        )
        results.append(result)

        best_model: AnyMixture = result.best_model
        best_k: int = result.best_k
        labels: np.ndarray = best_model.predict(X)
        means: np.ndarray = best_model.get_params().means

        similarities: list[np.ndarray] = cluster_cosine_similarities(
            X, labels, means, best_k
        )

        plot_pca_scatter(axes[row_idx, 0], X_pca, labels, best_k, family_name, best_k)
        plot_cosine_violin(axes[row_idx, 1], similarities, family_name, best_k)
        plot_aic_bic(
            axes[row_idx, 2],
            k_values,
            result.aics,
            result.bics,
            best_k,
            family_name,
        )

    fig.tight_layout()
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path: Path = (
        output_dir / f"{timestamp}_mixture_benchmark_{data_path.stem}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved to {output_path}")
    plt.close(fig)

    logger.info("=" * 60)
    logger.info("Best models by BIC:")
    logger.info("=" * 60)
    for result in results:
        best: AnyMixture = result.best_model
        params = best.get_params()
        logger.info(
            f"  {result.family_name:<20}  K={result.best_k}  "
            f"loglik={best.log_likelihood_:.2f}  "
            f"BIC={result.bics[result.best_idx]:.2f}  "
            f"weights={params.weights.round(3)}"
        )


if __name__ == "__main__":
    main()
