"""
pca_reduce.py

Read a NumPy array (shape N×D) from a .npy file,
apply PCA to reduce it to shape N×C (C << D),
optionally L2-normalize each row to the unit sphere,
and write the result back to a .npy file.

Usage
-----
    python pca_reduce.py --input data.npy --output data_pca.npy --components 50
    python pca_reduce.py --input data.npy --output data_pca.npy --components 50 --scale --normalize
"""

import sys

import click
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Default loguru config: remove the default handler and add a clean stderr sink.
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize each row of X to unit length.

    Parameters
    ----------
    X : np.ndarray of shape (N, C)
        Vectors to normalize.
    eps : float
        Small constant added to norms to avoid division by zero.

    Returns
    -------
    np.ndarray of shape (N, C)
        Row-normalized array where each vector has ‖x‖₂ = 1.
    """
    norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the input .npy file containing an N×D array.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path where the reduced N×C array will be saved as a .npy file.",
)
@click.option(
    "--components",
    "-c",
    required=True,
    type=click.IntRange(min=1),
    help="Number of principal components to keep (C). Must be less than D.",
)
@click.option(
    "--scale/--no-scale",
    default=False,
    show_default=True,
    help="Standardize features to zero mean and unit variance before PCA.",
)
@click.option(
    "--normalize/--no-normalize",
    default=False,
    show_default=True,
    help="L2-normalize each row to unit length after PCA reduction.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def main(
    input_path: str,
    output_path: str,
    components: int,
    scale: bool,
    normalize: bool,
    verbose: bool,
) -> None:
    """Reduce dimensionality of a .npy array with PCA."""
    # Enable DEBUG level when --verbose is passed
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> <level>{level:<8}</level> {message}",
        )

    # 1) Load data
    logger.info(f"Loading data from '{input_path}'")
    X: np.ndarray = np.load(input_path)

    if X.ndim != 2:
        logger.error(f"Expected a 2-D array (N×D), got shape {X.shape}.")
        sys.exit(1)

    N, D = X.shape
    logger.info(f"Loaded array — N={N} samples, D={D} features")

    if components >= D:
        logger.error(
            f"--components ({components}) must be less than the number of features ({D})."
        )
        sys.exit(1)

    # 2) Optional standardization
    if scale:
        logger.info("Standardizing data (zero mean, unit variance).")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 3) PCA
    logger.info(f"Fitting PCA — keeping {components} components out of {D}.")
    pca = PCA(n_components=components, svd_solver="auto")
    X_reduced: np.ndarray = pca.fit_transform(X)  # shape (N, C)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"Explained variance ratio: {explained:.4f} "
        f"({components}/{D} components retained)"
    )
    logger.debug(f"Per-component variance: {pca.explained_variance_ratio_}")
    logger.info(f"Reduced array shape: {X_reduced.shape}")

    # 4) Optional L2 normalization
    if normalize:
        logger.info("L2-normalizing rows to unit length.")
        X_reduced = l2_normalize(X_reduced)
        norms = np.linalg.norm(X_reduced, axis=1)
        logger.debug(
            f"Post-normalization norm stats — "
            f"min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}"
        )

    # 5) Save
    logger.info(f"Saving array to '{output_path}'")
    np.save(output_path, X_reduced)
    logger.success("Done.")


if __name__ == "__main__":
    main()
