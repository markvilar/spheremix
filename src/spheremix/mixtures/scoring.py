"""
Module for the prediction, scoring and model selection of spherical mixtures.
"""

import numpy as np

from .protocols import SphericalMixtureComponents


def compute_responsibilities(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> np.ndarray:
    """
    Compute the posterior probabilities p(z = k | x_n) of each sample.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    Responsibilities, shape (N, K), each row sums to 1.
    """
    log_joint: np.ndarray = components.log_joint_densities(X)
    log_marginal: np.ndarray = np.logaddexp.reduce(log_joint, axis=1, keepdims=True)
    return np.exp(log_joint - log_marginal)  # type: ignore[no-any-return]


def compute_log_likelihoods(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> np.ndarray:
    """
    Compute the observed-data log-likelihood log p(x_n) of each sample.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    Log-likelihoods, shape (N,).
    """
    log_joint: np.ndarray = components.log_joint_densities(X)
    return np.logaddexp.reduce(log_joint, axis=1)  # type: ignore[no-any-return]


def compute_aic(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> float:
    """
    Compute the Akaike Information Criterion. Lower is better.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    -2 log L + 2 p, with p the number of free parameters.
    """
    log_likelihood: float = float(compute_log_likelihoods(X, components).sum())
    return -2.0 * log_likelihood + 2.0 * components.free_parameter_count


def compute_bic(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> float:
    """
    Compute the Bayesian Information Criterion. Lower is better.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    -2 log L + p log N, with p the number of free parameters.
    """
    log_likelihood: float = float(compute_log_likelihoods(X, components).sum())
    penalty: float = components.free_parameter_count * float(np.log(X.shape[0]))
    return -2.0 * log_likelihood + penalty


def compute_icl(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> float:
    """
    Compute the Integrated Completed Likelihood. Lower is better.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    BIC + 2 EN, with EN the classification entropy of the responsibilities.
    """
    entropy: float = _classification_entropy(compute_responsibilities(X, components))
    return compute_bic(X, components) + 2.0 * entropy


def compute_bici(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> float:
    """
    Compute the complete-data BIC under hard MAP assignments. Lower is better.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    BIC - 2 sum_n log r_n, with r_n the responsibility of the MAP component.
    """
    responsibilities: np.ndarray = compute_responsibilities(X, components)
    map_responsibilities: np.ndarray = responsibilities.max(axis=1)
    log_map_probability: float = float(np.log(map_responsibilities).sum())
    return compute_bic(X, components) - 2.0 * log_map_probability


def compute_dic(
    X: np.ndarray,
    components: SphericalMixtureComponents,
) -> float:
    """
    Compute the complete-data Deviance Information Criterion. Lower is better.

    Arguments
    ---------
    X:
        Unit vectors, shape (N, D).
    components:
        Fitted mixture components.

    Returns
    -------
    -2 log L + 2 EN + 2 p, with EN the classification entropy and p the
    number of free parameters.
    """
    log_likelihood: float = float(compute_log_likelihoods(X, components).sum())
    entropy: float = _classification_entropy(compute_responsibilities(X, components))
    return -2.0 * log_likelihood + 2.0 * entropy + 2.0 * components.free_parameter_count


def _classification_entropy(responsibilities: np.ndarray) -> float:
    """
    Compute EN(R) = -sum_n sum_k R[n, k] log R[n, k] of responsibilities of
    shape (N, K), clipping near-zero values to avoid log(0).
    """
    log_responsibilities: np.ndarray = np.log(np.clip(responsibilities, 1e-300, 1.0))
    return float(-(responsibilities * log_responsibilities).sum())
