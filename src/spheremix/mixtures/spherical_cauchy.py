"""
Module for Spherical Cauchy mixture model.
"""

import numpy as np

from spheremix.distributions import log_spherical_cauchy_pdf


def normalize_rows(X: np.ndarray, eps=1e-12) -> np.ndarray:
    """L2-normalize each row of X."""
    norms: np.ndarray = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms


def spherical_cauchy_em(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    init_rho: float = 0.5,
    random_state: int | None = None,
    verbose: bool = False,
) -> tuple:
    """
    EM for mixture of spherical Cauchy distributions.

    Parameters
    ----------
    X : array, shape (N, D)
        L2-normalized data points (each row on unit sphere).
    K : int
        Number of mixture components.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Relative change in log-likelihood for convergence.
    init_rho : float
        Initial rho for all components.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        If True, prints log-likelihood per iteration.

    Returns
    -------
    pi : (K,) mixture weights
    mu : (K, D) component mean directions (unit vectors)
    rho : (K,) component concentration parameters in [0, 1)
    resp : (N, K) final responsibilities
    log_lik_history : list of log-likelihood values
    """
    rng = np.random.default_rng(random_state)
    X = normalize_rows(X)  # ensure on sphere
    N, D = X.shape

    # Initialize parameters
    # Randomly pick K data points as initial means
    indices = rng.choice(N, size=K, replace=False)
    mu = X[indices].copy()  # (K, D)
    mu = normalize_rows(mu)
    rho = np.full(K, init_rho, dtype=float)
    pi = np.full(K, 1.0 / K, dtype=float)

    log_lik_history = []

    for it in range(max_iter):
        # -------- E-step --------
        # log p(x_n | k)
        log_pdf = np.zeros((N, K))
        for k in range(K):
            log_pdf[:, k] = log_spherical_cauchy_pdf(X, mu[k], rho[k])

        # add log mixing weights
        log_pi = np.log(pi + 1e-12)
        log_joint = log_pdf + log_pi  # (N, K)

        # log-sum-exp over components
        max_log = np.max(log_joint, axis=1, keepdims=True)
        log_sum = max_log + np.log(
            np.sum(np.exp(log_joint - max_log), axis=1, keepdims=True)
        )
        log_resp = log_joint - log_sum  # (N, K)
        resp = np.exp(log_resp)  # (N, K)

        # log-likelihood
        log_lik = np.sum(log_sum)
        log_lik_history.append(log_lik)

        if verbose:
            print(f"Iter {it:3d}: log-lik = {log_lik:.6f}")

        # check convergence
        if it > 0:
            rel_change = (log_lik - log_lik_history[-2]) / (
                abs(log_lik_history[-2]) + 1e-12
            )
            if rel_change < tol:
                if verbose:
                    print("Converged.")
                break

        # -------- M-step --------
        Nk = resp.sum(axis=0) + 1e-12  # effective counts, (K,)

        # mixture weights
        pi = Nk / N

        # update mu_k (weighted mean direction)
        # mu_k = argmax_mu sum_n r_nk log f(x_n | mu, rho_k)
        # with our approximate objective we use weighted Euclidean mean then normalize
        for k in range(K):
            # weighted sum
            w_sum = (resp[:, k][:, None] * X).sum(axis=0)  # (D,)
            norm = np.linalg.norm(w_sum)
            if norm < 1e-12:
                # degenerate; reinitialize randomly
                mu[k] = X[rng.integers(N)]
            else:
                mu[k] = w_sum / norm

        # update rho_k by 1D optimization per component
        # we do simple fixed-point / gradient steps; for better robustness
        # you might want to replace this with a bounded scalar optimizer (e.g., Brent).
        for k in range(K):
            rho_k = rho[k]
            # simple fixed-point / gradient-like update (few iterations)
            for _ in range(10):
                # derivative of log-lik wrt rho (approximate)
                # log f(x) = log(1 - rho^2) - (D-1)/2 * log(1 + rho^2 - 2 rho x^T mu)
                # d/d rho log f(x) = (-2 rho)/(1 - rho^2)
                #                    - (D-1)/2 * (2 rho - 2 x^T mu) / (1 + rho^2 - 2 rho x^T mu)
                dot = X @ mu[k]  # (N,)
                sq = 1.0 + rho_k**2 - 2.0 * rho_k * dot
                sq = np.maximum(sq, 1e-12)
                term1 = (-2.0 * rho_k) / np.maximum(1.0 - rho_k**2, 1e-12)
                term2 = -0.5 * (D - 1) * (2.0 * rho_k - 2.0 * dot) / sq
                dlogf_drho = term1 + term2  # (N,)
                grad = np.sum(resp[:, k] * dlogf_drho)

                # simple gradient step with small learning rate, projected to [0, 1)
                step = 1e-3 * grad / (Nk[k] + 1e-12)
                rho_k_new = np.clip(rho_k + step, 1e-6, 1 - 1e-6)

                if np.abs(rho_k_new - rho_k) < 1e-5:
                    rho_k = rho_k_new
                    break
                rho_k = rho_k_new

            rho[k] = rho_k

    return pi, mu, rho, resp, log_lik_history
