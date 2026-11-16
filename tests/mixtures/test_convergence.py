"""
Tests for SphericalMixtureConvergence.
"""

import pytest

from spheremix.mixtures import SphericalMixtureConvergence


def test_convergence_keeps_history() -> None:
    convergence: SphericalMixtureConvergence = SphericalMixtureConvergence(
        log_likelihood=-1.0,
        n_iter=2,
        converged=True,
        log_likelihood_history=(-3.0, -1.0),
    )

    assert convergence.log_likelihood_history == (-3.0, -1.0)


@pytest.mark.parametrize("history", [(), (-1.0,), (-3.0, -2.0, -1.0)])
def test_convergence_rejects_history_of_wrong_length(
    history: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="must have n_iter = 2 values"):
        SphericalMixtureConvergence(
            log_likelihood=-1.0,
            n_iter=2,
            converged=True,
            log_likelihood_history=history,
        )


def test_convergence_rejects_history_ending_elsewhere() -> None:
    with pytest.raises(ValueError, match="must equal log_likelihood"):
        SphericalMixtureConvergence(
            log_likelihood=-1.0,
            n_iter=2,
            converged=True,
            log_likelihood_history=(-3.0, -2.0),
        )
