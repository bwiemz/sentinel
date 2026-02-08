"""State-level fusion algorithms.

Provides Covariance Intersection (CI) and Information Fusion for
combining state estimates from independent sensors without requiring
knowledge of cross-correlations.

CI is conservative (never overestimates), making it suitable for
safety-critical tracking where sensor error correlations are unknown.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar


def _spd_inv(P: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-definite matrix via Cholesky decomposition."""
    c, low = cho_factor(P)
    return cho_solve((c, low), np.eye(P.shape[0]))


def _spd_solve(P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve P @ x = b for SPD matrix P via Cholesky."""
    c, low = cho_factor(P)
    return cho_solve((c, low), b)


def covariance_intersection(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariance Intersection (CI) fusion.

    Fuses two state estimates without knowing cross-correlations.
    Finds the optimal mixing weight omega that minimizes the trace
    of the fused covariance.

    Args:
        x1, P1: First estimate (state, covariance).
        x2, P2: Second estimate (state, covariance).

    Returns:
        (x_fused, P_fused): Fused state and covariance.
        Guarantees P_fused <= P1 and P_fused <= P2 (in PSD sense).
    """
    omega = _optimize_omega(P1, P2)
    return _ci_with_omega(x1, P1, x2, P2, omega)


def _ci_with_omega(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CI with a given omega."""
    P1_inv = _spd_inv(P1)
    P2_inv = _spd_inv(P2)

    P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
    P_fused = _spd_inv(P_fused_inv)

    x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)
    return x_fused, P_fused


def _optimize_omega(P1: np.ndarray, P2: np.ndarray) -> float:
    """Find omega in [0, 1] that minimizes trace(P_fused).

    Uses scalar optimization since omega is 1D.
    """
    P1_inv = _spd_inv(P1)
    P2_inv = _spd_inv(P2)

    def objective(omega: float) -> float:
        P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
        P_fused = _spd_inv(P_fused_inv)
        return np.trace(P_fused)

    result = minimize_scalar(objective, bounds=(0.01, 0.99), method="bounded")
    return float(result.x)


def information_fusion(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Information (naive) fusion for independent estimates.

    Assumes the two estimates are truly independent (zero cross-correlation).
    More aggressive than CI -- produces tighter covariance, but incorrect
    if the estimates share common information.

    Args:
        x1, P1: First estimate.
        x2, P2: Second estimate.

    Returns:
        (x_fused, P_fused): Fused state and covariance.
    """
    P1_inv = _spd_inv(P1)
    P2_inv = _spd_inv(P2)

    P_fused_inv = P1_inv + P2_inv
    P_fused = _spd_inv(P_fused_inv)

    x_fused = P_fused @ (P1_inv @ x1 + P2_inv @ x2)
    return x_fused, P_fused
