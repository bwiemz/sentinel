"""Track quality metrics: NIS, NEES, and filter consistency monitoring.

Provides online filter health assessment without ground truth (NIS)
and simulation-mode validation with ground truth (NEES). These are
the canonical statistical measures used in defense-grade tracking
systems (Aegis, SPY-6, IBCS) to detect filter divergence, model
mismatch, and over/under-confidence.

NIS (Normalized Innovation Squared):
    NIS_k = y_k' * S_k^{-1} * y_k
    Expected: E[NIS] = dim_meas under correct model.

NEES (Normalized Estimation Error Squared):
    NEES_k = (x_true - x_est)' * P_k^{-1} * (x_true - x_est)
    Expected: E[NEES] = dim_state under correct model.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def compute_nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
    """Compute Normalized Innovation Squared.

    NIS = y' * S^{-1} * y

    Under a correctly specified filter, NIS ~ chi-squared(dim_meas).
    Expected value: dim_meas.

    Args:
        innovation: Innovation vector y = z - h(x_predicted).
        innovation_cov: Innovation covariance S = H*P*H' + R.

    Returns:
        NIS value (non-negative scalar).
    """
    y = np.asarray(innovation, dtype=float)
    S = np.asarray(innovation_cov, dtype=float)
    try:
        return float(y @ np.linalg.solve(S, y))
    except np.linalg.LinAlgError:
        return float("inf")


def compute_nees(
    x_true: np.ndarray,
    x_est: np.ndarray,
    P: np.ndarray,
) -> float:
    """Compute Normalized Estimation Error Squared (simulation mode).

    NEES = (x_true - x_est)' * P^{-1} * (x_true - x_est)

    Under a correctly specified filter, NEES ~ chi-squared(dim_state).
    Expected value: dim_state.

    Args:
        x_true: True state vector.
        x_est: Estimated state vector.
        P: Estimation error covariance matrix.

    Returns:
        NEES value (non-negative scalar).
    """
    e = np.asarray(x_true, dtype=float) - np.asarray(x_est, dtype=float)
    P = np.asarray(P, dtype=float)
    try:
        return float(e @ np.linalg.solve(P, e))
    except np.linalg.LinAlgError:
        return float("inf")


class FilterConsistencyMonitor:
    """Monitors Kalman filter consistency using NIS statistics.

    Maintains a rolling window of NIS values and triggers alerts when
    the filter becomes inconsistent. This is the online (no ground truth)
    equivalent of NEES validation.

    Filter health states:
    - nominal: NIS ratio in [under_threshold, over_threshold]
    - over_confident: NIS ratio > over_threshold (filter trusts itself too much)
    - under_confident: NIS ratio < under_threshold (filter is too conservative)
    - diverged: NIS ratio > 3 * over_threshold (filter has lost the target)

    Args:
        dim_meas: Measurement dimension (determines expected NIS value).
        window_size: Rolling window size for NIS averaging.
        over_confident_threshold: NIS ratio above which filter is over-confident.
        under_confident_threshold: NIS ratio below which filter is too conservative.
    """

    def __init__(
        self,
        dim_meas: int = 2,
        window_size: int = 20,
        over_confident_threshold: float = 2.0,
        under_confident_threshold: float = 0.3,
    ):
        self._dim_meas = max(dim_meas, 1)
        self._window_size = max(window_size, 1)
        self._over_threshold = over_confident_threshold
        self._under_threshold = under_confident_threshold
        self._nis_window: deque[float] = deque(maxlen=self._window_size)

    def record_innovation(
        self,
        innovation: np.ndarray,
        innovation_cov: np.ndarray,
    ) -> float:
        """Record one innovation and its covariance, compute and store NIS.

        Args:
            innovation: Innovation vector y.
            innovation_cov: Innovation covariance S.

        Returns:
            The computed NIS value.
        """
        nis = compute_nis(innovation, innovation_cov)
        if np.isfinite(nis):
            self._nis_window.append(nis)
        return nis

    @property
    def sample_count(self) -> int:
        """Number of NIS samples recorded."""
        return len(self._nis_window)

    @property
    def average_nis(self) -> float:
        """Average NIS over the rolling window.

        Expected value under correct model: dim_meas.
        """
        if not self._nis_window:
            return float(self._dim_meas)  # Default to expected value
        return float(np.mean(self._nis_window))

    @property
    def nis_ratio(self) -> float:
        """Ratio of average NIS to expected value.

        Should be approximately 1.0 for a well-tuned filter.
        > 1: filter is over-confident (underestimates uncertainty)
        < 1: filter is under-confident (overestimates uncertainty)
        """
        expected = float(self._dim_meas)
        if expected <= 0:
            return 1.0
        return self.average_nis / expected

    @property
    def filter_health(self) -> str:
        """Categorized filter health status.

        Returns one of: 'nominal', 'over_confident', 'under_confident', 'diverged'.
        """
        if self.sample_count < 3:
            return "nominal"  # Not enough data to judge

        ratio = self.nis_ratio

        if ratio > 3.0 * self._over_threshold:
            return "diverged"
        if ratio > self._over_threshold:
            return "over_confident"
        if ratio < self._under_threshold:
            return "under_confident"
        return "nominal"

    @property
    def consistency_score(self) -> float:
        """Quality score in [0, 1] based on statistical consistency.

        1.0 = perfectly consistent filter (NIS ratio ~1.0).
        0.0 = severely inconsistent / diverged.

        Uses a Gaussian-shaped scoring function centered at ratio=1.0.
        """
        if self.sample_count < 3:
            return 1.0  # Assume good until proven otherwise

        ratio = self.nis_ratio
        # Gaussian decay: score = exp(-0.5 * ((ratio - 1) / sigma)^2)
        # sigma = 0.5 gives reasonable sensitivity
        deviation = ratio - 1.0
        score = float(np.exp(-0.5 * (deviation / 0.5) ** 2))
        return max(0.0, min(1.0, score))

    def reset(self) -> None:
        """Clear the NIS history."""
        self._nis_window.clear()
