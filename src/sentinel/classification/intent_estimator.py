"""Kinematic-based intent/behavior estimation for tracked targets.

Uses geometry-based heuristics (approach rate, CPA, speed profile)
rather than ML — deterministic physics is more trustworthy here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.core.types import IntentType


@dataclass
class IntentEstimate:
    """Result of intent estimation."""

    intent: IntentType
    confidence: float
    time_to_cpa_s: float | None = None
    cpa_range_m: float | None = None


class IntentEstimator:
    """Estimates target intent from kinematics.

    Args:
        sensor_position: Sensor position [x, y] in meters (default: origin).
        approach_rate_threshold_mps: Min approach rate to classify as APPROACH.
        attack_speed_threshold_mps: Min speed + approach rate for ATTACK.
        patrol_speed_threshold_mps: Max speed for PATROL.
        min_track_age: Minimum track age before estimating intent.
    """

    def __init__(
        self,
        sensor_position: np.ndarray | None = None,
        approach_rate_threshold_mps: float = 20.0,
        attack_speed_threshold_mps: float = 300.0,
        patrol_speed_threshold_mps: float = 80.0,
        min_track_age: int = 5,
    ):
        self._sensor_pos = (
            np.asarray(sensor_position, dtype=np.float64)
            if sensor_position is not None
            else np.zeros(2)
        )
        self._approach_thresh = approach_rate_threshold_mps
        self._attack_speed = attack_speed_threshold_mps
        self._patrol_speed = patrol_speed_threshold_mps
        self._min_age = min_track_age

    def estimate(self, eft) -> IntentEstimate:
        """Estimate intent for a single fused track."""
        # Check minimum track age
        best_age = self._get_best_age(eft)
        if best_age < self._min_age:
            return IntentEstimate(intent=IntentType.UNKNOWN, confidence=0.0)

        pos = self._get_position(eft)
        vel = self._get_velocity(eft)

        if pos is None or vel is None:
            return IntentEstimate(intent=IntentType.UNKNOWN, confidence=0.0)

        speed = float(np.linalg.norm(vel))
        approach_rate = self._compute_approach_rate(pos, vel)
        t_cpa, r_cpa = self._compute_cpa(pos, vel)

        # Decision logic
        if approach_rate > self._approach_thresh and speed > self._attack_speed:
            return IntentEstimate(
                intent=IntentType.ATTACK,
                confidence=min(1.0, approach_rate / (self._attack_speed * 2)),
                time_to_cpa_s=t_cpa,
                cpa_range_m=r_cpa,
            )

        if approach_rate > self._approach_thresh:
            conf = min(1.0, approach_rate / (self._approach_thresh * 5))
            return IntentEstimate(
                intent=IntentType.APPROACH,
                confidence=conf,
                time_to_cpa_s=t_cpa,
                cpa_range_m=r_cpa,
            )

        if approach_rate < -self._approach_thresh:
            conf = min(1.0, abs(approach_rate) / (self._approach_thresh * 5))
            return IntentEstimate(
                intent=IntentType.EVASION,
                confidence=conf,
                time_to_cpa_s=t_cpa,
                cpa_range_m=r_cpa,
            )

        if speed < self._patrol_speed and abs(approach_rate) < self._approach_thresh:
            return IntentEstimate(
                intent=IntentType.PATROL,
                confidence=0.6,
                time_to_cpa_s=t_cpa,
                cpa_range_m=r_cpa,
            )

        return IntentEstimate(
            intent=IntentType.TRANSIT,
            confidence=0.5,
            time_to_cpa_s=t_cpa,
            cpa_range_m=r_cpa,
        )

    def _compute_approach_rate(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> float:
        """Positive = closing toward sensor."""
        rel_pos = position - self._sensor_pos
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-6:
            return 0.0
        pos_unit = rel_pos / dist
        return -float(np.dot(velocity, pos_unit))

    def _compute_cpa(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> tuple[float | None, float | None]:
        """Compute time-to-CPA and range-at-CPA."""
        rel_pos = position - self._sensor_pos
        speed_sq = float(np.dot(velocity, velocity))
        if speed_sq < 1e-12:
            return None, float(np.linalg.norm(rel_pos))

        t_cpa = -float(np.dot(rel_pos, velocity)) / speed_sq
        if not np.isfinite(t_cpa):
            return None, float(np.linalg.norm(rel_pos))
        if t_cpa < 0:
            # CPA is in the past; current range is closest
            return 0.0, float(np.linalg.norm(rel_pos))

        cpa_pos = rel_pos + t_cpa * velocity
        return t_cpa, float(np.linalg.norm(cpa_pos))

    @staticmethod
    def _get_position(eft) -> np.ndarray | None:
        if eft.position_m is not None:
            return np.asarray(eft.position_m, dtype=np.float64)
        for track in [eft.radar_track, eft.thermal_track, eft.quantum_radar_track]:
            if track is not None:
                p = track.position
                if p is not None:
                    return np.asarray(p, dtype=np.float64)
        return None

    @staticmethod
    def _get_velocity(eft) -> np.ndarray | None:
        for track in [eft.radar_track, eft.thermal_track, eft.quantum_radar_track]:
            if track is not None:
                v = track.velocity
                if v is not None:
                    return np.asarray(v, dtype=np.float64)
        return None

    @staticmethod
    def _get_best_age(eft) -> int:
        ages = []
        for track in [
            eft.camera_track,
            eft.radar_track,
            eft.thermal_track,
            eft.quantum_radar_track,
        ]:
            if track is not None:
                ages.append(track.age)
        return max(ages) if ages else 0
