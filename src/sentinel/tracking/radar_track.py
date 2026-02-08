"""Radar-specific track using Extended Kalman Filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from sentinel.core.types import Detection, TrackState, generate_track_id
from sentinel.tracking.filters import ExtendedKalmanFilter
from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    polar_to_cartesian,
)


class RadarTrack:
    """A single radar target track with EKF state estimation.

    Lifecycle: TENTATIVE -> CONFIRMED -> COASTING -> DELETED

    State space: Cartesian [x, vx, y, vy] in meters.
    Measurement: Polar [range_m, azimuth_rad] from radar.

    Args:
        detection: Initial radar detection (must have range_m, azimuth_deg).
        track_id: Unique identifier (auto-generated if None).
        dt: Time step (1/scan_rate_hz).
        confirm_hits: Hits needed for confirmation.
        max_coast: Max scans to coast before deletion.
    """

    def __init__(
        self,
        detection: Detection,
        track_id: Optional[str] = None,
        dt: float = 0.1,
        confirm_hits: int = 3,
        max_coast: int = 5,
    ):
        self.track_id = track_id or generate_track_id()
        self.state = TrackState.TENTATIVE
        self.ekf = ExtendedKalmanFilter(dim_state=4, dim_meas=2, dt=dt)

        # Initialize EKF state from polar detection
        if detection.range_m is not None and detection.azimuth_deg is not None:
            az_rad = azimuth_deg_to_rad(detection.azimuth_deg)
            pos = polar_to_cartesian(detection.range_m, az_rad)
            self.ekf.x[0] = pos[0]  # x
            self.ekf.x[2] = pos[1]  # y
            # Velocity initialized to zero
            self.ekf.P[0, 0] = 100.0
            self.ekf.P[2, 2] = 100.0

        # Lifecycle counters
        self.hits = 1
        self.consecutive_hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.age = 0
        self.score = 0.0

        # Configuration
        self._confirm_hits = confirm_hits
        self._max_coast = max_coast

        # Last detection
        self.last_detection: Optional[Detection] = detection

        self._update_score()

    def predict(self) -> np.ndarray:
        """Predict next state. Call once per scan."""
        self.age += 1
        return self.ekf.predict()

    def update(self, detection: Detection) -> None:
        """Update track with a radar detection."""
        if detection.range_m is not None and detection.azimuth_deg is not None:
            z = np.array([
                detection.range_m,
                azimuth_deg_to_rad(detection.azimuth_deg),
            ])
            self.ekf.update(z)

        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        self.last_detection = detection

        self._update_state()
        self._update_score()

    def mark_missed(self) -> None:
        """Mark this track as having no associated detection this scan."""
        self.misses += 1
        self.consecutive_misses += 1
        self.consecutive_hits = 0
        self._update_state()
        self._update_score()

    def _update_state(self) -> None:
        if self.state == TrackState.TENTATIVE:
            if self.consecutive_hits >= self._confirm_hits:
                self.state = TrackState.CONFIRMED
            elif self.consecutive_misses >= 3:
                self.state = TrackState.DELETED
        elif self.state == TrackState.CONFIRMED:
            if self.consecutive_misses >= 5:
                self.state = TrackState.COASTING
        elif self.state == TrackState.COASTING:
            if self.consecutive_hits >= 2:
                self.state = TrackState.CONFIRMED
            elif self.consecutive_misses >= self._max_coast:
                self.state = TrackState.DELETED

    def _update_score(self) -> None:
        if self.age == 0:
            self.score = 0.5
            return
        hit_ratio = self.hits / max(self.age, 1)
        recency = max(0, 1.0 - self.consecutive_misses * 0.15)
        confirmation = 1.0 if self.state == TrackState.CONFIRMED else 0.5
        self.score = min(1.0, hit_ratio * 0.4 + recency * 0.3 + confirmation * 0.3)

    @property
    def position(self) -> np.ndarray:
        """Position [x, y] in meters."""
        return self.ekf.position

    @property
    def velocity(self) -> np.ndarray:
        """Velocity [vx, vy] in m/s."""
        return self.ekf.velocity

    @property
    def range_m(self) -> float:
        """Current estimated range to target in meters."""
        pos = self.position
        return float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))

    @property
    def azimuth_deg(self) -> float:
        """Current estimated azimuth to target in degrees."""
        _, az = cartesian_to_polar(self.position[0], self.position[1])
        return azimuth_rad_to_deg(az)

    @property
    def is_alive(self) -> bool:
        return self.state != TrackState.DELETED

    @property
    def predicted_bbox(self) -> Optional[np.ndarray]:
        """Not applicable for radar tracks."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "position_m": self.position.tolist(),
            "velocity_mps": self.velocity.tolist(),
            "range_m": round(self.range_m, 1),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "score": round(self.score, 3),
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
        }
