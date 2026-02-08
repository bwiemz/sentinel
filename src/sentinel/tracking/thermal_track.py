"""Thermal track using bearing-only Extended Kalman Filter."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sentinel.core.types import Detection, TrackState, generate_track_id
from sentinel.tracking.filters import BearingOnlyEKF
from sentinel.utils.coords import azimuth_deg_to_rad, azimuth_rad_to_deg, polar_to_cartesian


class ThermalTrack:
    """Track from thermal sensor using bearing-only EKF.

    Cannot estimate range from thermal alone -- range is initialized
    to an assumed value and refined through multi-sensor fusion.
    """

    def __init__(
        self,
        detection: Detection,
        assumed_range_m: float = 10000.0,
        track_id: Optional[str] = None,
        dt: float = 0.033,
        confirm_hits: int = 3,
        max_coast: int = 10,
    ):
        self.track_id = track_id or generate_track_id()
        self.state = TrackState.TENTATIVE
        self._confirm_hits = confirm_hits
        self._max_coast = max_coast

        # Initialize EKF
        self.ekf = BearingOnlyEKF(dt=dt)
        az_rad = azimuth_deg_to_rad(detection.azimuth_deg or 0.0)
        pos = polar_to_cartesian(assumed_range_m, az_rad)
        self.ekf.x[0] = pos[0]
        self.ekf.x[2] = pos[1]

        # Track lifecycle counters
        self.hits = 1
        self.consecutive_hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.age = 0

        # Sensor data
        self.last_detection: Optional[Detection] = detection
        self._last_temperature_k: Optional[float] = detection.temperature_k
        self._range_fused = False  # True if range has been updated by radar fusion

    def predict(self) -> np.ndarray:
        self.age += 1
        return self.ekf.predict()

    def update(self, detection: Detection) -> None:
        az_rad = azimuth_deg_to_rad(detection.azimuth_deg or 0.0)
        z = np.array([az_rad])
        self.ekf.update(z)

        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        self.last_detection = detection
        self._last_temperature_k = detection.temperature_k
        self._update_state()

    def mark_missed(self) -> None:
        self.misses += 1
        self.consecutive_misses += 1
        self.consecutive_hits = 0
        self._update_state()

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

    @property
    def is_alive(self) -> bool:
        return self.state != TrackState.DELETED

    @property
    def position(self) -> np.ndarray:
        return self.ekf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.ekf.velocity

    @property
    def azimuth_deg(self) -> float:
        return azimuth_rad_to_deg(np.arctan2(self.position[1], self.position[0]))

    @property
    def temperature_k(self) -> Optional[float]:
        return self._last_temperature_k

    @property
    def range_confidence(self) -> float:
        """How confident we are in the range estimate (0-1).

        Low for bearing-only, higher after radar fusion updates range.
        """
        if self._range_fused:
            return 0.8
        # Bearing-only range confidence improves with age but stays low
        return min(0.3, 0.05 * self.age)

    @property
    def predicted_bbox(self) -> None:
        """Thermal tracks have no bounding box."""
        return None

    @property
    def score(self) -> float:
        hit_ratio = self.hits / max(self.age, 1)
        recency = max(0, 1.0 - self.consecutive_misses * 0.15)
        confirmation = 1.0 if self.state == TrackState.CONFIRMED else 0.5
        return min(1.0, hit_ratio * 0.4 + recency * 0.3 + confirmation * 0.3)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "position": self.position.tolist(),
            "azimuth_deg": self.azimuth_deg,
            "temperature_k": self.temperature_k,
            "hits": self.hits,
            "age": self.age,
            "score": self.score,
        }
