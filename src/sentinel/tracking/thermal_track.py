"""Thermal track using bearing-only Extended Kalman Filter."""

from __future__ import annotations

import numpy as np

from sentinel.core.types import Detection
from sentinel.tracking.base_track import TrackBase
from sentinel.tracking.filters import BearingOnlyEKF
from sentinel.utils.coords import azimuth_deg_to_rad, azimuth_rad_to_deg, polar_to_cartesian


class ThermalTrack(TrackBase):
    """Track from thermal sensor using bearing-only EKF.

    Cannot estimate range from thermal alone -- range is initialized
    to an assumed value and refined through multi-sensor fusion.
    """

    def __init__(
        self,
        detection: Detection,
        assumed_range_m: float = 10000.0,
        track_id: str | None = None,
        dt: float = 0.033,
        confirm_hits: int = 3,
        max_coast: int = 10,
        confirm_window: int | None = None,
        tentative_delete_misses: int = 3,
        confirmed_coast_misses: int = 5,
        coast_reconfirm_hits: int = 2,
    ):
        super().__init__(
            track_id=track_id,
            confirm_hits=confirm_hits,
            max_coast=max_coast,
            confirm_window=confirm_window,
            tentative_delete_misses=tentative_delete_misses,
            confirmed_coast_misses=confirmed_coast_misses,
            coast_reconfirm_hits=coast_reconfirm_hits,
        )

        # Initialize EKF
        self.ekf = BearingOnlyEKF(dt=dt)
        az_rad = azimuth_deg_to_rad(detection.azimuth_deg or 0.0)
        pos = polar_to_cartesian(assumed_range_m, az_rad)
        self.ekf.x[0] = pos[0]
        self.ekf.x[2] = pos[1]

        # Sensor data
        self.last_detection: Detection | None = detection
        self._last_temperature_k: float | None = detection.temperature_k
        self._range_fused = False  # True if range has been updated by radar fusion

    def predict(self) -> np.ndarray:
        self.age += 1
        return self.ekf.predict()

    def update(self, detection: Detection) -> None:
        az_rad = azimuth_deg_to_rad(detection.azimuth_deg or 0.0)
        z = np.array([az_rad])
        self.ekf.update(z)

        self.last_detection = detection
        self._last_temperature_k = detection.temperature_k
        self._record_hit()

    def mark_missed(self) -> None:
        self._record_miss()

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
    def temperature_k(self) -> float | None:
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
