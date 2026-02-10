"""Thermal track using bearing-only Extended Kalman Filter."""

from __future__ import annotations

import numpy as np

from sentinel.core.types import Detection
from sentinel.tracking.base_track import TrackBase
from sentinel.tracking.filters import BearingOnlyEKF
from sentinel.utils.coords import azimuth_deg_to_rad, azimuth_rad_to_deg, polar_to_cartesian
from sentinel.utils.geo_context import GeoContext


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
        geo_context: GeoContext | None = None,
    ):
        super().__init__(
            track_id=track_id,
            confirm_hits=confirm_hits,
            max_coast=max_coast,
            confirm_window=confirm_window,
            tentative_delete_misses=tentative_delete_misses,
            confirmed_coast_misses=confirmed_coast_misses,
            coast_reconfirm_hits=coast_reconfirm_hits,
            measurement_dim=1,
            geo_context=geo_context,
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

        # Record NIS before update for filter consistency monitoring
        if self.quality_monitor is not None:
            innovation = z - self.ekf.predicted_measurement
            innovation[0] = (innovation[0] + np.pi) % (2 * np.pi) - np.pi
            self.quality_monitor.record_innovation(innovation, self.ekf.innovation_covariance)

        self.ekf.update(z)

        self.last_detection = detection
        self.last_update_time = detection.timestamp
        self._last_temperature_k = detection.temperature_k
        self._record_hit()

    def predict_to_time(self, target_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Non-mutating forward prediction to a target time.

        Propagates the current state/covariance to target_time without
        modifying the track's internal EKF state.

        Args:
            target_time: Target epoch in seconds.

        Returns:
            (x_pred, P_pred) — predicted 4D state and covariance.
        """
        dt = target_time - self.last_update_time
        if dt <= 0:
            return self.ekf.x.copy(), self.ekf.P.copy()

        n = self.ekf.dim_state
        F = np.eye(n)
        # BearingOnlyEKF: state [x, vx, y, vy]
        F[0, 1] = dt
        F[2, 3] = dt

        x_pred = F @ self.ekf.x
        # Scale process noise for the prediction interval
        dt_nominal = getattr(self.ekf, 'dt', dt)
        Q_scale = dt / dt_nominal if dt_nominal > 0 else 1.0
        P_pred = F @ self.ekf.P @ F.T + self.ekf.Q * Q_scale
        return x_pred, P_pred

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
    def position_geo(self) -> tuple[float, float, float] | None:
        """Geodetic position (lat, lon, alt) or None if no geo_context."""
        if self._geo_context is None:
            return None
        pos = self.position
        return self._geo_context.xy_to_geodetic(pos[0], pos[1])

    @property
    def predicted_bbox(self) -> None:
        """Thermal tracks have no bounding box."""
        return None

    def to_dict(self) -> dict:
        d = {
            "track_id": self.track_id,
            "state": self.state.value,
            "position": self.position.tolist(),
            "azimuth_deg": self.azimuth_deg,
            "temperature_k": self.temperature_k,
            "hits": self.hits,
            "age": self.age,
            "score": self.score,
        }
        if self._geo_context is not None:
            geo = self.position_geo
            if geo is not None:
                d["position_geo"] = {
                    "lat": round(geo[0], 7),
                    "lon": round(geo[1], 7),
                    "alt": round(geo[2], 2),
                }
        return d
