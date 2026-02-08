"""Radar-specific track using Extended Kalman Filter."""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.core.types import Detection
from sentinel.tracking.base_track import TrackBase
from sentinel.tracking.filters import (
    ConstantAccelerationEKF,
    ExtendedKalmanFilter,
    ExtendedKalmanFilter3D,
    ExtendedKalmanFilterWithDoppler,
)
from sentinel.tracking.imm import IMMFilter
from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    polar_to_cartesian,
    polar_to_cartesian_3d,
)


class RadarTrack(TrackBase):
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
        confirm_window: Sliding window for M-of-N confirmation.
        tentative_delete_misses: Consecutive misses to delete tentative track.
        confirmed_coast_misses: Consecutive misses to start coasting.
        coast_reconfirm_hits: Consecutive hits to re-confirm from coasting.
    """

    def __init__(
        self,
        detection: Detection,
        track_id: str | None = None,
        dt: float = 0.1,
        confirm_hits: int = 3,
        max_coast: int = 5,
        confirm_window: int | None = None,
        tentative_delete_misses: int = 3,
        confirmed_coast_misses: int = 5,
        coast_reconfirm_hits: int = 2,
        filter_type: str = "ekf",
        use_3d: bool = False,
        use_doppler: bool = False,
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
        self._use_3d = use_3d
        self._use_doppler = use_doppler
        if use_3d:
            self.ekf = ExtendedKalmanFilter3D(dt=dt)
        elif use_doppler:
            self.ekf = ExtendedKalmanFilterWithDoppler(dim_state=4, dt=dt)
        elif filter_type == "ca":
            self.ekf = ConstantAccelerationEKF(dim_state=6, dim_meas=2, dt=dt)
        elif filter_type == "imm":
            self.ekf = IMMFilter(dt=dt, mode="radar")
        else:
            self.ekf = ExtendedKalmanFilter(dim_state=4, dim_meas=2, dt=dt)

        # Initialize EKF state from polar detection
        if detection.range_m is not None and detection.azimuth_deg is not None:
            az_rad = azimuth_deg_to_rad(detection.azimuth_deg)
            if use_3d:
                el_rad = np.radians(detection.elevation_deg or 0.0)
                pos = polar_to_cartesian_3d(detection.range_m, az_rad, el_rad)
                self.ekf.x[0] = pos[0]
                self.ekf.x[2] = pos[1]
                self.ekf.x[4] = pos[2]
                self.ekf.P[0, 0] = 100.0
                self.ekf.P[2, 2] = 100.0
                self.ekf.P[4, 4] = 100.0
            else:
                pos = polar_to_cartesian(detection.range_m, az_rad)
                self.ekf.x[0] = pos[0]
                self.ekf.x[2] = pos[1]
                self.ekf.P[0, 0] = 100.0
                self.ekf.P[2, 2] = 100.0

        # Last detection
        self.last_detection: Detection | None = detection

    def predict(self) -> np.ndarray:
        """Predict next state. Call once per scan."""
        self.age += 1
        return self.ekf.predict()

    def update(self, detection: Detection) -> None:
        """Update track with a radar detection."""
        if detection.range_m is not None and detection.azimuth_deg is not None:
            az_rad = azimuth_deg_to_rad(detection.azimuth_deg)
            if self._use_3d:
                el_rad = np.radians(detection.elevation_deg or 0.0)
                z = np.array([detection.range_m, az_rad, el_rad])
                self.ekf.update(z)
            elif self._use_doppler and detection.velocity_mps is not None:
                z = np.array([detection.range_m, az_rad, detection.velocity_mps])
                self.ekf.update(z)
            elif not self._use_doppler:
                z = np.array([detection.range_m, az_rad])
                self.ekf.update(z)
            # If use_doppler but no velocity_mps, skip EKF update (predict-only)

        self.last_detection = detection
        self._record_hit()

    def mark_missed(self) -> None:
        """Mark this track as having no associated detection this scan."""
        self._record_miss()

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
        return float(np.linalg.norm(pos))

    @property
    def azimuth_deg(self) -> float:
        """Current estimated azimuth to target in degrees."""
        pos = self.position
        _, az = cartesian_to_polar(pos[0], pos[1])
        return azimuth_rad_to_deg(az)

    @property
    def elevation_deg(self) -> float | None:
        """Current estimated elevation (3D mode only)."""
        if not self._use_3d:
            return None
        pos = self.position
        r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        return float(np.degrees(np.arctan2(pos[2], r_xy)))

    @property
    def predicted_bbox(self) -> np.ndarray | None:
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
