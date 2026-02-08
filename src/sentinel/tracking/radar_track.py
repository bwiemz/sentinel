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
        _meas_dim = 3 if use_3d or use_doppler else 2
        super().__init__(
            track_id=track_id,
            confirm_hits=confirm_hits,
            max_coast=max_coast,
            confirm_window=confirm_window,
            tentative_delete_misses=tentative_delete_misses,
            confirmed_coast_misses=confirmed_coast_misses,
            coast_reconfirm_hits=coast_reconfirm_hits,
            measurement_dim=_meas_dim,
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
                if self.quality_monitor is not None:
                    innovation = z - self.ekf.predicted_measurement
                    innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
                    innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi
                    self.quality_monitor.record_innovation(innovation, self.ekf.innovation_covariance)
                self.ekf.update(z)
            elif self._use_doppler and detection.velocity_mps is not None:
                z = np.array([detection.range_m, az_rad, detection.velocity_mps])
                if self.quality_monitor is not None:
                    innovation = z - self.ekf.predicted_measurement
                    innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
                    self.quality_monitor.record_innovation(innovation, self.ekf.innovation_covariance)
                self.ekf.update(z)
            elif not self._use_doppler:
                z = np.array([detection.range_m, az_rad])
                if self.quality_monitor is not None:
                    innovation = z - self.ekf.predicted_measurement
                    innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
                    self.quality_monitor.record_innovation(innovation, self.ekf.innovation_covariance)
                self.ekf.update(z)
            # If use_doppler but no velocity_mps, skip EKF update (predict-only)

        self.last_detection = detection
        self.last_update_time = detection.timestamp
        self._record_hit()

    def predict_to_time(self, target_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Non-mutating forward prediction to a target time.

        Propagates the current state/covariance to target_time without
        modifying the track's internal EKF state.

        Args:
            target_time: Target epoch in seconds.

        Returns:
            (x_pred, P_pred) — predicted state vector and covariance.
        """
        dt = target_time - self.last_update_time
        if dt <= 0:
            return self.ekf.x.copy(), self.ekf.P.copy()

        n = self.ekf.dim_state
        F = np.eye(n)
        if n == 6:
            # CA / 3D state layout: [x, vx, ax, y, vy, ay] or [x, vx, y, vy, z, vz]
            if self._use_3d:
                # 3D CV: [x, vx, y, vy, z, vz]
                F[0, 1] = dt
                F[2, 3] = dt
                F[4, 5] = dt
            else:
                # CA: [x, vx, ax, y, vy, ay]
                F[0, 1] = dt
                F[0, 2] = 0.5 * dt * dt
                F[1, 2] = dt
                F[3, 4] = dt
                F[3, 5] = 0.5 * dt * dt
                F[4, 5] = dt
        else:
            # CV: [x, vx, y, vy]
            F[0, 1] = dt
            F[2, 3] = dt

        x_pred = F @ self.ekf.x
        P_pred = F @ self.ekf.P @ F.T + self.ekf.Q
        return x_pred, P_pred

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
