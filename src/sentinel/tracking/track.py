"""Track class with state machine lifecycle management."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from sentinel.core.types import Detection, TrackState, generate_track_id
from sentinel.tracking.base_track import TrackBase
from sentinel.tracking.filters import ConstantAccelerationKF, KalmanFilter
from sentinel.tracking.imm import IMMFilter


class Track(TrackBase):
    """A single target track with Kalman filter state estimation.

    Lifecycle:  TENTATIVE -> CONFIRMED -> COASTING -> DELETED

    Args:
        detection: Initial detection that spawned this track.
        track_id: Unique track identifier (auto-generated if None).
        dt: Time step for the Kalman filter.
        confirm_hits: Consecutive hits needed to confirm (M in M/N logic).
        max_coast: Maximum frames to coast without updates before deletion.
        confirm_window: Sliding window for M-of-N confirmation.
        tentative_delete_misses: Consecutive misses to delete tentative track.
        confirmed_coast_misses: Consecutive misses to start coasting.
        coast_reconfirm_hits: Consecutive hits to re-confirm from coasting.
        process_noise_std: Process noise sigma_a for Kalman filter.
        measurement_noise_std: Measurement noise std for Kalman filter.
    """

    def __init__(
        self,
        detection: Detection,
        track_id: Optional[str] = None,
        dt: float = 1 / 30,
        confirm_hits: int = 3,
        max_coast: int = 15,
        confirm_window: Optional[int] = None,
        tentative_delete_misses: int = 3,
        confirmed_coast_misses: int = 5,
        coast_reconfirm_hits: int = 2,
        process_noise_std: Optional[float] = None,
        measurement_noise_std: Optional[float] = None,
        filter_type: str = "kf",
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
        if filter_type == "ca":
            self.kf = ConstantAccelerationKF(dim_state=6, dim_meas=2, dt=dt)
        elif filter_type == "imm":
            self.kf = IMMFilter(dt=dt, mode="camera")
        else:
            self.kf = KalmanFilter(dim_state=4, dim_meas=2, dt=dt)

        # Apply configured noise parameters
        if process_noise_std is not None:
            self.kf.set_process_noise_std(process_noise_std)
        if measurement_noise_std is not None:
            self.kf.set_measurement_noise_std(measurement_noise_std)

        # Detection history
        self.last_detection: Optional[Detection] = detection
        self.class_histogram: dict[str, int] = {}

        # Initialize KF from first detection
        if detection.bbox is not None:
            center = detection.bbox_center
            if center is not None:
                self.kf.x[0] = center[0]  # x position
                self.kf.x[2] = center[1]  # y position
                # Velocity initialized to zero (no prior info)

                # Tighter initial covariance on position
                self.kf.P[0, 0] = 50.0
                self.kf.P[2, 2] = 50.0

        # Record class
        if detection.class_name:
            self.class_histogram[detection.class_name] = 1

    def predict(self) -> np.ndarray:
        """Predict next state. Call once per frame before association."""
        self.age += 1
        return self.kf.predict()

    def update(self, detection: Detection) -> None:
        """Update track with an associated detection."""
        if detection.bbox is not None:
            center = detection.bbox_center
            if center is not None:
                self.kf.update(center)

        self.last_detection = detection

        # Update class histogram
        if detection.class_name:
            self.class_histogram[detection.class_name] = (
                self.class_histogram.get(detection.class_name, 0) + 1
            )

        self._record_hit()

    def mark_missed(self) -> None:
        """Mark this track as having no associated detection this frame."""
        self._record_miss()

    # --- Properties ---

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y] in pixels."""
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vx, vy] in pixels/frame."""
        return self.kf.velocity

    @property
    def predicted_bbox(self) -> Optional[np.ndarray]:
        """Estimated bounding box from KF state + last detection size."""
        if self.last_detection is None or self.last_detection.bbox is None:
            return None
        orig = self.last_detection.bbox
        w = orig[2] - orig[0]
        h = orig[3] - orig[1]
        cx, cy = self.kf.position
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def dominant_class(self) -> Optional[str]:
        """Most frequently detected class for this track."""
        if not self.class_histogram:
            return None
        return max(self.class_histogram, key=self.class_histogram.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "score": round(self.score, 3),
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
            "class_name": self.dominant_class,
            "confidence": (
                self.last_detection.confidence if self.last_detection else None
            ),
        }
