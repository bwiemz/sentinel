"""Track class with state machine lifecycle management."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from sentinel.core.types import Detection, TrackState, generate_track_id
from sentinel.tracking.filters import KalmanFilter


class Track:
    """A single target track with Kalman filter state estimation.

    Lifecycle:  TENTATIVE -> CONFIRMED -> COASTING -> DELETED

    Args:
        detection: Initial detection that spawned this track.
        track_id: Unique track identifier (auto-generated if None).
        dt: Time step for the Kalman filter.
        confirm_hits: Consecutive hits needed to confirm (M in M/N logic).
        max_coast: Maximum frames to coast without updates before deletion.
    """

    def __init__(
        self,
        detection: Detection,
        track_id: Optional[str] = None,
        dt: float = 1 / 30,
        confirm_hits: int = 3,
        max_coast: int = 15,
    ):
        self.track_id = track_id or generate_track_id()
        self.state = TrackState.TENTATIVE
        self.kf = KalmanFilter(dim_state=4, dim_meas=2, dt=dt)

        # Track statistics
        self.hits = 1
        self.consecutive_hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.age = 0
        self.score = 0.0

        # Configuration
        self._confirm_hits = confirm_hits
        self._max_coast = max_coast

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

        self._update_score()

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

        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        self.last_detection = detection

        # Update class histogram
        if detection.class_name:
            self.class_histogram[detection.class_name] = (
                self.class_histogram.get(detection.class_name, 0) + 1
            )

        self._update_state()
        self._update_score()

    def mark_missed(self) -> None:
        """Mark this track as having no associated detection this frame."""
        self.misses += 1
        self.consecutive_misses += 1
        self.consecutive_hits = 0
        self._update_state()
        self._update_score()

    # --- State machine ---

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
                self.state = TrackState.CONFIRMED  # Re-acquire
            elif self.consecutive_misses >= self._max_coast:
                self.state = TrackState.DELETED

    def _update_score(self) -> None:
        """Compute track quality score in [0, 1]."""
        # Score components:
        # - Hit ratio (total hits / total age)
        # - Recency (penalize consecutive misses)
        # - Confirmation bonus
        if self.age == 0:
            self.score = 0.5
            return

        hit_ratio = self.hits / max(self.age, 1)
        recency = max(0, 1.0 - self.consecutive_misses * 0.15)
        confirmation = 1.0 if self.state == TrackState.CONFIRMED else 0.5

        self.score = min(1.0, hit_ratio * 0.4 + recency * 0.3 + confirmation * 0.3)

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

    @property
    def is_alive(self) -> bool:
        return self.state != TrackState.DELETED

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
