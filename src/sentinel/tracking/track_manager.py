"""Multi-target track manager with Hungarian (optimal) association.

Uses scipy.optimize.linear_sum_assignment for globally optimal
detection-to-track matching based on combined Mahalanobis + IoU cost.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.association import HungarianAssociator
from sentinel.tracking.jpda import JPDAAssociator
from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)


class TrackManager:
    """Manages track lifecycle: initiation, update, coasting, deletion.

    Uses the Hungarian algorithm for globally optimal data association.
    """

    def __init__(self, config: DictConfig):
        self._tracks: dict[str, Track] = {}
        self._dt = config.filter.get("dt", 1 / 30)
        self._confirm_hits = config.track_management.get("confirm_hits", 3)
        self._max_coast = config.track_management.get("max_coast_frames", 15)
        self._max_tracks = config.track_management.get("max_tracks", 100)

        # Lifecycle thresholds (configurable)
        self._confirm_window = config.track_management.get("confirm_window", None)
        self._tent_delete = config.track_management.get("tentative_delete_misses", 3)
        self._conf_coast = config.track_management.get("confirmed_coast_misses", 5)
        self._coast_reconfirm = config.track_management.get("coast_reconfirm_hits", 2)

        # Filter noise parameters
        pn = config.filter.get("process_noise_std", None)
        self._process_noise_std = float(pn) if pn is not None else None
        mn = config.filter.get("measurement_noise_std", None)
        self._measurement_noise_std = float(mn) if mn is not None else None
        self._filter_type = config.filter.get("type", "kf")

        self._associator = HungarianAssociator(
            gate_threshold=config.association.get("gate_threshold", 9.21),
            iou_weight=config.association.get("iou_weight", 0.5),
            mahalanobis_weight=config.association.get("mahalanobis_weight", 0.5),
            cascaded=config.association.get("cascaded", False),
        )

        # Optional JPDA associator (replaces Hungarian when enabled)
        self._jpda: JPDAAssociator | None = None
        if config.association.get("method", "hungarian") == "jpda":
            self._jpda = JPDAAssociator(
                gate_threshold=config.association.get("gate_threshold", 9.21),
                P_D=config.association.get("detection_probability", 0.9),
                false_alarm_density=config.association.get("false_alarm_density", 1e-6),
            )

    def step(self, detections: list[Detection]) -> list[Track]:
        """Process one frame of detections. Returns active tracks.

        1. Predict all existing tracks
        2. Associate detections to tracks (Hungarian algorithm)
        3. Update matched tracks
        4. Mark unmatched tracks as missed
        5. Initiate new tracks from unmatched detections
        6. Prune dead tracks
        """
        # 1. Predict
        for track in self._tracks.values():
            if track.is_alive:
                track.predict()

        active = [t for t in self._tracks.values() if t.is_alive]

        if self._jpda is not None:
            # JPDA handles association + update in one step
            result = self._jpda.associate_and_update(active, detections)
            # Mark tracks that had no gated detections
            for track_idx in result.unmatched_tracks:
                active[track_idx].mark_missed()
        else:
            # Hungarian association + separate update
            result = self._associator.associate(active, detections)
            for track_idx, det_idx in result.matched_pairs:
                active[track_idx].update(detections[det_idx])
            for track_idx in result.unmatched_tracks:
                active[track_idx].mark_missed()

        # 5. Initiate new tracks from unmatched detections
        for det_idx in result.unmatched_detections:
            det = detections[det_idx]
            if det.bbox is not None and len(self._tracks) < self._max_tracks:
                self._initiate_track(det)

        # 6. Prune
        self._prune_tracks()

        return self.active_tracks

    def _initiate_track(self, detection: Detection) -> Track:
        """Create a new track from an unmatched detection."""
        track = Track(
            detection=detection,
            dt=self._dt,
            confirm_hits=self._confirm_hits,
            max_coast=self._max_coast,
            confirm_window=self._confirm_window,
            tentative_delete_misses=self._tent_delete,
            confirmed_coast_misses=self._conf_coast,
            coast_reconfirm_hits=self._coast_reconfirm,
            process_noise_std=self._process_noise_std,
            measurement_noise_std=self._measurement_noise_std,
            filter_type=self._filter_type,
        )
        self._tracks[track.track_id] = track
        logger.debug("Track initiated: %s (%s)", track.track_id, detection.class_name)
        return track

    def _prune_tracks(self) -> None:
        """Remove tracks that have been marked DELETED."""
        dead = [tid for tid, t in self._tracks.items() if t.state == TrackState.DELETED]
        for tid in dead:
            logger.debug("Track deleted: %s", tid)
            del self._tracks[tid]

    @property
    def active_tracks(self) -> list[Track]:
        """All non-deleted tracks."""
        return [t for t in self._tracks.values() if t.is_alive]

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Only confirmed tracks."""
        return [t for t in self._tracks.values() if t.state == TrackState.CONFIRMED]

    @property
    def track_count(self) -> int:
        return len(self.active_tracks)
