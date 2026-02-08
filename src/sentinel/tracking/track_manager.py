"""Multi-target track manager with Hungarian (optimal) association.

Uses scipy.optimize.linear_sum_assignment for globally optimal
detection-to-track matching based on combined Mahalanobis + IoU cost.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.association import HungarianAssociator
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

        self._associator = HungarianAssociator(
            gate_threshold=config.association.get("gate_threshold", 9.21),
            iou_weight=config.association.get("iou_weight", 0.5),
            mahalanobis_weight=config.association.get("mahalanobis_weight", 0.5),
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

        # 2. Associate (Hungarian algorithm for optimal assignment)
        result = self._associator.associate(active, detections)

        # 3. Update matched tracks
        for track_idx, det_idx in result.matched_pairs:
            active[track_idx].update(detections[det_idx])

        # 4. Mark unmatched tracks as missed
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
