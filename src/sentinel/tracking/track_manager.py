"""Single-target track manager for Phase 2.

Phase 3 upgrades this to full multi-target with Hungarian association.
For now, uses simple nearest-neighbor for single/few targets.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.filters import KalmanFilter
from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)


class TrackManager:
    """Manages track lifecycle: initiation, update, coasting, deletion.

    Phase 2: Simple nearest-neighbor association.
    Phase 3: Upgraded to Hungarian algorithm.
    """

    def __init__(self, config: DictConfig):
        self._tracks: dict[str, Track] = {}
        self._dt = config.filter.get("dt", 1 / 30)
        self._confirm_hits = config.track_management.get("confirm_hits", 3)
        self._max_coast = config.track_management.get("max_coast_frames", 15)
        self._max_tracks = config.track_management.get("max_tracks", 100)
        self._gate_threshold = config.association.get("gate_threshold", 9.21)

    def step(self, detections: list[Detection]) -> list[Track]:
        """Process one frame of detections. Returns active tracks.

        1. Predict all existing tracks
        2. Associate detections to tracks (nearest-neighbor)
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

        # 2. Associate (simple nearest-neighbor for Phase 2)
        matched_tracks: set[str] = set()
        matched_dets: set[int] = set()

        for i, det in enumerate(detections):
            if det.bbox is None:
                continue
            center = det.bbox_center
            if center is None:
                continue

            best_track: Optional[Track] = None
            best_dist = float("inf")

            for track in active:
                if track.track_id in matched_tracks:
                    continue
                dist = track.kf.gating_distance(center)
                if dist < self._gate_threshold and dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is not None:
                # 3. Update matched track
                best_track.update(det)
                matched_tracks.add(best_track.track_id)
                matched_dets.add(i)

        # 4. Mark unmatched tracks as missed
        for track in active:
            if track.track_id not in matched_tracks:
                track.mark_missed()

        # 5. Initiate new tracks from unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets and det.bbox is not None:
                if len(self._tracks) < self._max_tracks:
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
