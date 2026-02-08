"""Radar track manager -- mirrors TrackManager but uses RadarTrack + EKF."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.radar_association import RadarAssociator
from sentinel.tracking.radar_track import RadarTrack

logger = logging.getLogger(__name__)


class RadarTrackManager:
    """Manages radar track lifecycle using EKF and polar association.

    Args:
        config: DictConfig with filter, association, track_management sections.
    """

    def __init__(self, config: DictConfig):
        self._tracks: dict[str, RadarTrack] = {}
        self._dt = config.filter.get("dt", 0.1)
        self._confirm_hits = config.track_management.get("confirm_hits", 3)
        self._max_coast = config.track_management.get("max_coast_frames", 5)
        self._max_tracks = config.track_management.get("max_tracks", 50)

        self._associator = RadarAssociator(
            gate_threshold=config.association.get("gate_threshold", 9.21),
        )

    def step(self, detections: list[Detection]) -> list[RadarTrack]:
        """Process one radar scan of detections. Returns active tracks.

        1. Predict all existing tracks
        2. Associate detections to tracks (Hungarian)
        3. Update matched tracks
        4. Mark unmatched as missed
        5. Initiate new tracks from unmatched detections
        6. Prune dead tracks
        """
        # 1. Predict
        for track in self._tracks.values():
            if track.is_alive:
                track.predict()

        active = [t for t in self._tracks.values() if t.is_alive]

        # 2. Associate
        result = self._associator.associate(active, detections)

        # 3. Update matched
        for track_idx, det_idx in result.matched_pairs:
            active[track_idx].update(detections[det_idx])

        # 4. Mark unmatched as missed
        for track_idx in result.unmatched_tracks:
            active[track_idx].mark_missed()

        # 5. Initiate new tracks
        for det_idx in result.unmatched_detections:
            det = detections[det_idx]
            if det.range_m is not None and len(self._tracks) < self._max_tracks:
                self._initiate_track(det)

        # 6. Prune
        self._prune_tracks()

        return self.active_tracks

    def _initiate_track(self, detection: Detection) -> RadarTrack:
        track = RadarTrack(
            detection=detection,
            dt=self._dt,
            confirm_hits=self._confirm_hits,
            max_coast=self._max_coast,
        )
        self._tracks[track.track_id] = track
        logger.debug("Radar track initiated: %s", track.track_id)
        return track

    def _prune_tracks(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.state == TrackState.DELETED]
        for tid in dead:
            logger.debug("Radar track deleted: %s", tid)
            del self._tracks[tid]

    @property
    def active_tracks(self) -> list[RadarTrack]:
        return [t for t in self._tracks.values() if t.is_alive]

    @property
    def confirmed_tracks(self) -> list[RadarTrack]:
        return [t for t in self._tracks.values() if t.state == TrackState.CONFIRMED]

    @property
    def track_count(self) -> int:
        return len(self.active_tracks)
