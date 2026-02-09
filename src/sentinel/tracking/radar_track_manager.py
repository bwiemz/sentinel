"""Radar track manager -- mirrors TrackManager but uses RadarTrack + EKF."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.jpda import RadarJPDAAssociator
from sentinel.tracking.radar_association import RadarAssociator
from sentinel.tracking.radar_track import RadarTrack

logger = logging.getLogger(__name__)


class RadarTrackManager:
    """Manages radar track lifecycle using EKF and polar association.

    Args:
        config: DictConfig with filter, association, track_management sections.
    """

    def __init__(self, config: DictConfig, geo_context=None):
        self._tracks: dict[str, RadarTrack] = {}
        self._geo_context = geo_context
        self._dt = config.filter.get("dt", 0.1)
        self._confirm_hits = config.track_management.get("confirm_hits", 3)
        self._max_coast = config.track_management.get("max_coast_frames", 5)
        self._max_tracks = config.track_management.get("max_tracks", 50)

        # Lifecycle thresholds (configurable)
        self._confirm_window = config.track_management.get("confirm_window", None)
        self._tent_delete = config.track_management.get("tentative_delete_misses", 3)
        self._conf_coast = config.track_management.get("confirmed_coast_misses", 5)
        self._coast_reconfirm = config.track_management.get("coast_reconfirm_hits", 2)
        self._filter_type = config.filter.get("type", "ekf")
        self._use_doppler = config.filter.get("use_doppler", False)

        vel_gate = config.association.get("velocity_gate_mps", None)
        self._associator = RadarAssociator(
            gate_threshold=config.association.get("gate_threshold", 9.21),
            velocity_gate_mps=float(vel_gate) if vel_gate is not None else None,
            cascaded=config.association.get("cascaded", False),
        )

        # Optional JPDA associator
        self._jpda: RadarJPDAAssociator | None = None
        if config.association.get("method", "hungarian") == "jpda":
            self._jpda = RadarJPDAAssociator(
                gate_threshold=config.association.get("gate_threshold", 9.21),
                P_D=config.association.get("detection_probability", 0.9),
                false_alarm_density=config.association.get("false_alarm_density", 1e-6),
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

        if self._jpda is not None:
            result = self._jpda.associate_and_update(active, detections)
            for track_idx in result.unmatched_tracks:
                active[track_idx].mark_missed()
        else:
            result = self._associator.associate(active, detections)
            for track_idx, det_idx in result.matched_pairs:
                active[track_idx].update(detections[det_idx])
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
            confirm_window=self._confirm_window,
            tentative_delete_misses=self._tent_delete,
            confirmed_coast_misses=self._conf_coast,
            coast_reconfirm_hits=self._coast_reconfirm,
            filter_type=self._filter_type,
            use_doppler=self._use_doppler,
            geo_context=self._geo_context,
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
