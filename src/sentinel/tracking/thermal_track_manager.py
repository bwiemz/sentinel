"""Thermal track lifecycle manager."""

from __future__ import annotations

from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.jpda import ThermalJPDAAssociator
from sentinel.tracking.thermal_association import ThermalAssociator
from sentinel.tracking.thermal_track import ThermalTrack


class ThermalTrackManager:
    """Manages thermal track creation, association, and lifecycle.

    Same pattern as RadarTrackManager but for bearing-only thermal detections.
    """

    def __init__(self, config: DictConfig, geo_context=None):
        self._geo_context = geo_context
        self._confirm_hits = config.get("track_management", {}).get("confirm_hits", 3)
        self._max_coast = config.get("track_management", {}).get("max_coast_frames", 10)
        self._max_tracks = config.get("track_management", {}).get("max_tracks", 50)
        self._dt = config.get("filter", {}).get("dt", 0.033)
        self._assumed_range = config.get("filter", {}).get("assumed_initial_range_m", 10000.0)
        gate = config.get("association", {}).get("gate_threshold", 6.635)
        self._associator = ThermalAssociator(gate_threshold=gate)
        self._tracks: dict[str, ThermalTrack] = {}

        # Optional JPDA associator
        assoc_cfg = config.get("association", {})
        self._jpda: ThermalJPDAAssociator | None = None
        if assoc_cfg.get("method", "hungarian") == "jpda":
            self._jpda = ThermalJPDAAssociator(
                gate_threshold=gate,
                P_D=assoc_cfg.get("detection_probability", 0.9),
                false_alarm_density=assoc_cfg.get("false_alarm_density", 1e-6),
            )

        # Lifecycle thresholds (configurable)
        tm = config.get("track_management", {})
        self._confirm_window = tm.get("confirm_window", None)
        self._tent_delete = tm.get("tentative_delete_misses", 3)
        self._conf_coast = tm.get("confirmed_coast_misses", 5)
        self._coast_reconfirm = tm.get("coast_reconfirm_hits", 2)

    def step(self, detections: list[Detection]) -> list[ThermalTrack]:
        """Process one frame of thermal detections."""
        active = [t for t in self._tracks.values() if t.is_alive]

        # 1. Predict
        for track in active:
            track.predict()

        # 2. Associate (+ update for JPDA)
        if self._jpda is not None:
            result = self._jpda.associate_and_update(active, detections)
            for ti in result.unmatched_tracks:
                active[ti].mark_missed()
        else:
            result = self._associator.associate(active, detections)
            for ti, di in result.matched_pairs:
                active[ti].update(detections[di])
            for ti in result.unmatched_tracks:
                active[ti].mark_missed()

        # 5. Initiate new tracks
        for di in result.unmatched_detections:
            if len(self._tracks) >= self._max_tracks:
                break
            track = ThermalTrack(
                detection=detections[di],
                assumed_range_m=self._assumed_range,
                dt=self._dt,
                confirm_hits=self._confirm_hits,
                max_coast=self._max_coast,
                confirm_window=self._confirm_window,
                tentative_delete_misses=self._tent_delete,
                confirmed_coast_misses=self._conf_coast,
                coast_reconfirm_hits=self._coast_reconfirm,
                geo_context=self._geo_context,
            )
            self._tracks[track.track_id] = track

        # 6. Prune dead tracks
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.is_alive}

        return list(self._tracks.values())

    @property
    def active_tracks(self) -> list[ThermalTrack]:
        return [t for t in self._tracks.values() if t.is_alive]

    @property
    def confirmed_tracks(self) -> list[ThermalTrack]:
        return [t for t in self._tracks.values() if t.state == TrackState.CONFIRMED]

    @property
    def track_count(self) -> int:
        return len(self._tracks)
