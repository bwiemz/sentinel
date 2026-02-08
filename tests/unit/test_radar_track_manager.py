"""Tests for RadarTrackManager."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.radar_track_manager import RadarTrackManager


def _make_config():
    return OmegaConf.create({
        "filter": {"dt": 0.1, "type": "ekf"},
        "association": {"gate_threshold": 50.0},
        "track_management": {
            "confirm_hits": 3,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    })


def _rdet(range_m=3000.0, azimuth_deg=0.0, velocity_mps=0.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=velocity_mps,
        rcs_dbsm=15.0,
    )


class TestRadarTrackManager:
    def test_initiate_track(self):
        mgr = RadarTrackManager(_make_config())
        tracks = mgr.step([_rdet(3000.0, 0.0)])
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TENTATIVE

    def test_no_detections(self):
        mgr = RadarTrackManager(_make_config())
        tracks = mgr.step([])
        assert len(tracks) == 0

    def test_track_association(self):
        mgr = RadarTrackManager(_make_config())
        mgr.step([_rdet(3000.0, 0.0)])
        tracks = mgr.step([_rdet(3005.0, 0.1)])
        assert len(tracks) == 1
        assert tracks[0].hits == 2

    def test_track_confirmation(self):
        mgr = RadarTrackManager(_make_config())
        for i in range(4):
            mgr.step([_rdet(3000.0 + i, 0.0)])
        confirmed = mgr.confirmed_tracks
        assert len(confirmed) == 1

    def test_track_deletion_on_miss(self):
        mgr = RadarTrackManager(_make_config())
        mgr.step([_rdet(3000.0, 0.0)])
        for _ in range(5):
            mgr.step([])
        assert mgr.track_count == 0

    def test_multiple_tracks(self):
        mgr = RadarTrackManager(_make_config())
        tracks = mgr.step([_rdet(3000.0, 0.0), _rdet(5000.0, 30.0)])
        assert len(tracks) == 2

    def test_tracks_stay_separate(self):
        mgr = RadarTrackManager(_make_config())
        mgr.step([_rdet(3000.0, 0.0), _rdet(5000.0, 30.0)])
        mgr.step([_rdet(3002.0, 0.05), _rdet(5003.0, 30.05)])
        tracks = mgr.active_tracks
        assert len(tracks) == 2
        ranges = sorted([t.range_m for t in tracks])
        assert ranges[0] < 4000
        assert ranges[1] > 4000

    def test_max_tracks_limit(self):
        cfg = _make_config()
        cfg.track_management.max_tracks = 3
        mgr = RadarTrackManager(cfg)
        dets = [_rdet(1000.0 * (i + 1), i * 10.0) for i in range(10)]
        tracks = mgr.step(dets)
        assert len(tracks) <= 3

    def test_track_count_property(self):
        mgr = RadarTrackManager(_make_config())
        assert mgr.track_count == 0
        mgr.step([_rdet()])
        assert mgr.track_count == 1
