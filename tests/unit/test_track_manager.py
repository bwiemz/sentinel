"""Tests for TrackManager."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.track_manager import TrackManager


def _make_config():
    return OmegaConf.create({
        "filter": {"dt": 1 / 30, "type": "kf"},
        "association": {"gate_threshold": 50.0, "method": "hungarian"},
        "track_management": {
            "confirm_hits": 3,
            "confirm_window": 5,
            "max_coast_frames": 10,
            "max_tracks": 50,
        },
    })


def _det(cx, cy, w=100, h=100, cls="person", conf=0.9):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=conf,
    )


class TestTrackManager:
    def test_initiate_track(self):
        mgr = TrackManager(_make_config())
        tracks = mgr.step([_det(100, 100)])
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TENTATIVE

    def test_no_detections(self):
        mgr = TrackManager(_make_config())
        tracks = mgr.step([])
        assert len(tracks) == 0

    def test_track_association(self):
        mgr = TrackManager(_make_config())
        # Frame 1: initiate
        mgr.step([_det(100, 100)])
        # Frame 2: nearby detection should associate to same track
        tracks = mgr.step([_det(105, 102)])
        assert len(tracks) == 1
        assert tracks[0].hits == 2

    def test_track_confirmation(self):
        mgr = TrackManager(_make_config())
        for i in range(4):
            tracks = mgr.step([_det(100 + i, 100)])
        confirmed = mgr.confirmed_tracks
        assert len(confirmed) == 1

    def test_track_deletion_on_miss(self):
        mgr = TrackManager(_make_config())
        mgr.step([_det(100, 100)])
        # Miss enough frames for tentative deletion
        for _ in range(5):
            mgr.step([])
        assert mgr.track_count == 0

    def test_multiple_tracks(self):
        mgr = TrackManager(_make_config())
        tracks = mgr.step([_det(100, 100), _det(500, 500)])
        assert len(tracks) == 2

    def test_tracks_stay_separate(self):
        mgr = TrackManager(_make_config())
        mgr.step([_det(100, 100), _det(500, 500)])
        mgr.step([_det(102, 102), _det(502, 502)])
        tracks = mgr.active_tracks
        assert len(tracks) == 2
        # Check they're tracking different locations
        positions = sorted([tuple(t.position) for t in tracks])
        assert positions[0][0] < 200
        assert positions[1][0] > 400

    def test_max_tracks_limit(self):
        cfg = _make_config()
        cfg.track_management.max_tracks = 3
        mgr = TrackManager(cfg)
        dets = [_det(i * 200, 100) for i in range(10)]
        tracks = mgr.step(dets)
        assert len(tracks) <= 3

    def test_track_count_property(self):
        mgr = TrackManager(_make_config())
        assert mgr.track_count == 0
        mgr.step([_det(100, 100)])
        assert mgr.track_count == 1
