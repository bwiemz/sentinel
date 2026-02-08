"""Tests for RadarTrack."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.radar_track import RadarTrack


def _radar_det(range_m=3000.0, azimuth_deg=0.0, velocity_mps=0.0, rcs_dbsm=15.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=velocity_mps,
        rcs_dbsm=rcs_dbsm,
    )


class TestRadarTrack:
    def test_init(self):
        track = RadarTrack(detection=_radar_det())
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1
        assert track.age == 0
        assert track.is_alive

    def test_init_position_from_polar(self):
        # Target at range=5000, azimuth=0 -> (5000, 0)
        track = RadarTrack(detection=_radar_det(5000.0, 0.0))
        pos = track.position
        assert pos[0] == pytest.approx(5000.0, abs=1.0)
        assert pos[1] == pytest.approx(0.0, abs=1.0)

    def test_init_position_45_degrees(self):
        track = RadarTrack(detection=_radar_det(1000.0, 45.0))
        pos = track.position
        assert pos[0] == pytest.approx(1000 * np.cos(np.radians(45)), abs=1.0)
        assert pos[1] == pytest.approx(1000 * np.sin(np.radians(45)), abs=1.0)

    def test_predict_increments_age(self):
        track = RadarTrack(detection=_radar_det())
        track.predict()
        assert track.age == 1

    def test_update_increments_hits(self):
        track = RadarTrack(detection=_radar_det())
        track.predict()
        track.update(_radar_det(3002.0, 0.1))
        assert track.hits == 2
        assert track.consecutive_hits == 2

    def test_mark_missed(self):
        track = RadarTrack(detection=_radar_det())
        track.predict()
        track.mark_missed()
        assert track.misses == 1
        assert track.consecutive_misses == 1
        assert track.consecutive_hits == 0

    def test_tentative_to_confirmed(self):
        track = RadarTrack(detection=_radar_det(), confirm_hits=3)
        for _ in range(3):
            track.predict()
            track.update(_radar_det(3000.0, 0.0))
        assert track.state == TrackState.CONFIRMED

    def test_tentative_to_deleted(self):
        track = RadarTrack(detection=_radar_det())
        for _ in range(3):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.DELETED

    def test_confirmed_to_coasting(self):
        track = RadarTrack(detection=_radar_det(), confirm_hits=2)
        for _ in range(2):
            track.predict()
            track.update(_radar_det())
        assert track.state == TrackState.CONFIRMED

        for _ in range(5):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.COASTING

    def test_coasting_to_deleted(self):
        track = RadarTrack(detection=_radar_det(), confirm_hits=2, max_coast=3)
        for _ in range(2):
            track.predict()
            track.update(_radar_det())

        for _ in range(5):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.COASTING

        for _ in range(3):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.DELETED

    def test_range_property(self):
        track = RadarTrack(detection=_radar_det(5000.0, 0.0))
        assert track.range_m == pytest.approx(5000.0, abs=1.0)

    def test_azimuth_property(self):
        track = RadarTrack(detection=_radar_det(5000.0, 30.0))
        assert track.azimuth_deg == pytest.approx(30.0, abs=1.0)

    def test_predicted_bbox_is_none(self):
        track = RadarTrack(detection=_radar_det())
        assert track.predicted_bbox is None

    def test_to_dict(self):
        track = RadarTrack(detection=_radar_det())
        d = track.to_dict()
        assert "track_id" in d
        assert "state" in d
        assert "position_m" in d
        assert "range_m" in d
        assert "azimuth_deg" in d
        assert d["state"] == "tentative"

    def test_score_range(self):
        track = RadarTrack(detection=_radar_det())
        assert 0.0 <= track.score <= 1.0
        for _ in range(10):
            track.predict()
            track.update(_radar_det())
        assert 0.0 <= track.score <= 1.0
