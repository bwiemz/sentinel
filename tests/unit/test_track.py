"""Tests for Track class and state machine."""

import numpy as np

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.track import Track


def _make_detection(x1=100, y1=100, x2=200, y2=200, cls="person", conf=0.9):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=conf,
    )


class TestTrack:
    def test_init(self):
        det = _make_detection()
        track = Track(det)
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1
        assert track.age == 0
        assert track.is_alive

    def test_init_position(self):
        det = _make_detection(100, 100, 200, 200)
        track = Track(det)
        # Center of [100,100,200,200] = (150, 150)
        np.testing.assert_allclose(track.position, [150, 150], atol=1)

    def test_predict_increments_age(self):
        track = Track(_make_detection())
        assert track.age == 0
        track.predict()
        assert track.age == 1

    def test_update_increments_hits(self):
        track = Track(_make_detection())
        assert track.hits == 1
        track.predict()
        track.update(_make_detection(105, 105, 205, 205))
        assert track.hits == 2
        assert track.consecutive_hits == 2

    def test_miss_increments(self):
        track = Track(_make_detection())
        track.predict()
        track.mark_missed()
        assert track.misses == 1
        assert track.consecutive_misses == 1
        assert track.consecutive_hits == 0

    def test_tentative_to_confirmed(self):
        track = Track(_make_detection(), confirm_hits=3)
        assert track.state == TrackState.TENTATIVE
        for i in range(3):
            track.predict()
            track.update(_make_detection(100 + i, 100, 200 + i, 200))
        assert track.state == TrackState.CONFIRMED

    def test_tentative_to_deleted(self):
        track = Track(_make_detection())
        for _ in range(3):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.DELETED
        assert not track.is_alive

    def test_confirmed_to_coasting(self):
        track = Track(_make_detection(), confirm_hits=1)
        track.predict()
        track.update(_make_detection())
        assert track.state == TrackState.CONFIRMED

        for _ in range(5):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.COASTING

    def test_coasting_to_deleted(self):
        track = Track(_make_detection(), confirm_hits=1, max_coast=5)
        track.predict()
        track.update(_make_detection())

        for _ in range(10):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.DELETED

    def test_coasting_reacquire(self):
        track = Track(_make_detection(), confirm_hits=1)
        track.predict()
        track.update(_make_detection())
        # Push to coasting
        for _ in range(5):
            track.predict()
            track.mark_missed()
        assert track.state == TrackState.COASTING
        # Re-acquire
        for _ in range(2):
            track.predict()
            track.update(_make_detection())
        assert track.state == TrackState.CONFIRMED

    def test_dominant_class(self):
        track = Track(_make_detection(cls="person"))
        for _ in range(3):
            track.predict()
            track.update(_make_detection(cls="person"))
        track.predict()
        track.update(_make_detection(cls="car"))
        assert track.dominant_class == "person"

    def test_score_range(self):
        track = Track(_make_detection())
        assert 0 <= track.score <= 1
        for _ in range(10):
            track.predict()
            track.update(_make_detection())
        assert 0 <= track.score <= 1

    def test_predicted_bbox(self):
        det = _make_detection(100, 100, 200, 200)
        track = Track(det)
        bbox = track.predicted_bbox
        assert bbox is not None
        assert len(bbox) == 4

    def test_to_dict(self):
        track = Track(_make_detection())
        d = track.to_dict()
        assert "track_id" in d
        assert "state" in d
        assert d["state"] == "tentative"
        assert "position" in d
        assert "velocity" in d

    def test_velocity_after_tracking(self):
        track = Track(_make_detection(100, 100, 200, 200), confirm_hits=1)
        # Move target rightward
        for i in range(20):
            track.predict()
            track.update(_make_detection(100 + i * 5, 100, 200 + i * 5, 200))
        # Velocity should indicate rightward motion
        vx, vy = track.velocity
        assert vx > 0  # Moving right
