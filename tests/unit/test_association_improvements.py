"""Tests for association improvements: velocity gating, cascaded, config wiring."""

import numpy as np

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.association import HungarianAssociator
from sentinel.tracking.radar_association import RadarAssociator
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.track import Track


def _camera_det(x1=100, y1=100, x2=200, y2=200, cls="person"):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=0.9,
    )


def _radar_det(range_m=5000.0, azimuth_deg=30.0, velocity_mps=None):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=velocity_mps,
        confidence=0.9,
    )


class TestCascadedCameraAssociation:
    """Test cascaded association for camera tracks."""

    def test_cascaded_disabled_by_default(self):
        assoc = HungarianAssociator()
        assert assoc._cascaded is False

    def test_cascaded_enabled(self):
        assoc = HungarianAssociator(cascaded=True)
        assert assoc._cascaded is True

    def test_confirmed_gets_priority(self):
        """Confirmed track should win over tentative for the same detection."""
        assoc = HungarianAssociator(cascaded=True, gate_threshold=100.0)

        # Create two tracks at the same position
        det = _camera_det(100, 100, 200, 200)
        t_tentative = Track(det, confirm_hits=10)  # stays tentative
        t_confirmed = Track(det, confirm_hits=1)  # immediately TENTATIVE (needs 1 more hit)

        # Confirm the second track
        t_confirmed._record_hit()  # now confirmed
        assert t_confirmed.state == TrackState.CONFIRMED
        assert t_tentative.state == TrackState.TENTATIVE

        # Predict both
        t_tentative.predict()
        t_confirmed.predict()

        # One detection near both tracks
        new_det = _camera_det(105, 105, 205, 205)
        tracks = [t_tentative, t_confirmed]

        result = assoc.associate(tracks, [new_det])
        # Confirmed track (index 1) should match
        assert len(result.matched_pairs) == 1
        matched_track_idx = result.matched_pairs[0][0]
        assert matched_track_idx == 1  # confirmed track wins

    def test_non_cascaded_still_works(self):
        assoc = HungarianAssociator(cascaded=False)
        det = _camera_det()
        track = Track(det)
        track.predict()
        new_det = _camera_det(105, 105, 205, 205)
        result = assoc.associate([track], [new_det])
        assert len(result.matched_pairs) == 1


class TestCascadedRadarAssociation:
    """Test cascaded association for radar tracks."""

    def test_confirmed_gets_priority(self):
        assoc = RadarAssociator(cascaded=True, gate_threshold=100.0)

        det = _radar_det(5000.0, 30.0)
        t_tent = RadarTrack(det, confirm_hits=10)
        t_conf = RadarTrack(det, confirm_hits=1)
        t_conf._record_hit()
        assert t_conf.state == TrackState.CONFIRMED

        t_tent.predict()
        t_conf.predict()

        new_det = _radar_det(5001.0, 30.01)
        result = assoc.associate([t_tent, t_conf], [new_det])
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0][0] == 1

    def test_second_pass_uses_remaining(self):
        """Tentative tracks should get remaining detections after confirmed pass."""
        assoc = RadarAssociator(cascaded=True, gate_threshold=100.0)

        det1 = _radar_det(5000.0, 30.0)
        det2 = _radar_det(8000.0, 60.0)
        t_conf = RadarTrack(det1, confirm_hits=1)
        t_conf._record_hit()
        t_tent = RadarTrack(det2, confirm_hits=10)

        t_conf.predict()
        t_tent.predict()

        # Two detections: one near each track
        d1 = _radar_det(5001.0, 30.01)
        d2 = _radar_det(8001.0, 60.01)

        result = assoc.associate([t_conf, t_tent], [d1, d2])
        assert len(result.matched_pairs) == 2
        assert len(result.unmatched_tracks) == 0
        assert len(result.unmatched_detections) == 0


class TestVelocityGating:
    """Test velocity gating in radar association."""

    def test_velocity_gate_disabled_by_default(self):
        assoc = RadarAssociator()
        assert assoc._velocity_gate is None

    def test_velocity_gate_rejects(self):
        """Detection with mismatched velocity should be rejected."""
        assoc = RadarAssociator(gate_threshold=100.0, velocity_gate_mps=10.0)

        det = _radar_det(5000.0, 30.0)
        track = RadarTrack(det)
        track.predict()

        # Track velocity is ~0, detection says 100 m/s -> rejected
        fast_det = _radar_det(5001.0, 30.01, velocity_mps=100.0)
        result = assoc.associate([track], [fast_det])
        assert len(result.matched_pairs) == 0

    def test_velocity_gate_accepts(self):
        """Detection with matching velocity should be accepted."""
        assoc = RadarAssociator(gate_threshold=100.0, velocity_gate_mps=50.0)

        det = _radar_det(5000.0, 30.0)
        track = RadarTrack(det)
        track.predict()

        # Track velocity ~0, detection velocity 5 -> within gate
        slow_det = _radar_det(5001.0, 30.01, velocity_mps=5.0)
        result = assoc.associate([track], [slow_det])
        assert len(result.matched_pairs) == 1

    def test_velocity_gate_null_velocity_ok(self):
        """Detection without velocity_mps should pass velocity gate."""
        assoc = RadarAssociator(gate_threshold=100.0, velocity_gate_mps=10.0)

        det = _radar_det(5000.0, 30.0)
        track = RadarTrack(det)
        track.predict()

        # No velocity in detection -> velocity gate doesn't apply
        no_vel_det = _radar_det(5001.0, 30.01, velocity_mps=None)
        result = assoc.associate([track], [no_vel_det])
        assert len(result.matched_pairs) == 1


class TestAssociationEdgeCases:
    """Test edge cases in association."""

    def test_empty_tracks(self):
        assoc = HungarianAssociator()
        result = assoc.associate([], [_camera_det()])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_detections) == 1

    def test_empty_detections(self):
        assoc = HungarianAssociator()
        track = Track(_camera_det())
        result = assoc.associate([track], [])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_tracks) == 1

    def test_cascaded_all_tentative(self):
        """Cascaded with no confirmed tracks should still work."""
        assoc = HungarianAssociator(cascaded=True)
        det = _camera_det()
        track = Track(det, confirm_hits=10)  # stays tentative
        track.predict()
        result = assoc.associate([track], [_camera_det(105, 105, 205, 205)])
        assert len(result.matched_pairs) == 1

    def test_cascaded_all_confirmed(self):
        """Cascaded with no tentative tracks should still work."""
        assoc = RadarAssociator(cascaded=True, gate_threshold=100.0)
        det = _radar_det()
        track = RadarTrack(det, confirm_hits=1)
        track._record_hit()
        track.predict()
        result = assoc.associate([track], [_radar_det(5001.0, 30.01)])
        assert len(result.matched_pairs) == 1
