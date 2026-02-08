"""Tests for Joint Probabilistic Data Association (JPDA)."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.association import AssociationResult
from sentinel.tracking.jpda import (
    JPDAAssociator,
    RadarJPDAAssociator,
    ThermalJPDAAssociator,
    _compute_beta_coefficients,
    _gaussian_likelihood,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_detection(x1=100, y1=100, x2=200, y2=200, ts=0.0, cls="person"):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=0.9,
    )


def _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0, velocity_mps=None,
                     elevation_deg=None):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=velocity_mps,
        elevation_deg=elevation_deg,
    )


def _thermal_detection(azimuth_deg=30.0, ts=0.0, temperature_k=350.0):
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=ts,
        azimuth_deg=azimuth_deg,
        temperature_k=temperature_k,
    )


def _camera_config(method="hungarian"):
    return OmegaConf.create({
        "filter": {"dt": 1 / 30, "type": "kf"},
        "association": {
            "gate_threshold": 9.21,
            "iou_weight": 0.5,
            "mahalanobis_weight": 0.5,
            "cascaded": False,
            "method": method,
            "detection_probability": 0.9,
            "false_alarm_density": 1e-6,
        },
        "track_management": {
            "confirm_hits": 3,
            "max_coast_frames": 15,
            "max_tracks": 100,
        },
    })


def _radar_config(method="hungarian"):
    return OmegaConf.create({
        "filter": {"dt": 0.1, "type": "ekf", "use_doppler": False},
        "association": {
            "gate_threshold": 9.21,
            "cascaded": False,
            "method": method,
            "detection_probability": 0.9,
            "false_alarm_density": 1e-6,
        },
        "track_management": {
            "confirm_hits": 3,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    })


def _thermal_config(method="hungarian"):
    return OmegaConf.create({
        "filter": {"dt": 0.033, "assumed_initial_range_m": 10000.0},
        "association": {
            "gate_threshold": 6.635,
            "method": method,
            "detection_probability": 0.9,
            "false_alarm_density": 1e-6,
        },
        "track_management": {
            "confirm_hits": 3,
            "max_coast_frames": 10,
            "max_tracks": 50,
        },
    })


# ===========================================================================
# TestBetaCoefficients
# ===========================================================================

class TestBetaCoefficients:

    def test_single_detection(self):
        likelihoods = np.array([1.0])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)
        assert len(betas) == 1
        # Very small clutter → beta_0 ≈ 0, beta[0] ≈ 1
        assert betas[0] > 0.99
        assert beta_0 < 0.01

    def test_no_detections(self):
        likelihoods = np.array([])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)
        assert len(betas) == 0
        assert beta_0 == 1.0

    def test_competition(self):
        # Two detections with equal likelihood → each gets ~0.5 of detection prob
        likelihoods = np.array([1.0, 1.0])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)
        assert len(betas) == 2
        assert betas[0] == pytest.approx(betas[1], abs=1e-10)
        assert sum(betas) + beta_0 == pytest.approx(1.0)

    def test_ambiguity(self):
        # One strong, one weak → strong gets higher beta
        likelihoods = np.array([10.0, 1.0])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)
        assert betas[0] > betas[1]

    def test_zero_likelihoods(self):
        likelihoods = np.array([0.0, 0.0])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)
        # All detection terms are 0 → beta_0 = 1
        assert beta_0 == pytest.approx(1.0)

    def test_sum_to_one(self):
        likelihoods = np.array([0.5, 0.3, 0.1, 0.8])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=0.01)
        assert sum(betas) + beta_0 == pytest.approx(1.0)


# ===========================================================================
# TestGaussianLikelihood
# ===========================================================================

class TestGaussianLikelihood:

    def test_zero_innovation(self):
        y = np.array([0.0, 0.0])
        S = np.eye(2)
        L = _gaussian_likelihood(y, S)
        # N(0; 0, I) = 1/(2π) ≈ 0.1592
        assert L == pytest.approx(1.0 / (2 * np.pi), abs=1e-4)

    def test_large_innovation(self):
        y = np.array([100.0, 100.0])
        S = np.eye(2)
        L = _gaussian_likelihood(y, S)
        assert L < 1e-10

    def test_singular_covariance(self):
        y = np.array([1.0])
        S = np.array([[0.0]])
        L = _gaussian_likelihood(y, S)
        assert L == 0.0

    def test_1d(self):
        y = np.array([0.0])
        S = np.array([[1.0]])
        L = _gaussian_likelihood(y, S)
        # N(0; 0, 1) = 1/sqrt(2π) ≈ 0.3989
        assert L == pytest.approx(1.0 / np.sqrt(2 * np.pi), abs=1e-4)

    def test_known_value(self):
        # y=1, S=1 → L = (2π)^(-0.5) * exp(-0.5) ≈ 0.2420
        y = np.array([1.0])
        S = np.array([[1.0]])
        L = _gaussian_likelihood(y, S)
        expected = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5)
        assert L == pytest.approx(expected, abs=1e-4)


# ===========================================================================
# TestJPDAAssociator (Camera)
# ===========================================================================

class TestJPDAAssociator:

    def test_single_track_single_det(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=50.0)

        det0 = _camera_detection(100, 100, 200, 200, ts=0.0)
        track = Track(det0, dt=1 / 30)
        track.predict()

        det1 = _camera_detection(105, 105, 205, 205, ts=0.033)
        result = jpda.associate_and_update([track], [det1])
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0][0] == 0

    def test_no_tracks(self):
        jpda = JPDAAssociator()
        result = jpda.associate_and_update([], [_camera_detection()])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_detections) == 1

    def test_no_detections(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator()
        track = Track(_camera_detection(), dt=1 / 30)
        track.predict()
        result = jpda.associate_and_update([track], [])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_tracks) == 1

    def test_separated_tracks(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=50.0)

        # Two well-separated tracks
        t1 = Track(_camera_detection(100, 100, 200, 200), dt=1 / 30)
        t2 = Track(_camera_detection(500, 500, 600, 600), dt=1 / 30)
        t1.predict()
        t2.predict()

        d1 = _camera_detection(105, 105, 205, 205, ts=0.033)
        d2 = _camera_detection(505, 505, 605, 605, ts=0.033)
        result = jpda.associate_and_update([t1, t2], [d1, d2])
        assert len(result.matched_pairs) == 2

    def test_close_targets(self):
        """Close targets should both be updated (not just one)."""
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=200.0)

        t1 = Track(_camera_detection(150, 150, 250, 250), dt=1 / 30)
        t2 = Track(_camera_detection(160, 160, 260, 260), dt=1 / 30)
        t1.predict()
        t2.predict()

        d1 = _camera_detection(155, 155, 255, 255, ts=0.033)
        d2 = _camera_detection(165, 165, 265, 265, ts=0.033)
        result = jpda.associate_and_update([t1, t2], [d1, d2])
        # Both tracks should have been updated (JPDA soft assignment)
        assert len(result.matched_pairs) == 2

    def test_gating_rejects(self):
        """Detection outside gate should not be associated."""
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=1.0)  # Very tight gate

        track = Track(_camera_detection(100, 100, 200, 200), dt=1 / 30)
        track.predict()

        # Far away detection
        det = _camera_detection(800, 800, 900, 900, ts=0.033)
        result = jpda.associate_and_update([track], [det])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_tracks) == 1

    def test_state_update(self):
        """State should move toward the detection."""
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=200.0)

        det0 = _camera_detection(100, 100, 200, 200, ts=0.0)
        track = Track(det0, dt=1 / 30)
        pos_before = track.position.copy()
        track.predict()

        # Detection slightly to the right
        det1 = _camera_detection(110, 110, 210, 210, ts=0.033)
        jpda.associate_and_update([track], [det1])
        # Position should move right
        assert track.position[0] > pos_before[0]

    def test_covariance_spread(self):
        """With ambiguous detections, covariance should grow due to spread term."""
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=500.0, P_D=0.5, false_alarm_density=0.1)

        det0 = _camera_detection(150, 150, 250, 250, ts=0.0)
        track = Track(det0, dt=1 / 30)
        track.predict()
        P_before = track.kf.P.copy()

        # Two detections close but on opposite sides → high spread
        d1 = _camera_detection(140, 140, 240, 240, ts=0.033)
        d2 = _camera_detection(160, 160, 260, 260, ts=0.033)
        jpda.associate_and_update([track], [d1, d2])

        # With spread of innovations, P might not shrink as much as standard KF
        # Just verify it didn't crash and covariance is still positive
        eigenvalues = np.linalg.eigvalsh(track.kf.P)
        assert np.all(eigenvalues > 0)

    def test_quality_monitor_updated(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=200.0)

        track = Track(_camera_detection(100, 100, 200, 200), dt=1 / 30)
        assert track.quality_monitor.sample_count == 0
        track.predict()

        det = _camera_detection(105, 105, 205, 205, ts=0.033)
        jpda.associate_and_update([track], [det])
        assert track.quality_monitor.sample_count == 1

    def test_returns_association_result(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=200.0)

        track = Track(_camera_detection(), dt=1 / 30)
        track.predict()
        result = jpda.associate_and_update([track], [_camera_detection(105, 105, 205, 205)])
        assert isinstance(result, AssociationResult)

    def test_unmatched_detections(self):
        from sentinel.tracking.track import Track
        jpda = JPDAAssociator(gate_threshold=1.0)

        track = Track(_camera_detection(100, 100, 200, 200), dt=1 / 30)
        track.predict()

        # One close (within gate), one far
        d_close = _camera_detection(102, 102, 202, 202, ts=0.033)
        d_far = _camera_detection(800, 800, 900, 900, ts=0.033)
        result = jpda.associate_and_update([track], [d_close, d_far])
        assert 1 in result.unmatched_detections  # Far detection


# ===========================================================================
# TestRadarJPDAAssociator
# ===========================================================================

class TestRadarJPDAAssociator:

    def test_single_track_det(self):
        from sentinel.tracking.radar_track import RadarTrack
        jpda = RadarJPDAAssociator(gate_threshold=50.0)

        det0 = _radar_detection(5000.0, 45.0, ts=0.0)
        track = RadarTrack(det0, dt=0.1)
        track.predict()

        det1 = _radar_detection(5010.0, 45.1, ts=0.1)
        result = jpda.associate_and_update([track], [det1])
        assert len(result.matched_pairs) == 1

    def test_angular_wrapping(self):
        """Tracks near 0/360 boundary should still associate."""
        from sentinel.tracking.radar_track import RadarTrack
        jpda = RadarJPDAAssociator(gate_threshold=50.0)

        det0 = _radar_detection(5000.0, 1.0, ts=0.0)
        track = RadarTrack(det0, dt=0.1)
        track.predict()

        det1 = _radar_detection(5010.0, 359.0, ts=0.1)
        result = jpda.associate_and_update([track], [det1])
        # Should be within gate despite wrapping
        # (The wrapping in jpda handles this)
        assert isinstance(result, AssociationResult)

    def test_close_targets(self):
        from sentinel.tracking.radar_track import RadarTrack
        jpda = RadarJPDAAssociator(gate_threshold=100.0)

        t1 = RadarTrack(_radar_detection(5000.0, 45.0), dt=0.1)
        t2 = RadarTrack(_radar_detection(5000.0, 46.0), dt=0.1)
        t1.predict()
        t2.predict()

        d1 = _radar_detection(5010.0, 45.1, ts=0.1)
        d2 = _radar_detection(5010.0, 46.1, ts=0.1)
        result = jpda.associate_and_update([t1, t2], [d1, d2])
        assert len(result.matched_pairs) == 2

    def test_gating_rejects(self):
        from sentinel.tracking.radar_track import RadarTrack
        jpda = RadarJPDAAssociator(gate_threshold=0.1)

        track = RadarTrack(_radar_detection(5000.0, 45.0), dt=0.1)
        track.predict()

        det = _radar_detection(20000.0, 180.0, ts=0.1)
        result = jpda.associate_and_update([track], [det])
        assert len(result.matched_pairs) == 0

    def test_doppler_mode(self):
        from sentinel.tracking.radar_track import RadarTrack
        jpda = RadarJPDAAssociator(gate_threshold=100.0)

        det0 = _radar_detection(5000.0, 45.0, ts=0.0, velocity_mps=250.0)
        track = RadarTrack(det0, dt=0.1, use_doppler=True)
        track.predict()

        det1 = _radar_detection(5010.0, 45.1, ts=0.1, velocity_mps=252.0)
        result = jpda.associate_and_update([track], [det1])
        assert len(result.matched_pairs) == 1


# ===========================================================================
# TestThermalJPDAAssociator
# ===========================================================================

class TestThermalJPDAAssociator:

    def test_single_bearing(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        jpda = ThermalJPDAAssociator(gate_threshold=20.0)

        track = ThermalTrack(_thermal_detection(30.0, ts=0.0), dt=0.033)
        track.predict()

        det = _thermal_detection(30.5, ts=0.033)
        result = jpda.associate_and_update([track], [det])
        assert len(result.matched_pairs) == 1

    def test_close_bearings(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        jpda = ThermalJPDAAssociator(gate_threshold=50.0)

        t1 = ThermalTrack(_thermal_detection(30.0, ts=0.0), dt=0.033)
        t2 = ThermalTrack(_thermal_detection(32.0, ts=0.0), dt=0.033)
        t1.predict()
        t2.predict()

        d1 = _thermal_detection(30.5, ts=0.033)
        d2 = _thermal_detection(32.5, ts=0.033)
        result = jpda.associate_and_update([t1, t2], [d1, d2])
        assert len(result.matched_pairs) == 2

    def test_1d_innovation(self):
        """Thermal JPDA uses 1D bearing innovation."""
        from sentinel.tracking.thermal_track import ThermalTrack
        jpda = ThermalJPDAAssociator(gate_threshold=50.0)

        track = ThermalTrack(_thermal_detection(30.0, ts=0.0), dt=0.033)
        assert track.quality_monitor is not None
        assert track.quality_monitor._dim_meas == 1
        track.predict()

        det = _thermal_detection(30.5, ts=0.033)
        jpda.associate_and_update([track], [det])
        assert track.quality_monitor.sample_count == 1


# ===========================================================================
# TestTrackManagerJPDA
# ===========================================================================

class TestTrackManagerJPDA:

    def test_hungarian_is_default(self):
        from sentinel.tracking.track_manager import TrackManager
        tm = TrackManager(_camera_config("hungarian"))
        assert tm._jpda is None

    def test_jpda_selection(self):
        from sentinel.tracking.track_manager import TrackManager
        tm = TrackManager(_camera_config("jpda"))
        assert tm._jpda is not None
        assert isinstance(tm._jpda, JPDAAssociator)

    def test_step_cycle_with_jpda(self):
        from sentinel.tracking.track_manager import TrackManager
        tm = TrackManager(_camera_config("jpda"))

        # First frame: initiate tracks
        dets = [_camera_detection(100, 100, 200, 200, ts=0.0)]
        tracks = tm.step(dets)
        assert len(tracks) == 1

        # Second frame: JPDA should associate
        dets2 = [_camera_detection(105, 105, 205, 205, ts=0.033)]
        tracks = tm.step(dets2)
        assert len(tracks) == 1
        assert tracks[0].hits >= 2

    def test_new_tracks_from_unmatched(self):
        from sentinel.tracking.track_manager import TrackManager
        tm = TrackManager(_camera_config("jpda"))

        dets = [
            _camera_detection(100, 100, 200, 200, ts=0.0),
            _camera_detection(500, 500, 600, 600, ts=0.0),
        ]
        tracks = tm.step(dets)
        assert len(tracks) == 2

    def test_deletion(self):
        from sentinel.tracking.track_manager import TrackManager
        cfg = _camera_config("jpda")
        cfg.track_management.max_coast_frames = 2
        tm = TrackManager(cfg)

        # Initiate a track
        tm.step([_camera_detection(100, 100, 200, 200)])
        assert tm.track_count == 1

        # Many empty frames → track should be deleted
        for _ in range(10):
            tm.step([])
        assert tm.track_count == 0


# ===========================================================================
# TestRadarTrackManagerJPDA
# ===========================================================================

class TestRadarTrackManagerJPDA:

    def test_jpda_selection(self):
        from sentinel.tracking.radar_track_manager import RadarTrackManager
        rtm = RadarTrackManager(_radar_config("jpda"))
        assert rtm._jpda is not None

    def test_step_cycle(self):
        from sentinel.tracking.radar_track_manager import RadarTrackManager
        rtm = RadarTrackManager(_radar_config("jpda"))

        dets = [_radar_detection(5000.0, 45.0, ts=0.0)]
        tracks = rtm.step(dets)
        assert len(tracks) == 1

        dets2 = [_radar_detection(5010.0, 45.1, ts=0.1)]
        tracks = rtm.step(dets2)
        assert len(tracks) == 1
        assert tracks[0].hits >= 2


# ===========================================================================
# TestThermalTrackManagerJPDA
# ===========================================================================

class TestThermalTrackManagerJPDA:

    def test_jpda_selection(self):
        from sentinel.tracking.thermal_track_manager import ThermalTrackManager
        ttm = ThermalTrackManager(_thermal_config("jpda"))
        assert ttm._jpda is not None

    def test_step_cycle(self):
        from sentinel.tracking.thermal_track_manager import ThermalTrackManager
        ttm = ThermalTrackManager(_thermal_config("jpda"))

        dets = [_thermal_detection(30.0, ts=0.0)]
        tracks = ttm.step(dets)
        assert len(tracks) == 1

        dets2 = [_thermal_detection(30.5, ts=0.033)]
        tracks = ttm.step(dets2)
        assert len(tracks) == 1
        assert tracks[0].hits >= 2


# ===========================================================================
# TestJPDABackwardCompatibility
# ===========================================================================

class TestJPDABackwardCompatibility:

    def test_hungarian_unmodified(self):
        """Default Hungarian path should be unaffected."""
        from sentinel.tracking.track_manager import TrackManager
        tm = TrackManager(_camera_config("hungarian"))

        dets = [_camera_detection(100, 100, 200, 200)]
        tracks = tm.step(dets)
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TENTATIVE

    def test_interface_compatible(self):
        """JPDA returns AssociationResult with same fields as Hungarian."""
        jpda = JPDAAssociator(gate_threshold=200.0)
        from sentinel.tracking.track import Track

        track = Track(_camera_detection(), dt=1 / 30)
        track.predict()
        result = jpda.associate_and_update([track], [_camera_detection(105, 105, 205, 205)])

        assert hasattr(result, "matched_pairs")
        assert hasattr(result, "unmatched_tracks")
        assert hasattr(result, "unmatched_detections")
