"""Tests for temporal alignment and predict_to_time functionality."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.fusion.temporal_alignment import (
    AlignedTrackState,
    align_tracks_to_epoch,
    build_cv_process_noise,
    build_cv_transition,
    predict_track_to_epoch,
    _get_sensor_type,
    _extract_position_cov,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_detection(x1=100, y1=100, x2=200, y2=200, ts=0.0):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=ts,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_id=0,
        class_name="person",
        confidence=0.9,
    )


def _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
    )


def _thermal_detection(azimuth_deg=30.0, ts=0.0):
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=ts,
        azimuth_deg=azimuth_deg,
        temperature_k=350.0,
    )


# ===========================================================================
# TestBuildCVTransition
# ===========================================================================

class TestBuildCVTransition:

    def test_zero_dt(self):
        F = build_cv_transition(0.0, dim=2)
        np.testing.assert_array_equal(F, np.eye(4))

    def test_correct_structure_2d(self):
        F = build_cv_transition(0.5, dim=2)
        assert F.shape == (4, 4)
        assert F[0, 1] == 0.5  # x += vx*dt
        assert F[2, 3] == 0.5  # y += vy*dt
        assert F[0, 0] == 1.0
        assert F[1, 1] == 1.0
        # Off-diagonal cross terms should be zero
        assert F[0, 2] == 0.0
        assert F[0, 3] == 0.0

    def test_3d(self):
        F = build_cv_transition(1.0, dim=3)
        assert F.shape == (6, 6)
        assert F[0, 1] == 1.0  # x += vx*dt
        assert F[2, 3] == 1.0  # y += vy*dt
        assert F[4, 5] == 1.0  # z += vz*dt

    def test_negative_dt(self):
        F = build_cv_transition(-0.5, dim=2)
        assert F[0, 1] == -0.5  # Backward prediction


# ===========================================================================
# TestBuildCVProcessNoise
# ===========================================================================

class TestBuildCVProcessNoise:

    def test_shape(self):
        Q = build_cv_process_noise(0.1, dim=2, sigma_a=1.0)
        assert Q.shape == (4, 4)

    def test_positive_semidefinite(self):
        Q = build_cv_process_noise(0.1, dim=2, sigma_a=5.0)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-12)

    def test_scales_with_sigma(self):
        Q1 = build_cv_process_noise(0.1, dim=2, sigma_a=1.0)
        Q2 = build_cv_process_noise(0.1, dim=2, sigma_a=2.0)
        # Q scales with sigma^2
        np.testing.assert_allclose(Q2, Q1 * 4.0, atol=1e-12)

    def test_3d_shape(self):
        Q = build_cv_process_noise(0.1, dim=3, sigma_a=1.0)
        assert Q.shape == (6, 6)


# ===========================================================================
# TestPredictToTime
# ===========================================================================

class TestPredictToTime:

    def test_camera_forward_prediction(self):
        from sentinel.tracking.track import Track
        det = _camera_detection(100, 100, 200, 200, ts=0.0)
        track = Track(det, dt=1 / 30)
        track.last_update_time = 0.0

        x_pred, P_pred = track.predict_to_time(1.0)
        assert x_pred.shape == (4,)
        assert P_pred.shape == (4, 4)
        # Position should still be near (150,150) since velocity is 0
        assert abs(x_pred[0] - 150) < 10
        assert abs(x_pred[2] - 150) < 10

    def test_camera_zero_dt_returns_copy(self):
        from sentinel.tracking.track import Track
        det = _camera_detection(ts=0.5)
        track = Track(det, dt=1 / 30)
        track.last_update_time = 0.5

        x_pred, P_pred = track.predict_to_time(0.5)
        np.testing.assert_array_equal(x_pred, track.kf.x)
        np.testing.assert_array_equal(P_pred, track.kf.P)
        # Ensure they are copies
        x_pred[0] = 999
        assert track.kf.x[0] != 999

    def test_camera_no_mutation(self):
        from sentinel.tracking.track import Track
        det = _camera_detection(ts=0.0)
        track = Track(det, dt=1 / 30)
        track.last_update_time = 0.0

        x_before = track.kf.x.copy()
        P_before = track.kf.P.copy()
        track.predict_to_time(1.0)
        np.testing.assert_array_equal(track.kf.x, x_before)
        np.testing.assert_array_equal(track.kf.P, P_before)

    def test_radar_forward_prediction(self):
        from sentinel.tracking.radar_track import RadarTrack
        det = _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0)
        track = RadarTrack(det, dt=0.1)
        track.last_update_time = 0.0

        x_pred, P_pred = track.predict_to_time(0.5)
        assert x_pred.shape == (4,)
        assert P_pred.shape == (4, 4)

    def test_radar_no_mutation(self):
        from sentinel.tracking.radar_track import RadarTrack
        det = _radar_detection(ts=0.0)
        track = RadarTrack(det, dt=0.1)
        track.last_update_time = 0.0

        x_before = track.ekf.x.copy()
        P_before = track.ekf.P.copy()
        track.predict_to_time(1.0)
        np.testing.assert_array_equal(track.ekf.x, x_before)
        np.testing.assert_array_equal(track.ekf.P, P_before)

    def test_thermal_forward_prediction(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        det = _thermal_detection(azimuth_deg=30.0, ts=0.0)
        track = ThermalTrack(det, dt=0.033)
        track.last_update_time = 0.0

        x_pred, P_pred = track.predict_to_time(0.5)
        assert x_pred.shape == (4,)
        assert P_pred.shape == (4, 4)

    def test_thermal_no_mutation(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        det = _thermal_detection(ts=0.0)
        track = ThermalTrack(det, dt=0.033)
        track.last_update_time = 0.0

        x_before = track.ekf.x.copy()
        P_before = track.ekf.P.copy()
        track.predict_to_time(1.0)
        np.testing.assert_array_equal(track.ekf.x, x_before)
        np.testing.assert_array_equal(track.ekf.P, P_before)

    def test_covariance_grows_with_time(self):
        from sentinel.tracking.track import Track
        det = _camera_detection(ts=0.0)
        track = Track(det, dt=1 / 30)
        track.last_update_time = 0.0

        _, P_short = track.predict_to_time(0.1)
        _, P_long = track.predict_to_time(1.0)
        # Covariance should be larger for longer prediction
        assert np.trace(P_long) > np.trace(P_short)


# ===========================================================================
# TestAlignedTrackState
# ===========================================================================

class TestAlignedTrackState:

    def test_dataclass_fields(self):
        state = AlignedTrackState(
            position=np.array([100.0, 200.0]),
            covariance=np.eye(2),
            track_id="T-001",
            sensor_type="camera",
            alignment_time=1.5,
            original_track=None,
        )
        assert state.track_id == "T-001"
        assert state.sensor_type == "camera"
        assert state.alignment_time == 1.5
        np.testing.assert_array_equal(state.position, [100.0, 200.0])

    def test_sensor_type_detection(self):
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        cam = Track(_camera_detection())
        radar = RadarTrack(_radar_detection())
        thermal = ThermalTrack(_thermal_detection())

        assert _get_sensor_type(cam) == "camera"
        assert _get_sensor_type(radar) == "radar"
        assert _get_sensor_type(thermal) == "thermal"


# ===========================================================================
# TestExtractPositionCov
# ===========================================================================

class TestExtractPositionCov:

    def test_4d_extraction(self):
        x = np.array([10.0, 1.0, 20.0, 2.0])
        P = np.diag([100.0, 10.0, 200.0, 20.0])
        pos, cov = _extract_position_cov(x, P)
        np.testing.assert_array_equal(pos, [10.0, 20.0])
        expected_cov = np.array([[100.0, 0.0], [0.0, 200.0]])
        np.testing.assert_array_equal(cov, expected_cov)


# ===========================================================================
# TestAlignTracksToEpoch
# ===========================================================================

class TestAlignTracksToEpoch:

    def test_empty_list(self):
        result = align_tracks_to_epoch([], 1.0)
        assert result == []

    def test_single_track(self):
        from sentinel.tracking.track import Track
        track = Track(_camera_detection(ts=0.0), dt=1 / 30)
        track.last_update_time = 0.0

        result = align_tracks_to_epoch([track], 1.0)
        assert len(result) == 1
        assert result[0].alignment_time == 1.0
        assert result[0].sensor_type == "camera"

    def test_multiple_tracks(self):
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack

        cam = Track(_camera_detection(ts=0.0), dt=1 / 30)
        cam.last_update_time = 0.0
        radar = RadarTrack(_radar_detection(ts=0.05), dt=0.1)
        radar.last_update_time = 0.05

        result = align_tracks_to_epoch([cam, radar], 0.1)
        assert len(result) == 2
        assert result[0].sensor_type == "camera"
        assert result[1].sensor_type == "radar"
        # Both should be aligned to the same time
        assert result[0].alignment_time == 0.1
        assert result[1].alignment_time == 0.1

    def test_mixed_sensors(self):
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        cam = Track(_camera_detection(ts=0.0), dt=1 / 30)
        cam.last_update_time = 0.0
        radar = RadarTrack(_radar_detection(ts=0.0), dt=0.1)
        radar.last_update_time = 0.0
        thermal = ThermalTrack(_thermal_detection(ts=0.0), dt=0.033)
        thermal.last_update_time = 0.0

        result = align_tracks_to_epoch([cam, radar, thermal], 0.5)
        assert len(result) == 3
        types = {r.sensor_type for r in result}
        assert types == {"camera", "radar", "thermal"}

    def test_fields_correct(self):
        from sentinel.tracking.track import Track
        track = Track(_camera_detection(ts=0.0), dt=1 / 30)
        track.last_update_time = 0.0

        result = predict_track_to_epoch(track, 0.5)
        assert isinstance(result, AlignedTrackState)
        assert result.position.shape == (2,)
        assert result.covariance.shape == (2, 2)
        assert result.track_id == track.track_id
        assert result.original_track is track
