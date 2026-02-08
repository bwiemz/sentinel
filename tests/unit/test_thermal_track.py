"""Tests for bearing-only EKF and thermal tracking."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.filters import BearingOnlyEKF
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.tracking.thermal_track_manager import ThermalTrackManager

# === BearingOnlyEKF ===


class TestBearingOnlyEKF:
    def test_init(self):
        ekf = BearingOnlyEKF(dt=0.1)
        assert ekf.dim_state == 4
        assert ekf.dim_meas == 1

    def test_predict_moves_state(self):
        ekf = BearingOnlyEKF(dt=0.1)
        ekf.x = np.array([1000.0, 10.0, 500.0, 5.0])
        ekf.predict()
        assert ekf.x[0] == pytest.approx(1001.0, abs=0.01)
        assert ekf.x[2] == pytest.approx(500.5, abs=0.01)

    def test_h_function_along_x(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([1000.0, 0.0, 0.0, 0.0])
        h = ekf.h(ekf.x)
        assert h[0] == pytest.approx(0.0)  # azimuth = 0 along x-axis

    def test_h_function_along_y(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([0.0, 0.0, 1000.0, 0.0])
        h = ekf.h(ekf.x)
        assert h[0] == pytest.approx(np.pi / 2)

    def test_jacobian_shape(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([1000.0, 0.0, 500.0, 0.0])
        H = ekf.H_jacobian(ekf.x)
        assert H.shape == (1, 4)

    def test_update_bearing(self):
        ekf = BearingOnlyEKF(dt=0.1)
        ekf.x = np.array([1000.0, 0.0, 0.0, 0.0])
        ekf.P = np.eye(4) * 100.0
        z = np.array([0.1])  # Slight bearing offset
        ekf.update(z)
        # State should adjust y component toward positive
        assert ekf.x[2] > 0  # y moved toward bearing

    def test_gating_distance_close(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([1000.0, 0.0, 0.0, 0.0])
        ekf.P = np.eye(4) * 10.0
        z = np.array([0.001])  # Very close to predicted bearing (0)
        dist = ekf.gating_distance(z)
        assert dist < 5.0

    def test_gating_distance_far(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([1000.0, 0.0, 0.0, 0.0])
        ekf.P = np.eye(4) * 10.0
        z = np.array([1.0])  # ~57 degrees off
        dist = ekf.gating_distance(z)
        assert dist > 100.0

    def test_position_property(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([100.0, 1.0, 200.0, 2.0])
        np.testing.assert_array_equal(ekf.position, [100.0, 200.0])

    def test_velocity_property(self):
        ekf = BearingOnlyEKF()
        ekf.x = np.array([100.0, 1.0, 200.0, 2.0])
        np.testing.assert_array_equal(ekf.velocity, [1.0, 2.0])


# === ThermalTrack ===


def _thermal_det(azimuth_deg=10.0, temperature_k=800.0):
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=1.0,
        azimuth_deg=azimuth_deg,
        temperature_k=temperature_k,
        thermal_band="mwir",
    )


class TestThermalTrack:
    def test_creation(self):
        det = _thermal_det()
        track = ThermalTrack(det, assumed_range_m=5000.0)
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1
        assert track.is_alive

    def test_azimuth_property(self):
        det = _thermal_det(azimuth_deg=0.0)
        track = ThermalTrack(det, assumed_range_m=5000.0)
        assert abs(track.azimuth_deg) < 1.0  # Close to 0

    def test_temperature_property(self):
        det = _thermal_det(temperature_k=1500.0)
        track = ThermalTrack(det)
        assert track.temperature_k == 1500.0

    def test_predict_increments_age(self):
        track = ThermalTrack(_thermal_det())
        track.predict()
        assert track.age == 1

    def test_update_increments_hits(self):
        track = ThermalTrack(_thermal_det())
        track.update(_thermal_det())
        assert track.hits == 2
        assert track.consecutive_hits == 2

    def test_mark_missed(self):
        track = ThermalTrack(_thermal_det())
        track.mark_missed()
        assert track.misses == 1
        assert track.consecutive_misses == 1

    def test_lifecycle_tentative_to_confirmed(self):
        track = ThermalTrack(_thermal_det(), confirm_hits=3)
        track.predict()
        track.update(_thermal_det())
        track.predict()
        track.update(_thermal_det())
        assert track.state == TrackState.CONFIRMED

    def test_lifecycle_tentative_to_deleted(self):
        track = ThermalTrack(_thermal_det())
        track.mark_missed()
        track.mark_missed()
        track.mark_missed()
        assert track.state == TrackState.DELETED

    def test_range_confidence_low(self):
        track = ThermalTrack(_thermal_det())
        assert track.range_confidence < 0.1

    def test_predicted_bbox_is_none(self):
        track = ThermalTrack(_thermal_det())
        assert track.predicted_bbox is None

    def test_score_range(self):
        track = ThermalTrack(_thermal_det())
        assert 0.0 <= track.score <= 1.0

    def test_to_dict(self):
        track = ThermalTrack(_thermal_det())
        d = track.to_dict()
        assert "track_id" in d
        assert "temperature_k" in d
        assert "azimuth_deg" in d


# === ThermalTrackManager ===


@pytest.fixture
def thermal_config():
    return OmegaConf.create(
        {
            "filter": {"dt": 0.033, "assumed_initial_range_m": 10000.0},
            "association": {"gate_threshold": 9.21},
            "track_management": {"confirm_hits": 3, "max_coast_frames": 10, "max_tracks": 50},
        }
    )


class TestThermalTrackManager:
    def test_step_creates_tracks(self, thermal_config):
        mgr = ThermalTrackManager(thermal_config)
        dets = [_thermal_det(azimuth_deg=5.0), _thermal_det(azimuth_deg=20.0)]
        tracks = mgr.step(dets)
        assert len(tracks) == 2

    def test_step_no_detections(self, thermal_config):
        mgr = ThermalTrackManager(thermal_config)
        tracks = mgr.step([])
        assert tracks == []

    def test_track_confirmation(self, thermal_config):
        mgr = ThermalTrackManager(thermal_config)
        det = _thermal_det(azimuth_deg=10.0)
        for _ in range(5):
            mgr.step([det])
        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1

    def test_track_deletion_on_miss(self, thermal_config):
        mgr = ThermalTrackManager(thermal_config)
        mgr.step([_thermal_det()])
        assert mgr.track_count == 1
        # Miss repeatedly
        for _ in range(5):
            mgr.step([])
        assert mgr.track_count == 0

    def test_track_count_property(self, thermal_config):
        mgr = ThermalTrackManager(thermal_config)
        mgr.step([_thermal_det(azimuth_deg=5.0)])
        assert mgr.track_count == 1
