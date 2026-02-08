"""Tests for config parameter wiring to tracking components."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType
from sentinel.tracking.track import Track
from sentinel.tracking.track_manager import TrackManager
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.radar_track_manager import RadarTrackManager
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.tracking.thermal_track_manager import ThermalTrackManager


def _camera_det(x1=100, y1=100, x2=200, y2=200, cls="person"):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=0.9,
    )


def _radar_det(range_m=5000.0, azimuth_deg=30.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        confidence=0.9,
    )


def _thermal_det(azimuth_deg=30.0, temperature_k=400.0):
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=0.0,
        azimuth_deg=azimuth_deg,
        temperature_k=temperature_k,
        confidence=0.9,
    )


class TestTrackNoiseWiring:
    """Test that process/measurement noise config reaches the Kalman filter."""

    def test_process_noise_wired(self):
        det = _camera_det()
        track = Track(det, process_noise_std=2.5)
        # Q should be based on sigma_a=2.5 instead of default 5.0
        # The diagonal should reflect the configured value
        assert track.kf.Q is not None
        # With sigma_a=2.5, Q values should be smaller than default sigma_a=5.0
        default_track = Track(det)
        # Q scales with sigma_a^2, so ratio should be (2.5/5.0)^2 = 0.25
        assert track.kf.Q[0, 0] < default_track.kf.Q[0, 0]

    def test_measurement_noise_wired(self):
        det = _camera_det()
        track = Track(det, measurement_noise_std=5.0)
        # R diagonal should be 5.0^2 = 25.0
        np.testing.assert_allclose(track.kf.R[0, 0], 25.0)
        np.testing.assert_allclose(track.kf.R[1, 1], 25.0)


class TestTrackManagerConfigWiring:
    """Test that TrackManager reads and applies config parameters."""

    def _make_config(self, **overrides):
        base = {
            "filter": {
                "dt": 0.033,
                "process_noise_std": 1.0,
                "measurement_noise_std": 10.0,
            },
            "association": {
                "gate_threshold": 9.21,
                "iou_weight": 0.5,
                "mahalanobis_weight": 0.5,
            },
            "track_management": {
                "confirm_hits": 3,
                "confirm_window": 5,
                "max_coast_frames": 15,
                "max_tracks": 100,
                "tentative_delete_misses": 3,
                "confirmed_coast_misses": 5,
                "coast_reconfirm_hits": 2,
            },
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_lifecycle_thresholds_wired(self):
        cfg = self._make_config()
        cfg.track_management.tentative_delete_misses = 2
        cfg.track_management.confirmed_coast_misses = 4
        cfg.track_management.coast_reconfirm_hits = 1
        mgr = TrackManager(cfg)
        assert mgr._tent_delete == 2
        assert mgr._conf_coast == 4
        assert mgr._coast_reconfirm == 1

    def test_confirm_window_wired(self):
        cfg = self._make_config()
        cfg.track_management.confirm_window = 7
        mgr = TrackManager(cfg)
        assert mgr._confirm_window == 7

    def test_noise_params_wired_to_tracks(self):
        cfg = self._make_config()
        cfg.filter.process_noise_std = 2.0
        cfg.filter.measurement_noise_std = 5.0
        mgr = TrackManager(cfg)
        assert mgr._process_noise_std == 2.0
        assert mgr._measurement_noise_std == 5.0

    def test_lifecycle_thresholds_passed_to_tracks(self):
        """Track created by manager should use configured thresholds."""
        cfg = self._make_config()
        cfg.track_management.tentative_delete_misses = 2
        mgr = TrackManager(cfg)

        det = _camera_det()
        tracks = mgr.step([det])
        assert len(tracks) == 1
        track = tracks[0]
        assert track._tent_delete == 2

    def test_m_of_n_passed_to_tracks(self):
        cfg = self._make_config()
        cfg.track_management.confirm_window = 6
        mgr = TrackManager(cfg)

        det = _camera_det()
        tracks = mgr.step([det])
        track = tracks[0]
        assert track._confirm_window == 6
        assert track._hit_window is not None
        assert track._hit_window.maxlen == 6


class TestRadarTrackManagerConfigWiring:
    """Test that RadarTrackManager wires lifecycle thresholds."""

    def _make_config(self, **overrides):
        base = {
            "filter": {"dt": 0.1, "type": "ekf"},
            "association": {"gate_threshold": 9.21},
            "track_management": {
                "confirm_hits": 3,
                "max_coast_frames": 5,
                "max_tracks": 50,
                "confirm_window": None,
                "tentative_delete_misses": 3,
                "confirmed_coast_misses": 5,
                "coast_reconfirm_hits": 2,
            },
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_lifecycle_defaults(self):
        cfg = self._make_config()
        mgr = RadarTrackManager(cfg)
        assert mgr._tent_delete == 3
        assert mgr._conf_coast == 5
        assert mgr._coast_reconfirm == 2

    def test_custom_lifecycle(self):
        cfg = self._make_config()
        cfg.track_management.tentative_delete_misses = 2
        cfg.track_management.coast_reconfirm_hits = 1
        mgr = RadarTrackManager(cfg)
        assert mgr._tent_delete == 2
        assert mgr._coast_reconfirm == 1

    def test_thresholds_passed_to_tracks(self):
        cfg = self._make_config()
        cfg.track_management.tentative_delete_misses = 4
        mgr = RadarTrackManager(cfg)
        det = _radar_det()
        tracks = mgr.step([det])
        assert len(tracks) == 1
        assert tracks[0]._tent_delete == 4


class TestThermalTrackManagerConfigWiring:
    """Test that ThermalTrackManager wires lifecycle thresholds."""

    def _make_config(self):
        return OmegaConf.create({
            "filter": {
                "dt": 0.033,
                "assumed_initial_range_m": 10000.0,
            },
            "association": {"gate_threshold": 6.635},
            "track_management": {
                "confirm_hits": 3,
                "max_coast_frames": 10,
                "max_tracks": 50,
                "confirm_window": None,
                "tentative_delete_misses": 3,
                "confirmed_coast_misses": 5,
                "coast_reconfirm_hits": 2,
            },
        })

    def test_lifecycle_wired(self):
        cfg = self._make_config()
        cfg.track_management.tentative_delete_misses = 2
        mgr = ThermalTrackManager(cfg)
        assert mgr._tent_delete == 2

    def test_thresholds_passed_to_tracks(self):
        cfg = self._make_config()
        cfg.track_management.coast_reconfirm_hits = 3
        mgr = ThermalTrackManager(cfg)
        det = _thermal_det()
        tracks = mgr.step([det])
        assert len(tracks) == 1
        assert tracks[0]._coast_reconfirm == 3

    def test_gate_threshold_default_1dof(self):
        """Thermal gate should default to chi2(1 DOF) = 6.635."""
        cfg = self._make_config()
        # Remove gate_threshold to test default
        del cfg.association.gate_threshold
        mgr = ThermalTrackManager(cfg)
        assert mgr._associator._gate == pytest.approx(6.635, abs=0.01)


class TestMultiSensorFusionConfigWiring:
    """Test that MultiSensorFusion accepts config parameters."""

    def test_constructor_params(self):
        from sentinel.fusion.multi_sensor_fusion import MultiSensorFusion

        fusion = MultiSensorFusion(
            min_fusion_quality=0.4,
            hypersonic_temp_threshold_k=2000.0,
            stealth_rcs_variation_db=20.0,
        )
        assert fusion._min_fusion_quality == 0.4
        assert fusion._hypersonic_temp_k == 2000.0
        assert fusion._stealth_rcs_var_db == 20.0

    def test_multifreq_correlator_stealth_param(self):
        from sentinel.fusion.multifreq_correlator import MultiFreqCorrelator

        corr = MultiFreqCorrelator(stealth_rcs_variation_db=25.0)
        assert corr._stealth_rcs_variation_db == 25.0
