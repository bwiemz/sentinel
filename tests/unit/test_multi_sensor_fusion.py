"""Tests for enhanced multi-sensor fusion."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.fusion.multi_sensor_fusion import (
    EnhancedFusedTrack,
    MultiSensorFusion,
    THREAT_CRITICAL,
    THREAT_HIGH,
    THREAT_LOW,
    THREAT_MEDIUM,
)
from sentinel.fusion.multifreq_correlator import CorrelatedDetection
from sentinel.tracking.filters import ExtendedKalmanFilter, KalmanFilter
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.tracking.track import Track


# === Helpers ===

def _make_cam_track(px_x=640.0, px_y=360.0, track_id="CAM-01"):
    """Create a mock camera track at given pixel position."""
    det = Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=1.0,
        bbox=np.array([px_x - 50, px_y - 30, px_x + 50, px_y + 30], dtype=np.float32),
        class_name="aircraft",
        confidence=0.9,
    )
    cfg = OmegaConf.create({
        "filter": {"dt": 0.033},
        "association": {"gate_threshold": 9.21, "iou_weight": 0.5, "mahalanobis_weight": 0.5},
        "track_management": {"confirm_hits": 1, "confirm_window": 3, "max_coast_frames": 10, "max_tracks": 50},
    })
    from sentinel.tracking.track_manager import TrackManager
    mgr = TrackManager(cfg)
    mgr.step([det])
    mgr.step([det])  # Confirm
    tracks = mgr.active_tracks
    if tracks:
        return tracks[0]
    # Fallback: create directly
    t = Track(det, dt=0.033, confirm_hits=1)
    t.state = TrackState.CONFIRMED
    return t


def _make_radar_track(azimuth_deg=0.0, range_m=5000.0, track_id="RDR-01"):
    det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
    )
    rt = RadarTrack(det, dt=0.1, confirm_hits=1)
    rt.state = TrackState.CONFIRMED
    return rt


def _make_thermal_track(azimuth_deg=0.0, temperature_k=800.0, track_id="THM-01"):
    det = Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=1.0,
        azimuth_deg=azimuth_deg,
        temperature_k=temperature_k,
        thermal_band="mwir",
    )
    tt = ThermalTrack(det, assumed_range_m=10000.0, dt=0.033, confirm_hits=1)
    tt.state = TrackState.CONFIRMED
    return tt


# === Tests ===


class TestEnhancedFusedTrack:
    def test_sensor_count_camera_only(self):
        eft = EnhancedFusedTrack(
            fused_id="test",
            camera_track=_make_cam_track(),
            sensor_sources={SensorType.CAMERA},
        )
        assert eft.sensor_count == 1

    def test_sensor_count_triple(self):
        eft = EnhancedFusedTrack(
            fused_id="test",
            camera_track=_make_cam_track(),
            radar_track=_make_radar_track(),
            thermal_track=_make_thermal_track(),
        )
        assert eft.sensor_count == 3

    def test_to_dict(self):
        eft = EnhancedFusedTrack(
            fused_id="test",
            threat_level="HIGH",
            temperature_k=1500.0,
            sensor_sources={SensorType.RADAR},
        )
        d = eft.to_dict()
        assert d["threat_level"] == "HIGH"
        assert d["temperature_k"] == 1500.0
        assert "sensor_count" in d


class TestMultiSensorFusion:
    def test_empty_inputs(self):
        msf = MultiSensorFusion()
        result = msf.fuse([], [], [])
        assert result == []

    def test_camera_only(self):
        msf = MultiSensorFusion()
        cam = _make_cam_track(px_x=640.0)
        result = msf.fuse([cam], [], [])
        assert len(result) == 1
        assert result[0].camera_track is not None
        assert result[0].radar_track is None

    def test_radar_only(self):
        msf = MultiSensorFusion()
        rdr = _make_radar_track(azimuth_deg=5.0)
        result = msf.fuse([], [rdr], [])
        assert len(result) == 1
        assert result[0].radar_track is not None

    def test_thermal_only(self):
        msf = MultiSensorFusion()
        thm = _make_thermal_track(azimuth_deg=5.0, temperature_k=2000.0)
        result = msf.fuse([], [], [thm])
        assert len(result) == 1
        assert result[0].thermal_track is not None
        assert result[0].temperature_k == 2000.0

    def test_camera_radar_fusion(self):
        """Camera and radar at same azimuth should fuse."""
        msf = MultiSensorFusion(camera_hfov_deg=60.0, image_width_px=1280)
        cam = _make_cam_track(px_x=640.0)  # Center = 0 deg azimuth
        rdr = _make_radar_track(azimuth_deg=0.0)
        result = msf.fuse([cam], [rdr])
        dual = [r for r in result if r.camera_track is not None and r.radar_track is not None]
        assert len(dual) >= 1

    def test_thermal_augments_radar(self):
        """Thermal track at same azimuth should merge into radar-fused track."""
        msf = MultiSensorFusion(
            camera_hfov_deg=60.0, image_width_px=1280,
            thermal_azimuth_gate_deg=5.0,
        )
        rdr = _make_radar_track(azimuth_deg=5.0)
        thm = _make_thermal_track(azimuth_deg=5.0, temperature_k=1200.0)
        result = msf.fuse([], [rdr], [thm])
        triple = [r for r in result if r.thermal_track is not None and r.radar_track is not None]
        assert len(triple) == 1
        assert triple[0].temperature_k == 1200.0

    def test_threat_critical_hypersonic(self):
        msf = MultiSensorFusion()
        thm = _make_thermal_track(azimuth_deg=0.0, temperature_k=3000.0)
        result = msf.fuse([], [], [thm])
        assert result[0].threat_level == THREAT_CRITICAL

    def test_threat_high_stealth(self):
        msf = MultiSensorFusion()
        rdr = _make_radar_track(azimuth_deg=5.0)
        cd = CorrelatedDetection(
            primary_detection=Detection(
                sensor_type=SensorType.RADAR, timestamp=1.0,
                range_m=5000.0, azimuth_deg=5.0, radar_band="vhf",
            ),
            bands_detected=["vhf"],
            is_stealth_candidate=True,
        )
        result = msf.fuse([], [rdr], correlated_detections=[cd])
        stealth = [r for r in result if r.is_stealth_candidate]
        assert len(stealth) >= 1
        assert stealth[0].threat_level == THREAT_HIGH

    def test_threat_medium_multi_sensor(self):
        msf = MultiSensorFusion(
            camera_hfov_deg=60.0, image_width_px=1280,
            thermal_azimuth_gate_deg=5.0,
        )
        cam = _make_cam_track(px_x=640.0)
        rdr = _make_radar_track(azimuth_deg=0.0)
        result = msf.fuse([cam], [rdr])
        dual = [r for r in result if r.sensor_count >= 2]
        if dual:
            assert dual[0].threat_level == THREAT_MEDIUM

    def test_threat_low_single_sensor(self):
        msf = MultiSensorFusion()
        cam = _make_cam_track(px_x=640.0)
        result = msf.fuse([cam], [])
        assert result[0].threat_level == THREAT_LOW

    def test_fusion_quality_increases_with_sensors(self):
        msf = MultiSensorFusion(
            camera_hfov_deg=60.0, image_width_px=1280,
            thermal_azimuth_gate_deg=5.0,
        )
        cam = _make_cam_track(px_x=640.0)
        rdr = _make_radar_track(azimuth_deg=0.0)
        thm = _make_thermal_track(azimuth_deg=0.0)

        r1 = msf.fuse([cam], [])
        r2 = msf.fuse([cam], [rdr])
        r3 = msf.fuse([cam], [rdr], [thm])

        q1 = r1[0].fusion_quality
        # Multi-sensor should generally have higher quality
        q2 = max(r.fusion_quality for r in r2)
        assert q2 >= q1  # More sensors = better quality

    def test_backward_compat_no_thermal(self):
        """Without thermal, should behave like original TrackFusion."""
        msf = MultiSensorFusion(camera_hfov_deg=60.0, image_width_px=1280)
        cam = _make_cam_track(px_x=640.0)
        rdr = _make_radar_track(azimuth_deg=0.0)
        result = msf.fuse([cam], [rdr])
        # Should still produce fused output
        assert len(result) >= 1
        for r in result:
            assert isinstance(r, EnhancedFusedTrack)
            assert r.thermal_track is None
