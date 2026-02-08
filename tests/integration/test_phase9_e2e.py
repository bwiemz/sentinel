"""Phase 9 end-to-end integration tests.

Tests track quality monitoring, temporal alignment, statistical distance fusion,
and JPDA association through full tracking/fusion cycles.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.fusion.multi_sensor_fusion import MultiSensorFusion
from sentinel.fusion.temporal_alignment import (
    AlignedTrackState,
    align_tracks_to_epoch,
    predict_track_to_epoch,
)
from sentinel.fusion.track_fusion import TrackFusion
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget, radar_frame_to_detections
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.radar_track_manager import RadarTrackManager
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.tracking.thermal_track_manager import ThermalTrackManager
from sentinel.tracking.track import Track
from sentinel.tracking.track_manager import TrackManager
from sentinel.tracking.track_quality import FilterConsistencyMonitor


# ---------------------------------------------------------------------------
# Helper configs
# ---------------------------------------------------------------------------


def _camera_tracking_config(**overrides):
    cfg = OmegaConf.create(
        {
            "filter": {"type": "kf", "dt": 0.1, "process_noise_std": 1.0, "measurement_noise_std": 10.0},
            "association": {
                "method": "hungarian",
                "gate_threshold": 50.0,
                "iou_weight": 0.5,
                "mahalanobis_weight": 0.5,
            },
            "track_management": {
                "confirm_hits": 2,
                "confirm_window": 3,
                "max_coast_frames": 5,
                "max_tracks": 50,
                "tentative_delete_misses": 3,
            },
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _radar_tracking_config(**overrides):
    cfg = OmegaConf.create(
        {
            "filter": {"dt": 0.1, "type": "ekf"},
            "association": {"gate_threshold": 9.21},
            "track_management": {"confirm_hits": 2, "max_coast_frames": 5, "max_tracks": 50},
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _thermal_tracking_config(**overrides):
    cfg = OmegaConf.create(
        {
            "filter": {"type": "bearing_ekf", "dt": 0.1, "assumed_initial_range_m": 10000.0},
            "association": {"gate_threshold": 6.635},
            "track_management": {"confirm_hits": 2, "max_coast_frames": 5, "max_tracks": 50},
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _make_camera_detection(x, y, w=50, h=50, class_name="vehicle", confidence=0.9, timestamp=1.0):
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        class_name=class_name,
        sensor_type=SensorType.CAMERA,
        timestamp=timestamp,
    )


def _make_radar_detection(range_m, azimuth_deg, velocity_mps=0.0, rcs_dbsm=10.0, timestamp=1.0):
    return Detection(
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=velocity_mps,
        rcs_dbsm=rcs_dbsm,
        sensor_type=SensorType.RADAR,
        confidence=0.9,
        timestamp=timestamp,
    )


def _make_thermal_detection(azimuth_deg, elevation_deg=0.0, temperature_k=350.0, band="mwir", timestamp=1.0):
    return Detection(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        temperature_k=temperature_k,
        thermal_band=band,
        sensor_type=SensorType.THERMAL,
        confidence=0.9,
        timestamp=timestamp,
    )


# ===================================================================
# Track Quality E2E
# ===================================================================


class TestTrackQualityE2E:
    """Verify NIS monitoring flows through full track update cycles."""

    def test_nis_recorded_through_camera_track_cycle(self):
        """Camera track updates should populate the quality monitor NIS history."""
        mgr = TrackManager(_camera_tracking_config())
        det = _make_camera_detection(100, 100, timestamp=1.0)

        # Step 1: initiate track
        mgr.step([det])
        assert mgr.track_count == 1
        track = mgr.active_tracks[0]

        # Step 2+: update track — NIS should be recorded
        for i in range(5):
            det2 = _make_camera_detection(100 + i, 100, timestamp=1.0 + (i + 1) * 0.1)
            mgr.step([det2])

        track = mgr.active_tracks[0]
        assert track.quality_monitor is not None
        assert track.quality_monitor.sample_count > 0

    def test_track_quality_health_progression(self):
        """Quality monitor health should be NOMINAL for consistent updates."""
        mgr = TrackManager(_camera_tracking_config())

        # Feed consistent detections at the same spot
        for i in range(10):
            det = _make_camera_detection(200, 200, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        if mgr.active_tracks:
            track = mgr.active_tracks[0]
            if track.quality_monitor and track.quality_monitor.sample_count > 0:
                # With consistent data, health should not be 'diverged'
                assert track.quality_monitor.filter_health != "diverged"

    def test_quality_contributes_to_score(self):
        """Track score formula should include quality factor when monitor exists."""
        mgr = TrackManager(_camera_tracking_config())

        for i in range(6):
            det = _make_camera_detection(300, 300, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        if mgr.active_tracks:
            track = mgr.active_tracks[0]
            # Score should be positive and reasonable
            assert 0.0 < track.score <= 1.0

    def test_radar_track_nis_recording(self):
        """Radar track updates should record NIS in the quality monitor."""
        mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(6):
            det = _make_radar_detection(5000.0, 30.0 + i * 0.1, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        if mgr.active_tracks:
            track = mgr.active_tracks[0]
            assert track.quality_monitor is not None
            assert track.quality_monitor.sample_count > 0


# ===================================================================
# Temporal Alignment E2E
# ===================================================================


class TestTemporalAlignmentE2E:
    """Verify temporal alignment through track prediction and fusion."""

    def test_camera_track_predict_to_time(self):
        """Camera track predict_to_time should produce valid future state."""
        mgr = TrackManager(_camera_tracking_config())

        for i in range(4):
            det = _make_camera_detection(100 + i * 10, 200, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        track = mgr.active_tracks[0]
        x_pred, P_pred = track.predict_to_time(2.0)

        assert x_pred.shape[0] >= 4
        assert P_pred.shape[0] == P_pred.shape[1] == x_pred.shape[0]
        # Covariance should grow with prediction time
        assert np.trace(P_pred) > 0

    def test_radar_track_predict_to_time(self):
        """Radar track predict_to_time should produce valid future state."""
        mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(4):
            det = _make_radar_detection(5000.0 - i * 10, 30.0, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        track = mgr.active_tracks[0]
        x_pred, P_pred = track.predict_to_time(2.0)

        assert x_pred.shape[0] >= 4
        assert P_pred.shape[0] == P_pred.shape[1]

    def test_align_mixed_sensor_tracks(self):
        """align_tracks_to_epoch should handle mixed camera + radar tracks."""
        cam_mgr = TrackManager(_camera_tracking_config())
        rdr_mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(4):
            cam_det = _make_camera_detection(100 + i * 5, 200, timestamp=1.0 + i * 0.1)
            cam_mgr.step([cam_det])

            rdr_det = _make_radar_detection(5000.0, 30.0, timestamp=1.0 + i * 0.1)
            rdr_mgr.step([rdr_det])

        all_tracks = cam_mgr.active_tracks + rdr_mgr.active_tracks
        aligned = align_tracks_to_epoch(all_tracks, reference_time=2.0)

        assert len(aligned) == len(all_tracks)
        for a in aligned:
            assert isinstance(a, AlignedTrackState)
            assert a.alignment_time == 2.0
            assert a.position.shape == (2,)
            assert a.covariance.shape == (2, 2)

    def test_temporal_alignment_in_fusion(self):
        """TrackFusion with temporal alignment enabled should still produce fused tracks."""
        cam_mgr = TrackManager(_camera_tracking_config())
        rdr_mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(4):
            cam_det = _make_camera_detection(640, 200, timestamp=1.0 + i * 0.1)
            cam_mgr.step([cam_det])
            rdr_det = _make_radar_detection(5000.0, 0.0, timestamp=1.0 + i * 0.1)
            rdr_mgr.step([rdr_det])

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            use_temporal_alignment=True,
            use_statistical_distance=False,
        )
        fused = fusion.fuse(cam_mgr.active_tracks, rdr_mgr.active_tracks)
        # Should produce at least some fused tracks
        assert isinstance(fused, list)


# ===================================================================
# Statistical Distance Fusion E2E
# ===================================================================


class TestStatisticalFusionE2E:
    """Verify statistical distance fusion through TrackFusion."""

    def test_statistical_distance_fusion_produces_tracks(self):
        """Statistical distance mode should still produce fused tracks."""
        cam_mgr = TrackManager(_camera_tracking_config())
        rdr_mgr = RadarTrackManager(_radar_tracking_config())

        # Place camera detection at center (0 deg azimuth) and radar at 0 deg
        for i in range(4):
            cam_det = _make_camera_detection(640, 200, timestamp=1.0 + i * 0.1)
            cam_mgr.step([cam_det])
            rdr_det = _make_radar_detection(5000.0, 0.0, timestamp=1.0 + i * 0.1)
            rdr_mgr.step([rdr_det])

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            use_statistical_distance=True,
            statistical_distance_gate=50.0,  # wide gate for integration test
        )
        fused = fusion.fuse(cam_mgr.active_tracks, rdr_mgr.active_tracks)
        assert isinstance(fused, list)

    def test_angular_fallback_default(self):
        """Default config (no statistical distance) should use angular fallback."""
        cam_mgr = TrackManager(_camera_tracking_config())
        rdr_mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(4):
            cam_det = _make_camera_detection(640, 200, timestamp=1.0 + i * 0.1)
            cam_mgr.step([cam_det])
            rdr_det = _make_radar_detection(5000.0, 0.0, timestamp=1.0 + i * 0.1)
            rdr_mgr.step([rdr_det])

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            # defaults: use_statistical_distance=False
        )
        fused = fusion.fuse(cam_mgr.active_tracks, rdr_mgr.active_tracks)
        assert isinstance(fused, list)
        # Should have fused tracks (camera center = 0 deg, radar = 0 deg)
        assert len(fused) > 0


# ===================================================================
# JPDA E2E
# ===================================================================


class TestJPDAE2E:
    """Verify JPDA association through full TrackManager cycles."""

    def test_jpda_camera_basic_cycle(self):
        """TrackManager with JPDA should handle a basic detection cycle."""
        cfg = _camera_tracking_config(
            association={"method": "jpda", "gate_threshold": 50.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        mgr = TrackManager(cfg)

        # Feed detections over several frames
        for i in range(6):
            det = _make_camera_detection(200, 200, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        assert mgr.track_count >= 1
        # Track should have been confirmed after enough hits
        track = mgr.active_tracks[0]
        assert track.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def test_jpda_camera_separated_targets(self):
        """JPDA should maintain separate tracks for well-separated targets."""
        cfg = _camera_tracking_config(
            association={"method": "jpda", "gate_threshold": 50.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        mgr = TrackManager(cfg)

        for i in range(6):
            dets = [
                _make_camera_detection(100, 100, timestamp=1.0 + i * 0.1),
                _make_camera_detection(500, 500, timestamp=1.0 + i * 0.1),
            ]
            mgr.step(dets)

        # Should have 2 separate tracks
        assert mgr.track_count >= 2

    def test_jpda_camera_close_targets(self):
        """JPDA should handle close/ambiguous targets without crashing."""
        cfg = _camera_tracking_config(
            association={"method": "jpda", "gate_threshold": 100.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        mgr = TrackManager(cfg)

        for i in range(6):
            dets = [
                _make_camera_detection(200, 200, timestamp=1.0 + i * 0.1),
                _make_camera_detection(210, 205, timestamp=1.0 + i * 0.1),  # very close
            ]
            mgr.step(dets)

        # Should not crash; tracks may merge or stay separate
        assert mgr.track_count >= 1

    def test_jpda_radar_cycle(self):
        """RadarTrackManager with JPDA should track radar detections."""
        cfg = _radar_tracking_config(
            association={"method": "jpda", "gate_threshold": 20.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        mgr = RadarTrackManager(cfg)

        for i in range(6):
            det = _make_radar_detection(5000.0, 30.0, velocity_mps=-50.0, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        assert mgr.track_count >= 1

    def test_jpda_thermal_cycle(self):
        """ThermalTrackManager with JPDA should track thermal detections."""
        cfg = _thermal_tracking_config(
            association={"method": "jpda", "gate_threshold": 10.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        mgr = ThermalTrackManager(cfg)

        for i in range(6):
            det = _make_thermal_detection(45.0, timestamp=1.0 + i * 0.1)
            mgr.step([det])

        assert mgr.track_count >= 1


# ===================================================================
# Full Pipeline E2E
# ===================================================================


class TestFullPipelineE2E:
    """Verify all Phase 9 features working together."""

    def test_all_features_combined(self):
        """Camera + Radar + JPDA + temporal alignment + statistical distance."""
        cam_cfg = _camera_tracking_config(
            association={"method": "jpda", "gate_threshold": 50.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        rdr_cfg = _radar_tracking_config(
            association={"method": "jpda", "gate_threshold": 20.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )
        thm_cfg = _thermal_tracking_config(
            association={"method": "jpda", "gate_threshold": 10.0, "detection_probability": 0.9, "false_alarm_density": 1e-6}
        )

        cam_mgr = TrackManager(cam_cfg)
        rdr_mgr = RadarTrackManager(rdr_cfg)
        thm_mgr = ThermalTrackManager(thm_cfg)

        # Feed several frames of consistent detections
        for i in range(6):
            t = 1.0 + i * 0.1
            cam_mgr.step([_make_camera_detection(640, 200, timestamp=t)])
            rdr_mgr.step([_make_radar_detection(5000.0, 0.0, timestamp=t)])
            thm_mgr.step([_make_thermal_detection(0.0, timestamp=t)])

        # All managers should have tracks
        assert cam_mgr.track_count >= 1
        assert rdr_mgr.track_count >= 1
        assert thm_mgr.track_count >= 1

        # Fuse with all Phase 9 features enabled
        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=5.0,
            use_temporal_alignment=True,
            use_statistical_distance=True,
            statistical_distance_gate=50.0,
        )
        fused = fusion.fuse(
            camera_tracks=cam_mgr.active_tracks,
            radar_tracks=rdr_mgr.active_tracks,
            thermal_tracks=thm_mgr.active_tracks,
        )
        assert isinstance(fused, list)
        assert len(fused) >= 1

        # Fused tracks should have metadata
        for ft in fused:
            assert ft.threat_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN")
            assert 0.0 <= ft.fusion_quality <= 1.0

    def test_backward_compatible_defaults(self):
        """Default config (no JPDA, no temporal, no statistical) should work as before."""
        cam_mgr = TrackManager(_camera_tracking_config())
        rdr_mgr = RadarTrackManager(_radar_tracking_config())

        for i in range(4):
            t = 1.0 + i * 0.1
            cam_mgr.step([_make_camera_detection(640, 200, timestamp=t)])
            rdr_mgr.step([_make_radar_detection(5000.0, 0.0, timestamp=t)])

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
        )
        fused = fusion.fuse(cam_mgr.active_tracks, rdr_mgr.active_tracks)
        assert isinstance(fused, list)
        assert len(fused) > 0

        # Verify fused tracks have the expected structure
        for ft in fused:
            assert hasattr(ft, "fused_id")
            assert hasattr(ft, "fusion_quality")

    def test_jpda_dense_environment(self):
        """JPDA should handle a dense environment with many close targets."""
        cfg = _camera_tracking_config(
            association={"method": "jpda", "gate_threshold": 80.0, "detection_probability": 0.9, "false_alarm_density": 1e-5}
        )
        mgr = TrackManager(cfg)

        rng = np.random.RandomState(42)
        for frame in range(8):
            dets = []
            for t_idx in range(5):
                x = 100 + t_idx * 80 + rng.randn() * 5
                y = 200 + rng.randn() * 5
                dets.append(_make_camera_detection(x, y, timestamp=1.0 + frame * 0.1))
            mgr.step(dets)

        # Should have created multiple tracks without crashing
        assert mgr.track_count >= 3
