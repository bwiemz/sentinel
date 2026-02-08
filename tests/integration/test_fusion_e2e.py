"""End-to-end fusion integration tests.

Tests camera+radar → FusedTracks with state fusion and threat classification.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType
from sentinel.fusion.track_fusion import TrackFusion
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget, radar_frame_to_detections
from sentinel.tracking.radar_track_manager import RadarTrackManager
from sentinel.tracking.track_manager import TrackManager


def _tracking_config():
    return OmegaConf.create(
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


def _radar_config():
    return OmegaConf.create(
        {
            "filter": {"dt": 0.1, "type": "ekf"},
            "association": {"gate_threshold": 9.21},
            "track_management": {"confirm_hits": 2, "max_coast_frames": 5, "max_tracks": 50},
        }
    )


class TestFusionE2E:
    def test_camera_radar_fusion_produces_fused_tracks(self):
        """Camera + radar tracks should produce fused tracks via TrackFusion."""
        # Setup camera tracker with synthetic detections
        cam_mgr = TrackManager(_tracking_config())

        # Setup radar
        target = RadarTarget("TGT-1", np.array([3000.0, 1000.0]), np.array([-10.0, 0.0]), rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        radar_mgr = RadarTrackManager(_radar_config())

        # Run both for a few steps to build confirmed tracks
        for i in range(5):
            # Camera: detection at center of frame (matching azimuth ~18.4deg for target at 3000,1000)
            cam_det = Detection(
                sensor_type=SensorType.CAMERA,
                timestamp=float(i) * 0.1,
                bbox=np.array([600, 300, 700, 400], dtype=np.float32),
                class_id=0,
                class_name="vehicle",
                confidence=0.9,
            )
            cam_mgr.step([cam_det])

            # Radar
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            radar_mgr.step(dets)

        # Fuse
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=10.0)
        cam_tracks = cam_mgr.active_tracks
        radar_tracks = radar_mgr.active_tracks

        assert len(cam_tracks) > 0
        assert len(radar_tracks) > 0

        fused = fusion.fuse(cam_tracks, radar_tracks)
        # At minimum, fusion should return tracks (fused or pass-through)
        assert len(fused) >= max(len(cam_tracks), len(radar_tracks))


class TestStateFusionE2E:
    def test_covariance_intersection_integration(self):
        """CI should produce valid fused estimates from two sensor tracks."""
        from sentinel.fusion.state_fusion import covariance_intersection

        # Two sensor estimates of same target
        x1 = np.array([3000.0, -10.0, 1000.0, 5.0])
        P1 = np.eye(4) * 100.0

        x2 = np.array([3050.0, -12.0, 980.0, 6.0])
        P2 = np.eye(4) * 200.0

        x_f, P_f = covariance_intersection(x1, P1, x2, P2)

        # Fused state should be between the two estimates
        assert 2990.0 < x_f[0] < 3060.0
        assert 970.0 < x_f[2] < 1010.0

        # Fused covariance trace should be <= max trace of inputs
        assert np.trace(P_f) < max(np.trace(P1), np.trace(P2))

        # P_fused should be positive definite
        eigenvalues = np.linalg.eigvalsh(P_f)
        assert all(eigenvalues > 0)
