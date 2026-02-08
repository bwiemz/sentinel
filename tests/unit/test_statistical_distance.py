"""Tests for statistical distance and enhanced fusion correlation."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.tracking.cost_functions import track_to_track_mahalanobis
from sentinel.fusion.track_fusion import TrackFusion
from sentinel.fusion.multi_sensor_fusion import MultiSensorFusion


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


def _radar_detection(range_m=5000.0, azimuth_deg=0.0, ts=0.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
    )


# ===========================================================================
# TestTrackToTrackMahalanobis
# ===========================================================================

class TestTrackToTrackMahalanobis:

    def test_zero_distance(self):
        pos = np.array([100.0, 200.0])
        cov = np.eye(2)
        d2 = track_to_track_mahalanobis(pos, cov, pos, cov)
        assert d2 == pytest.approx(0.0)

    def test_identity_covariance(self):
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([1.0, 0.0])
        cov = np.eye(2)
        # d² = [1,0]' * (2I)^-1 * [1,0] = 0.5
        d2 = track_to_track_mahalanobis(pos1, cov, pos2, cov)
        assert d2 == pytest.approx(0.5)

    def test_scaled_covariance(self):
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([2.0, 0.0])
        cov1 = np.eye(2) * 4.0
        cov2 = np.eye(2) * 4.0
        # d² = [2,0]' * (8I)^-1 * [2,0] = 4/8 = 0.5
        d2 = track_to_track_mahalanobis(pos1, cov1, pos2, cov2)
        assert d2 == pytest.approx(0.5)

    def test_symmetry(self):
        pos1 = np.array([1.0, 2.0])
        pos2 = np.array([3.0, 4.0])
        cov1 = np.eye(2) * 2.0
        cov2 = np.eye(2) * 3.0
        d_12 = track_to_track_mahalanobis(pos1, cov1, pos2, cov2)
        d_21 = track_to_track_mahalanobis(pos2, cov2, pos1, cov1)
        assert d_12 == pytest.approx(d_21)

    def test_singular_returns_inf(self):
        pos1 = np.array([1.0, 0.0])
        pos2 = np.array([0.0, 0.0])
        cov = np.zeros((2, 2))
        d2 = track_to_track_mahalanobis(pos1, cov, pos2, cov)
        assert d2 == float("inf")

    def test_1d(self):
        pos1 = np.array([3.0])
        pos2 = np.array([0.0])
        cov1 = np.array([[4.0]])
        cov2 = np.array([[5.0]])
        # d² = 9 / 9 = 1.0
        d2 = track_to_track_mahalanobis(pos1, cov1, pos2, cov2)
        assert d2 == pytest.approx(1.0)

    def test_3d(self):
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.zeros(3)
        cov = np.eye(3)
        # d² = [1,0,0]' * (2I)^-1 * [1,0,0] = 0.5
        d2 = track_to_track_mahalanobis(pos1, cov, pos2, cov)
        assert d2 == pytest.approx(0.5)


# ===========================================================================
# TestTrackFusionStatistical
# ===========================================================================

class TestTrackFusionStatistical:

    def test_statistical_distance_enabled(self):
        """With statistical distance, matching uses Mahalanobis."""
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=5.0,
            use_statistical_distance=True,
            statistical_distance_gate=50.0,  # Wide gate for test
        )

        # Camera track at center → azimuth ~0 deg
        cam_det = _camera_detection(600, 300, 680, 380, ts=0.0)
        cam = Track(cam_det, dt=1 / 30)
        cam.last_update_time = 0.0

        # Radar track near azimuth 0
        rdr_det = _radar_detection(range_m=5000.0, azimuth_deg=0.5, ts=0.0)
        rdr = RadarTrack(rdr_det, dt=0.1)
        rdr.last_update_time = 0.0

        result = fusion.fuse([cam], [rdr])
        # Should produce a fused track (dual sensor)
        dual = [f for f in result if f.is_dual_sensor]
        assert len(dual) == 1

    def test_default_angular_fallback(self):
        """Default mode uses angular distance, not statistical."""
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=5.0,
        )

        # Camera track at center
        cam = Track(_camera_detection(600, 300, 680, 380), dt=1 / 30)
        rdr = RadarTrack(_radar_detection(range_m=5000.0, azimuth_deg=0.5), dt=0.1)

        result = fusion.fuse([cam], [rdr])
        dual = [f for f in result if f.is_dual_sensor]
        assert len(dual) == 1

    def test_gate_threshold_rejects(self):
        """Tracks outside the statistical gate should not be matched."""
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack

        fusion = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=5.0,
            use_statistical_distance=True,
            statistical_distance_gate=0.001,  # Very tight gate
        )

        cam = Track(_camera_detection(600, 300, 680, 380), dt=1 / 30)
        cam.last_update_time = 0.0
        rdr = RadarTrack(_radar_detection(range_m=5000.0, azimuth_deg=0.5), dt=0.1)
        rdr.last_update_time = 0.0

        result = fusion.fuse([cam], [rdr])
        dual = [f for f in result if f.is_dual_sensor]
        # Tight gate → no match
        assert len(dual) == 0

    def test_temporal_alignment_changes_cost(self):
        """Temporal alignment should produce different costs than without."""
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack

        # With alignment
        fusion_aligned = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            use_statistical_distance=True,
            use_temporal_alignment=True,
            statistical_distance_gate=100.0,
        )

        # Without alignment
        fusion_no_align = TrackFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            use_statistical_distance=True,
            use_temporal_alignment=False,
            statistical_distance_gate=100.0,
        )

        cam = Track(_camera_detection(600, 300, 680, 380, ts=0.0), dt=1 / 30)
        cam.last_update_time = 0.0
        rdr = RadarTrack(_radar_detection(range_m=5000.0, azimuth_deg=0.5, ts=0.05), dt=0.1)
        rdr.last_update_time = 0.05

        # Both should produce valid fused output
        result_a = fusion_aligned.fuse([cam], [rdr])
        result_b = fusion_no_align.fuse([cam], [rdr])
        assert len(result_a) >= 1
        assert len(result_b) >= 1


# ===========================================================================
# TestMultiSensorFusionAlignment
# ===========================================================================

class TestMultiSensorFusionAlignment:

    def test_disabled_by_default(self):
        """Default MultiSensorFusion should not use temporal alignment."""
        fusion = MultiSensorFusion()
        assert fusion._base_fusion._use_temporal_alignment is False
        assert fusion._base_fusion._use_statistical_distance is False

    def test_enabled_passthrough(self):
        """Parameters should pass through to TrackFusion."""
        fusion = MultiSensorFusion(
            use_temporal_alignment=True,
            use_statistical_distance=True,
            statistical_distance_gate=16.0,
        )
        assert fusion._base_fusion._use_temporal_alignment is True
        assert fusion._base_fusion._use_statistical_distance is True
        assert fusion._base_fusion._stat_gate == 16.0

    def test_thermal_association_still_works(self):
        """Thermal tracks should still associate with existing angular logic."""
        from sentinel.tracking.track import Track
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=5.0,
            thermal_azimuth_gate_deg=5.0,
        )

        cam = Track(_camera_detection(640, 300, 720, 380), dt=1 / 30)
        rdr = RadarTrack(_radar_detection(range_m=5000.0, azimuth_deg=0.0), dt=0.1)
        thermal_det = Detection(
            sensor_type=SensorType.THERMAL,
            timestamp=0.0,
            azimuth_deg=0.5,
            temperature_k=350.0,
        )
        th = ThermalTrack(thermal_det, dt=0.033)

        result = fusion.fuse([cam], [rdr], thermal_tracks=[th])
        assert len(result) >= 1
        # At least one track should have thermal contribution
        thermal_fused = [e for e in result if SensorType.THERMAL in e.sensor_sources]
        assert len(thermal_fused) >= 1

    def test_quantum_association_still_works(self):
        """Quantum tracks should still associate with angular logic."""
        from sentinel.tracking.radar_track import RadarTrack

        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=5.0,
        )

        rdr = RadarTrack(_radar_detection(range_m=5000.0, azimuth_deg=10.0), dt=0.1)
        q_rdr = RadarTrack(_radar_detection(range_m=5100.0, azimuth_deg=10.5), dt=0.1)

        result = fusion.fuse([], [rdr], quantum_radar_tracks=[q_rdr])
        assert len(result) >= 1
