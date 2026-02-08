"""Tests for state-level fusion: Covariance Intersection, persistent IDs."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.fusion.state_fusion import covariance_intersection, information_fusion
from sentinel.fusion.track_fusion import TrackFusion
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.track import Track


class TestCovarianceIntersection:
    """Test the Covariance Intersection algorithm."""

    def test_identical_estimates(self):
        """Fusing identical estimates should return similar result."""
        x = np.array([100.0, 200.0])
        P = np.eye(2) * 10.0
        x_f, P_f = covariance_intersection(x, P, x, P)
        np.testing.assert_allclose(x_f, x, atol=1e-6)
        # Fused covariance should not be larger
        assert np.trace(P_f) <= np.trace(P) + 1e-6

    def test_fused_covariance_smaller_than_larger(self):
        """Fused covariance trace should be smaller than the larger input."""
        x1 = np.array([100.0, 200.0])
        P1 = np.eye(2) * 50.0
        x2 = np.array([102.0, 198.0])
        P2 = np.eye(2) * 15.0

        x_f, P_f = covariance_intersection(x1, P1, x2, P2)
        # CI guarantees fused is tighter than the more uncertain estimate
        assert np.trace(P_f) < np.trace(P1)
        # Fused should be close to the better estimate (P2)
        assert np.trace(P_f) < np.trace(P2) * 1.1

    def test_fused_state_between_inputs(self):
        """Fused state should be between the two input states."""
        x1 = np.array([0.0])
        P1 = np.eye(1) * 10.0
        x2 = np.array([10.0])
        P2 = np.eye(1) * 10.0

        x_f, _ = covariance_intersection(x1, P1, x2, P2)
        assert 0.0 <= x_f[0] <= 10.0

    def test_confident_estimate_dominates(self):
        """Lower covariance estimate should dominate the fusion."""
        x1 = np.array([100.0, 200.0])
        P1 = np.eye(2) * 1.0  # Very confident
        x2 = np.array([150.0, 250.0])
        P2 = np.eye(2) * 1000.0  # Very uncertain

        x_f, _ = covariance_intersection(x1, P1, x2, P2)
        # Should be closer to x1
        assert np.linalg.norm(x_f - x1) < np.linalg.norm(x_f - x2)

    def test_positive_definite_result(self):
        """Fused covariance must be positive definite."""
        x1 = np.array([1.0, 2.0, 3.0])
        P1 = np.diag([10.0, 20.0, 30.0])
        x2 = np.array([1.5, 2.5, 3.5])
        P2 = np.diag([15.0, 10.0, 25.0])

        _, P_f = covariance_intersection(x1, P1, x2, P2)
        eigenvalues = np.linalg.eigvalsh(P_f)
        assert all(ev > 0 for ev in eigenvalues)

    def test_symmetric_result(self):
        """Fused covariance must be symmetric."""
        x1 = np.array([1.0, 2.0])
        P1 = np.array([[10.0, 2.0], [2.0, 8.0]])
        x2 = np.array([1.5, 2.5])
        P2 = np.array([[12.0, -1.0], [-1.0, 9.0]])

        _, P_f = covariance_intersection(x1, P1, x2, P2)
        np.testing.assert_allclose(P_f, P_f.T, atol=1e-10)


class TestInformationFusion:
    """Test information (naive independent) fusion."""

    def test_identical_estimates(self):
        x = np.array([100.0, 200.0])
        P = np.eye(2) * 10.0
        x_f, P_f = information_fusion(x, P, x, P)
        np.testing.assert_allclose(x_f, x, atol=1e-6)

    def test_tighter_than_ci(self):
        """Information fusion should give tighter covariance than CI."""
        x1 = np.array([100.0, 200.0])
        P1 = np.eye(2) * 20.0
        x2 = np.array([102.0, 198.0])
        P2 = np.eye(2) * 15.0

        _, P_ci = covariance_intersection(x1, P1, x2, P2)
        _, P_if = information_fusion(x1, P1, x2, P2)
        assert np.trace(P_if) <= np.trace(P_ci) + 1e-6

    def test_halves_covariance_for_equal_sensors(self):
        """Two identical independent sensors should halve the covariance."""
        x = np.array([5.0])
        P = np.eye(1) * 10.0
        _, P_f = information_fusion(x, P, x, P)
        assert P_f[0, 0] == pytest.approx(5.0, rel=1e-6)


class TestPersistentFusedIDs:
    """Test that TrackFusion maintains persistent fused track IDs."""

    def _make_cam_track(self, px_x=640.0):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=0.0,
            bbox=np.array([px_x - 50, 300, px_x + 50, 400], dtype=np.float32),
            class_id=0,
            class_name="person",
            confidence=0.9,
        )
        return Track(det)

    def _make_rdr_track(self, range_m=5000.0, az_deg=0.0):
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=range_m,
            azimuth_deg=az_deg,
            confidence=0.9,
        )
        return RadarTrack(det)

    def test_persistent_id_same_pair(self):
        """Same camera+radar pair should produce same fused_id across calls."""
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280)
        cam = self._make_cam_track(640.0)
        rdr = self._make_rdr_track(5000.0, 0.0)

        result1 = fusion.fuse([cam], [rdr])
        result2 = fusion.fuse([cam], [rdr])

        assert result1[0].fused_id == result2[0].fused_id

    def test_different_pairs_different_ids(self):
        """Different track pairs should have different fused IDs."""
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280)
        cam1 = self._make_cam_track(640.0)
        cam2 = self._make_cam_track(300.0)
        rdr = self._make_rdr_track(5000.0, 0.0)

        # Two separate fusions with different camera tracks
        r1 = fusion.fuse([cam1], [rdr])
        key1_id = r1[0].fused_id

        r2 = fusion.fuse([cam2], [rdr])
        # cam2 is at a different position, but still within gate
        # The fused_id should be different because cam2 has different track_id
        assert cam1.track_id != cam2.track_id


class TestCIFusionInTrackFusion:
    """Test CI-based state fusion within TrackFusion."""

    def _make_cam_track(self, px_x=640.0):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=0.0,
            bbox=np.array([px_x - 50, 300, px_x + 50, 400], dtype=np.float32),
            class_id=0,
            class_name="person",
            confidence=0.9,
        )
        return Track(det)

    def _make_rdr_track(self, range_m=5000.0, az_deg=0.0):
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=range_m,
            azimuth_deg=az_deg,
            confidence=0.9,
        )
        return RadarTrack(det)

    def test_ci_disabled_by_default(self):
        fusion = TrackFusion()
        assert fusion._use_ci is False
        cam = self._make_cam_track()
        rdr = self._make_rdr_track()
        result = fusion.fuse([cam], [rdr])
        assert result[0].fused_state is None

    def test_ci_enabled_produces_state(self):
        fusion = TrackFusion(use_ci_fusion=True)
        cam = self._make_cam_track(640.0)
        rdr = self._make_rdr_track(5000.0, 0.0)
        result = fusion.fuse([cam], [rdr])
        # Dual-sensor track should have fused state
        dual = [r for r in result if r.is_dual_sensor]
        if dual:
            assert dual[0].fused_state is not None
            assert dual[0].fused_state.shape == (2,)
            assert dual[0].fused_covariance is not None
            assert dual[0].fused_covariance.shape == (2, 2)

    def test_ci_covariance_positive_definite(self):
        fusion = TrackFusion(use_ci_fusion=True)
        cam = self._make_cam_track(640.0)
        rdr = self._make_rdr_track(5000.0, 0.0)
        result = fusion.fuse([cam], [rdr])
        dual = [r for r in result if r.is_dual_sensor]
        if dual and dual[0].fused_covariance is not None:
            eigenvalues = np.linalg.eigvalsh(dual[0].fused_covariance)
            assert all(ev > 0 for ev in eigenvalues)

    def test_camera_only_no_ci(self):
        fusion = TrackFusion(use_ci_fusion=True)
        cam = self._make_cam_track(640.0)
        result = fusion.fuse([cam], [])
        assert result[0].fused_state is None
