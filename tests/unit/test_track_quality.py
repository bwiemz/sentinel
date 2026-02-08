"""Tests for track quality metrics: NIS, NEES, FilterConsistencyMonitor."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType, TrackState
from sentinel.tracking.track_quality import (
    FilterConsistencyMonitor,
    compute_nees,
    compute_nis,
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


# ===========================================================================
# TestComputeNIS
# ===========================================================================

class TestComputeNIS:
    """Tests for the compute_nis() function."""

    def test_identity_covariance(self):
        y = np.array([1.0, 0.0])
        S = np.eye(2)
        assert compute_nis(y, S) == pytest.approx(1.0)

    def test_scaled_covariance(self):
        y = np.array([1.0, 1.0])
        S = 4.0 * np.eye(2)
        # NIS = y' * (4I)^-1 * y = (1/4)(1+1) = 0.5
        assert compute_nis(y, S) == pytest.approx(0.5)

    def test_1d(self):
        y = np.array([3.0])
        S = np.array([[9.0]])
        # NIS = 9/9 = 1.0
        assert compute_nis(y, S) == pytest.approx(1.0)

    def test_3d(self):
        y = np.array([1.0, 2.0, 3.0])
        S = np.eye(3)
        # NIS = 1+4+9 = 14
        assert compute_nis(y, S) == pytest.approx(14.0)

    def test_singular_covariance_returns_inf(self):
        y = np.array([1.0, 0.0])
        S = np.zeros((2, 2))
        assert compute_nis(y, S) == float("inf")

    def test_zero_innovation(self):
        y = np.array([0.0, 0.0])
        S = np.eye(2)
        assert compute_nis(y, S) == pytest.approx(0.0)


# ===========================================================================
# TestComputeNEES
# ===========================================================================

class TestComputeNEES:
    """Tests for the compute_nees() function."""

    def test_identity_covariance(self):
        x_true = np.array([1.0, 0.0])
        x_est = np.array([0.0, 0.0])
        P = np.eye(2)
        assert compute_nees(x_true, x_est, P) == pytest.approx(1.0)

    def test_scaled_covariance(self):
        x_true = np.array([2.0, 2.0])
        x_est = np.array([0.0, 0.0])
        P = 4.0 * np.eye(2)
        # NEES = [2,2]' * (4I)^-1 * [2,2] = (4+4)/4 = 2.0
        assert compute_nees(x_true, x_est, P) == pytest.approx(2.0)

    def test_1d(self):
        x_true = np.array([5.0])
        x_est = np.array([2.0])
        P = np.array([[9.0]])
        # NEES = 9/9 = 1.0
        assert compute_nees(x_true, x_est, P) == pytest.approx(1.0)

    def test_singular_covariance_returns_inf(self):
        x_true = np.array([1.0])
        x_est = np.array([0.0])
        P = np.array([[0.0]])
        assert compute_nees(x_true, x_est, P) == float("inf")

    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3)
        assert compute_nees(x, x, P) == pytest.approx(0.0)


# ===========================================================================
# TestFilterConsistencyMonitor
# ===========================================================================

class TestFilterConsistencyMonitor:
    """Tests for FilterConsistencyMonitor."""

    def test_initial_state(self):
        mon = FilterConsistencyMonitor(dim_meas=2)
        assert mon.sample_count == 0
        assert mon.average_nis == 2.0  # Default = dim_meas
        assert mon.nis_ratio == pytest.approx(1.0)
        assert mon.filter_health == "nominal"

    def test_record_and_count(self):
        mon = FilterConsistencyMonitor(dim_meas=2)
        y = np.array([1.0, 0.0])
        S = np.eye(2)
        nis = mon.record_innovation(y, S)
        assert nis == pytest.approx(1.0)
        assert mon.sample_count == 1

    def test_average_nis(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10)
        S = np.eye(2)
        # Record three innovations with NIS = 1, 4, 9
        mon.record_innovation(np.array([1.0, 0.0]), S)   # NIS=1
        mon.record_innovation(np.array([2.0, 0.0]), S)   # NIS=4
        mon.record_innovation(np.array([3.0, 0.0]), S)   # NIS=9
        assert mon.average_nis == pytest.approx((1 + 4 + 9) / 3)

    def test_nis_ratio(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10)
        S = np.eye(2)
        # Record NIS values that average to 4.0 → ratio = 4/2 = 2.0
        for _ in range(5):
            mon.record_innovation(np.array([2.0, 0.0]), S)  # NIS=4 each
        assert mon.nis_ratio == pytest.approx(2.0)

    def test_health_nominal(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10)
        S = np.eye(2)
        # NIS ~ 2 → ratio ~1.0 → nominal
        for _ in range(5):
            mon.record_innovation(np.array([1.0, 1.0]), S)  # NIS=2
        assert mon.filter_health == "nominal"

    def test_health_over_confident(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10,
                                       over_confident_threshold=2.0)
        S = np.eye(2)
        # NIS ~ 6 → ratio = 3.0 → over_confident (>2.0 but <=6.0)
        for _ in range(5):
            y = np.array([np.sqrt(3.0), np.sqrt(3.0)])  # NIS=6
            mon.record_innovation(y, S)
        assert mon.filter_health == "over_confident"

    def test_health_under_confident(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10,
                                       under_confident_threshold=0.3)
        S = np.eye(2)
        # Very small innovations → NIS ~ 0.02 → ratio = 0.01 → under_confident
        for _ in range(5):
            mon.record_innovation(np.array([0.1, 0.1]), S)  # NIS=0.02
        assert mon.filter_health == "under_confident"

    def test_health_diverged(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10,
                                       over_confident_threshold=2.0)
        S = np.eye(2)
        # NIS ~ 26 → ratio = 13.0 → diverged (>3*2.0=6.0)
        for _ in range(5):
            y = np.array([np.sqrt(13.0), np.sqrt(13.0)])  # NIS=26
            mon.record_innovation(y, S)
        assert mon.filter_health == "diverged"

    def test_consistency_score_perfect(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10)
        S = np.eye(2)
        # NIS = 2 → ratio = 1.0 → score = exp(0) = 1.0
        for _ in range(5):
            mon.record_innovation(np.array([1.0, 1.0]), S)
        assert mon.consistency_score == pytest.approx(1.0, abs=0.01)

    def test_consistency_score_poor(self):
        mon = FilterConsistencyMonitor(dim_meas=2, window_size=10)
        S = np.eye(2)
        # NIS = 20 → ratio = 10.0 → deviation = 9.0 → score ≈ exp(-162) ≈ 0
        for _ in range(5):
            y = np.array([np.sqrt(10.0), np.sqrt(10.0)])  # NIS=20
            mon.record_innovation(y, S)
        assert mon.consistency_score < 0.01

    def test_reset(self):
        mon = FilterConsistencyMonitor(dim_meas=2)
        mon.record_innovation(np.array([1.0, 0.0]), np.eye(2))
        assert mon.sample_count == 1
        mon.reset()
        assert mon.sample_count == 0

    def test_inf_nis_filtered_out(self):
        mon = FilterConsistencyMonitor(dim_meas=2)
        # Singular covariance → inf NIS → should NOT be stored
        nis = mon.record_innovation(np.array([1.0, 0.0]), np.zeros((2, 2)))
        assert nis == float("inf")
        assert mon.sample_count == 0

    def test_consistency_score_not_enough_samples(self):
        mon = FilterConsistencyMonitor(dim_meas=2)
        # < 3 samples → assume good
        mon.record_innovation(np.array([5.0, 5.0]), np.eye(2))  # bad NIS
        assert mon.consistency_score == 1.0  # Not enough data to judge


# ===========================================================================
# TestBaseTrackQualityIntegration
# ===========================================================================

class TestBaseTrackQualityIntegration:
    """Tests that TrackBase properly creates and uses the quality monitor."""

    def test_no_monitor_without_dim(self):
        from sentinel.tracking.base_track import TrackBase
        tb = TrackBase()
        assert tb.quality_monitor is None

    def test_monitor_created_with_dim(self):
        from sentinel.tracking.base_track import TrackBase
        tb = TrackBase(measurement_dim=2)
        assert tb.quality_monitor is not None
        assert tb.quality_monitor._dim_meas == 2

    def test_score_without_monitor(self):
        from sentinel.tracking.base_track import TrackBase
        tb = TrackBase()
        tb.hits = 5
        tb.age = 10
        tb.consecutive_misses = 0
        tb.state = TrackState.CONFIRMED
        tb._update_score()
        # Without monitor: 0.4*hit_ratio + 0.3*recency + 0.3*confirmation
        # = 0.4*0.5 + 0.3*1.0 + 0.3*1.0 = 0.2 + 0.3 + 0.3 = 0.8
        assert tb.score == pytest.approx(0.8)

    def test_score_with_monitor(self):
        from sentinel.tracking.base_track import TrackBase
        tb = TrackBase(measurement_dim=2)
        tb.hits = 5
        tb.age = 10
        tb.consecutive_misses = 0
        tb.state = TrackState.CONFIRMED
        # Feed 5 samples with perfect NIS (ratio=1.0 → consistency_score=1.0)
        for _ in range(5):
            tb.quality_monitor.record_innovation(np.array([1.0, 1.0]), np.eye(2))
        tb._update_score()
        # With monitor (>=3 samples): 0.3*hit_ratio + 0.25*recency + 0.25*conf + 0.2*quality
        # = 0.3*0.5 + 0.25*1.0 + 0.25*1.0 + 0.2*1.0 = 0.15 + 0.25 + 0.25 + 0.2 = 0.85
        assert tb.score == pytest.approx(0.85)


# ===========================================================================
# TestTrackNISRecording
# ===========================================================================

class TestTrackNISRecording:
    """Tests that NIS is recorded through track.update() for each sensor type."""

    def test_camera_track_records_nis(self):
        from sentinel.tracking.track import Track
        det0 = _camera_detection(100, 100, 200, 200, ts=0.0)
        track = Track(det0, dt=1 / 30)
        assert track.quality_monitor is not None
        assert track.quality_monitor.sample_count == 0

        # Predict then update
        track.predict()
        det1 = _camera_detection(105, 105, 205, 205, ts=0.033)
        track.update(det1)
        assert track.quality_monitor.sample_count == 1
        assert track.quality_monitor.average_nis >= 0

    def test_radar_2d_records_nis(self):
        from sentinel.tracking.radar_track import RadarTrack
        det0 = _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0)
        track = RadarTrack(det0, dt=0.1)
        assert track.quality_monitor is not None
        assert track.quality_monitor._dim_meas == 2

        track.predict()
        det1 = _radar_detection(range_m=5010.0, azimuth_deg=45.1, ts=0.1)
        track.update(det1)
        assert track.quality_monitor.sample_count == 1

    def test_radar_3d_records_nis(self):
        from sentinel.tracking.radar_track import RadarTrack
        det0 = _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0,
                                elevation_deg=10.0)
        track = RadarTrack(det0, dt=0.1, use_3d=True)
        assert track.quality_monitor is not None
        assert track.quality_monitor._dim_meas == 3

        track.predict()
        det1 = _radar_detection(range_m=5010.0, azimuth_deg=45.1, ts=0.1,
                                elevation_deg=10.1)
        track.update(det1)
        assert track.quality_monitor.sample_count == 1

    def test_radar_doppler_records_nis(self):
        from sentinel.tracking.radar_track import RadarTrack
        det0 = _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0,
                                velocity_mps=250.0)
        track = RadarTrack(det0, dt=0.1, use_doppler=True)
        assert track.quality_monitor is not None
        assert track.quality_monitor._dim_meas == 3

        track.predict()
        det1 = _radar_detection(range_m=5010.0, azimuth_deg=45.1, ts=0.1,
                                velocity_mps=252.0)
        track.update(det1)
        assert track.quality_monitor.sample_count == 1

    def test_thermal_track_records_nis(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        det0 = _thermal_detection(azimuth_deg=30.0, ts=0.0)
        track = ThermalTrack(det0, dt=0.033)
        assert track.quality_monitor is not None
        assert track.quality_monitor._dim_meas == 1

        track.predict()
        det1 = _thermal_detection(azimuth_deg=30.5, ts=0.033)
        track.update(det1)
        assert track.quality_monitor.sample_count == 1
