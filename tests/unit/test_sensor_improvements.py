"""Tests for sensor model improvements: range-dependent noise, SNR Pd, Doppler EKF."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.sensors.multifreq_radar_sim import MultiFreqRadarConfig
from sentinel.sensors.physics import _snr_to_pd, radar_snr
from sentinel.sensors.quantum_radar_sim import QuantumRadarConfig
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget
from sentinel.tracking.filters import ExtendedKalmanFilterWithDoppler
from sentinel.tracking.radar_track import RadarTrack


class TestRangeDependentNoise:
    """Test range-dependent noise scaling."""

    def test_disabled_by_default(self):
        cfg = RadarSimConfig()
        assert cfg.range_dependent_noise is False

    def test_noise_increases_with_range(self):
        """Range-dependent noise should produce larger errors at far range."""
        close_target = RadarTarget("CLOSE", np.array([1000.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)
        far_target = RadarTarget("FAR", np.array([9000.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)

        # Range-dependent enabled
        cfg = RadarSimConfig(
            max_range_m=10000.0,
            noise_range_m=10.0,
            noise_azimuth_deg=2.0,
            detection_probability=1.0,
            range_dependent_noise=True,
            targets=[close_target, far_target],
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        # Collect many detections and compare noise magnitudes
        close_errors = []
        far_errors = []
        for _ in range(200):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("target_id") == "CLOSE":
                    close_errors.append(abs(d["range_m"] - 1000.0))
                elif d.get("target_id") == "FAR":
                    far_errors.append(abs(d["range_m"] - 9000.0))

        # Far targets should have larger average error
        assert len(close_errors) > 0
        assert len(far_errors) > 0
        avg_close = np.mean(close_errors)
        avg_far = np.mean(far_errors)
        assert avg_far > avg_close * 1.3  # At least 30% more noise

    def test_disabled_gives_uniform_noise(self):
        """Without range-dependent noise, errors should be similar at all ranges."""
        close_target = RadarTarget("CLOSE", np.array([1000.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)
        far_target = RadarTarget("FAR", np.array([9000.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)

        cfg = RadarSimConfig(
            max_range_m=10000.0,
            noise_range_m=10.0,
            detection_probability=1.0,
            range_dependent_noise=False,
            targets=[close_target, far_target],
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        close_errors = []
        far_errors = []
        for _ in range(200):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("target_id") == "CLOSE":
                    close_errors.append(abs(d["range_m"] - 1000.0))
                elif d.get("target_id") == "FAR":
                    far_errors.append(abs(d["range_m"] - 9000.0))

        avg_close = np.mean(close_errors)
        avg_far = np.mean(far_errors)
        # Should be within ~40% of each other (statistical noise)
        ratio = avg_far / avg_close if avg_close > 0 else 999
        assert 0.5 < ratio < 2.0

    def test_multifreq_range_dependent(self):
        """Multi-freq radar should also support range-dependent noise."""
        cfg = MultiFreqRadarConfig(range_dependent_noise=True)
        assert cfg.range_dependent_noise is True

    def test_quantum_range_dependent(self):
        """Quantum radar should also support range-dependent noise."""
        cfg = QuantumRadarConfig(range_dependent_noise=True)
        assert cfg.range_dependent_noise is True


class TestSNRBasedPd:
    """Test SNR-based detection probability."""

    def test_snr_pd_disabled_by_default(self):
        cfg = RadarSimConfig()
        assert cfg.use_snr_pd is False

    def test_snr_based_pd_range_effect(self):
        """Closer targets should be detected more often with SNR-based Pd."""
        close_target = RadarTarget("CLOSE", np.array([2000.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)
        far_target = RadarTarget("FAR", np.array([9500.0, 0.0]), np.zeros(2), rcs_dbsm=15.0)

        cfg = RadarSimConfig(
            max_range_m=10000.0,
            use_snr_pd=True,
            targets=[close_target, far_target],
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        close_count = 0
        far_count = 0
        n_scans = 500
        for _ in range(n_scans):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("target_id") == "CLOSE":
                    close_count += 1
                elif d.get("target_id") == "FAR":
                    far_count += 1

        # Close target should be detected more often
        close_rate = close_count / n_scans
        far_rate = far_count / n_scans
        assert close_rate > far_rate

    def test_snr_based_pd_rcs_effect(self):
        """Larger RCS should increase detection probability."""
        big_rcs = RadarTarget("BIG", np.array([5000.0, 0.0]), np.zeros(2), rcs_dbsm=20.0)
        small_rcs = RadarTarget("SMALL", np.array([5000.0, 100.0]), np.zeros(2), rcs_dbsm=-5.0)

        cfg = RadarSimConfig(
            max_range_m=10000.0,
            use_snr_pd=True,
            targets=[big_rcs, small_rcs],
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        big_count = 0
        small_count = 0
        n_scans = 500
        for _ in range(n_scans):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("target_id") == "BIG":
                    big_count += 1
                elif d.get("target_id") == "SMALL":
                    small_count += 1

        # Big RCS should be detected more often
        assert big_count > small_count

    def test_snr_to_pd_known_values(self):
        """Check SNR-to-Pd conversion at known design points."""
        # Monotonic: higher SNR -> higher Pd
        pd_0 = _snr_to_pd(0.0)
        pd_13 = _snr_to_pd(13.0)
        pd_30 = _snr_to_pd(30.0)
        assert pd_0 < pd_13 < pd_30

        # Very high SNR -> Pd near 1
        assert pd_30 > 0.99

        # Very low SNR -> Pd near 0
        pd_neg20 = _snr_to_pd(-20.0)
        assert pd_neg20 < 0.1

    def test_radar_snr_inverse_r4(self):
        """SNR should decrease by 12 dB when range doubles (R^4 law)."""
        snr_5k = radar_snr(10.0, 5000.0)
        snr_10k = radar_snr(10.0, 10000.0)
        # Doubling range: 4 * 10*log10(2) = 12.04 dB
        assert snr_5k - snr_10k == pytest.approx(12.04, abs=0.1)


class TestDopplerEKF:
    """Test Extended Kalman Filter with Doppler measurement."""

    def test_init(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        assert ekf.dim_state == 4
        assert ekf.dim_meas == 3
        assert ekf.R.shape == (3, 3)

    def test_measurement_function(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        # Target at (3000, 10, 4000, -5) -> heading toward radar
        ekf.x = np.array([3000.0, 10.0, 4000.0, -5.0])
        h = ekf.h(ekf.x)
        expected_r = np.sqrt(3000**2 + 4000**2)
        expected_az = np.arctan2(4000.0, 3000.0)
        expected_vr = (3000 * 10 + 4000 * (-5)) / expected_r
        assert h[0] == pytest.approx(expected_r, rel=1e-6)
        assert h[1] == pytest.approx(expected_az, rel=1e-6)
        assert h[2] == pytest.approx(expected_vr, rel=1e-6)

    def test_jacobian_shape(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        ekf.x = np.array([3000.0, 10.0, 4000.0, -5.0])
        H = ekf.H_jacobian(ekf.x)
        assert H.shape == (3, 4)

    def test_jacobian_numerical_check(self):
        """Analytical Jacobian should match numerical differentiation."""
        ekf = ExtendedKalmanFilterWithDoppler()
        x0 = np.array([3000.0, 10.0, 4000.0, -5.0])
        ekf.x = x0.copy()
        H_analytical = ekf.H_jacobian(x0)

        # Numerical Jacobian
        eps = 1e-5
        H_numerical = np.zeros((3, 4))
        for i in range(4):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            H_numerical[:, i] = (ekf.h(x_plus) - ekf.h(x_minus)) / (2 * eps)

        np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-4)

    def test_position_velocity(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        ekf.x = np.array([100.0, 10.0, 200.0, 20.0])
        np.testing.assert_allclose(ekf.position, [100.0, 200.0])
        np.testing.assert_allclose(ekf.velocity, [10.0, 20.0])

    def test_predict(self):
        ekf = ExtendedKalmanFilterWithDoppler(dt=1.0)
        ekf.x = np.array([0.0, 10.0, 0.0, 20.0])
        ekf.predict()
        assert ekf.x[0] == pytest.approx(10.0, abs=0.1)
        assert ekf.x[2] == pytest.approx(20.0, abs=0.1)

    def test_update_converges(self):
        """Doppler EKF should converge to true position and velocity."""
        ekf = ExtendedKalmanFilterWithDoppler(dt=0.1)
        true_x, true_y = 5000.0, 3000.0
        true_vx, true_vy = -20.0, 10.0
        ekf.x = np.array([4000.0, 0.0, 2000.0, 0.0])

        true_r = np.sqrt(true_x**2 + true_y**2)
        true_az = np.arctan2(true_y, true_x)
        true_vr = (true_x * true_vx + true_y * true_vy) / true_r

        for _ in range(50):
            ekf.predict()
            z = np.array(
                [
                    true_r + np.random.randn() * 5.0,
                    true_az + np.random.randn() * np.radians(1.0),
                    true_vr + np.random.randn() * 0.5,
                ]
            )
            ekf.update(z)

        assert abs(ekf.position[0] - true_x) < 300.0
        assert abs(ekf.position[1] - true_y) < 300.0

    def test_velocity_converges_faster_than_no_doppler(self):
        """Doppler EKF should converge velocity faster than standard EKF."""
        from sentinel.tracking.filters import ExtendedKalmanFilter

        np.random.seed(42)
        true_x, true_y = 5000.0, 3000.0
        true_vx, true_vy = -20.0, 10.0
        true_r = np.sqrt(true_x**2 + true_y**2)
        true_az = np.arctan2(true_y, true_x)
        true_vr = (true_x * true_vx + true_y * true_vy) / true_r

        # Standard EKF (no Doppler)
        ekf_std = ExtendedKalmanFilter(dt=0.1)
        ekf_std.x = np.array([4000.0, 0.0, 2000.0, 0.0])

        # Doppler EKF
        ekf_dop = ExtendedKalmanFilterWithDoppler(dt=0.1)
        ekf_dop.x = np.array([4000.0, 0.0, 2000.0, 0.0])

        n_steps = 20
        for _ in range(n_steps):
            r_noise = np.random.randn() * 5.0
            az_noise = np.random.randn() * np.radians(1.0)
            vr_noise = np.random.randn() * 0.5

            ekf_std.predict()
            ekf_std.update(np.array([true_r + r_noise, true_az + az_noise]))

            ekf_dop.predict()
            ekf_dop.update(
                np.array(
                    [
                        true_r + r_noise,
                        true_az + az_noise,
                        true_vr + vr_noise,
                    ]
                )
            )

        # Doppler EKF should have better velocity estimate
        vel_err_std = np.linalg.norm(ekf_std.velocity - [true_vx, true_vy])
        vel_err_dop = np.linalg.norm(ekf_dop.velocity - [true_vx, true_vy])
        assert vel_err_dop < vel_err_std

    def test_gating_distance(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        ekf.x = np.array([5000.0, 10.0, 3000.0, -5.0])
        z = ekf.h(ekf.x)  # Perfect measurement
        dist = ekf.gating_distance(z)
        assert dist < 1.0

    def test_set_measurement_noise(self):
        ekf = ExtendedKalmanFilterWithDoppler()
        ekf.set_measurement_noise(10.0, np.radians(2.0), 1.0)
        assert ekf.R[0, 0] == pytest.approx(100.0)
        assert ekf.R[2, 2] == pytest.approx(1.0)
        assert ekf.R.shape == (3, 3)


class TestDopplerRadarTrack:
    """Test RadarTrack with Doppler EKF enabled."""

    def _make_radar_det(self, range_m=5000.0, az_deg=30.0, vel_mps=50.0):
        return Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=range_m,
            azimuth_deg=az_deg,
            velocity_mps=vel_mps,
            confidence=0.9,
        )

    def test_doppler_init(self):
        det = self._make_radar_det()
        track = RadarTrack(det, use_doppler=True)
        assert isinstance(track.ekf, ExtendedKalmanFilterWithDoppler)

    def test_doppler_update(self):
        det = self._make_radar_det()
        track = RadarTrack(det, use_doppler=True)
        track.predict()
        det2 = self._make_radar_det(range_m=4990.0, az_deg=30.1, vel_mps=50.2)
        track.update(det2)
        assert track.hits == 2

    def test_doppler_fallback_no_velocity(self):
        """Without velocity in detection, should still use range+azimuth."""
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=5000.0,
            azimuth_deg=30.0,
            velocity_mps=None,
            confidence=0.9,
        )
        track = RadarTrack(det, use_doppler=True)
        track.predict()
        # Update without velocity - falls back to standard 2D measurement
        det2 = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=4990.0,
            azimuth_deg=30.1,
            velocity_mps=None,
            confidence=0.9,
        )
        # This should not crash even though Doppler EKF expects 3 measurements
        # The update code checks for velocity_mps before constructing 3D measurement
        track.update(det2)
        assert track.hits == 2
