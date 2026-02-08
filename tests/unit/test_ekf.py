"""Tests for Extended Kalman Filter."""

import numpy as np
import pytest

from sentinel.tracking.filters import ExtendedKalmanFilter


class TestExtendedKalmanFilter:
    def test_init_state(self):
        ekf = ExtendedKalmanFilter()
        assert ekf.x.shape == (4,)
        assert ekf.P.shape == (4, 4)
        assert ekf.dim_state == 4
        assert ekf.dim_meas == 2
        np.testing.assert_array_equal(ekf.x, np.zeros(4))

    def test_h_function_along_x(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([100.0, 0.0, 0.0, 0.0])
        z = ekf.h(ekf.x)
        assert z[0] == pytest.approx(100.0)  # range
        assert z[1] == pytest.approx(0.0)  # azimuth

    def test_h_function_along_y(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([0.0, 0.0, 100.0, 0.0])
        z = ekf.h(ekf.x)
        assert z[0] == pytest.approx(100.0)
        assert z[1] == pytest.approx(np.pi / 2)

    def test_h_function_45_degrees(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([100.0, 0.0, 100.0, 0.0])
        z = ekf.h(ekf.x)
        assert z[0] == pytest.approx(np.sqrt(20000.0))
        assert z[1] == pytest.approx(np.pi / 4)

    def test_jacobian_at_known_point(self):
        """Verify analytical Jacobian matches numerical Jacobian."""
        ekf = ExtendedKalmanFilter()
        x0 = np.array([3000.0, 10.0, 4000.0, -5.0])
        H_analytical = ekf.H_jacobian(x0)

        # Numerical Jacobian via finite differences
        eps = 1e-5
        H_numerical = np.zeros((2, 4))
        for i in range(4):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += eps
            xm[i] -= eps
            H_numerical[:, i] = (ekf.h(xp) - ekf.h(xm)) / (2 * eps)

        np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-5)

    def test_jacobian_degenerate_origin(self):
        """Jacobian should not crash when state is near origin."""
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([0.0, 0.0, 0.0, 0.0])
        H = ekf.H_jacobian(ekf.x)
        assert H.shape == (2, 4)
        assert np.all(np.isfinite(H))

    def test_predict_constant_velocity(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x = np.array([100.0, 20.0, 200.0, -10.0])
        ekf.predict()
        assert ekf.x[0] == pytest.approx(102.0)  # x + vx*dt
        assert ekf.x[1] == pytest.approx(20.0)  # vx unchanged
        assert ekf.x[2] == pytest.approx(199.0)  # y + vy*dt
        assert ekf.x[3] == pytest.approx(-10.0)  # vy unchanged

    def test_predict_increases_uncertainty(self):
        ekf = ExtendedKalmanFilter()
        P_before = np.trace(ekf.P)
        ekf.predict()
        assert np.trace(ekf.P) > P_before

    def test_update_reduces_uncertainty(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([3000.0, 0.0, 4000.0, 0.0])
        ekf.predict()
        P_before = np.trace(ekf.P)
        z = ekf.h(ekf.x) + np.array([1.0, 0.001])  # small noise
        ekf.update(z)
        assert np.trace(ekf.P) < P_before

    def test_convergence_on_static_target(self):
        """EKF should converge to a static target with repeated noisy measurements."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.set_measurement_noise(5.0, np.radians(1.0))

        # True target at (3000, 4000)
        true_range = 5000.0
        true_azimuth = np.arctan2(4000.0, 3000.0)

        # Initialize state roughly
        ekf.x = np.array([2800.0, 0.0, 3800.0, 0.0])

        rng = np.random.RandomState(42)
        for _ in range(100):
            ekf.predict()
            z = np.array(
                [
                    true_range + rng.randn() * 5.0,
                    true_azimuth + rng.randn() * np.radians(1.0),
                ]
            )
            ekf.update(z)

        pos = ekf.position
        assert pos[0] == pytest.approx(3000.0, abs=20.0)
        assert pos[1] == pytest.approx(4000.0, abs=20.0)

    def test_convergence_on_moving_target(self):
        """EKF should track a constant-velocity target."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.set_measurement_noise(5.0, np.radians(1.0))

        # Target starts at (1000, 0) moving at (10, 5) m/s
        ekf.x = np.array([1000.0, 0.0, 0.0, 0.0])

        rng = np.random.RandomState(42)
        for step in range(200):
            t = step * 0.1
            true_x = 1000.0 + 10.0 * t
            true_y = 0.0 + 5.0 * t
            true_range = np.sqrt(true_x**2 + true_y**2)
            true_az = np.arctan2(true_y, true_x)

            ekf.predict()
            z = np.array(
                [
                    true_range + rng.randn() * 5.0,
                    true_az + rng.randn() * np.radians(1.0),
                ]
            )
            ekf.update(z)

        # After 200 steps (20s), target at (1200, 100)
        pos = ekf.position
        vel = ekf.velocity
        assert pos[0] == pytest.approx(1200.0, abs=30.0)
        assert pos[1] == pytest.approx(100.0, abs=30.0)
        assert vel[0] == pytest.approx(10.0, abs=3.0)
        assert vel[1] == pytest.approx(5.0, abs=3.0)

    def test_angular_wrapping(self):
        """EKF should handle targets crossing the +/-pi boundary."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.set_measurement_noise(5.0, np.radians(1.0))

        # Target at azimuth ~pi (just below), moving to cross the boundary
        ekf.x = np.array([-5000.0, 0.0, 10.0, 0.0])

        # Measurement at azimuth just above -pi (crossed boundary)
        z = np.array([5000.0, -np.pi + 0.01])
        ekf.predict()
        ekf.update(z)

        # Should not diverge
        assert np.all(np.isfinite(ekf.x))
        assert np.all(np.isfinite(ekf.P))

    def test_gating_distance_accepts_close(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([3000.0, 0.0, 4000.0, 0.0])
        z_close = ekf.h(ekf.x) + np.array([2.0, np.radians(0.5)])
        dist = ekf.gating_distance(z_close)
        assert dist < 9.21  # within 99% gate

    def test_gating_distance_rejects_far(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([3000.0, 0.0, 4000.0, 0.0])
        z_far = np.array([10000.0, np.pi / 2])  # completely different
        dist = ekf.gating_distance(z_far)
        assert dist > 9.21

    def test_covariance_stays_symmetric(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x = np.array([1000.0, 5.0, 2000.0, -3.0])
        rng = np.random.RandomState(42)
        for _ in range(50):
            ekf.predict()
            r = np.sqrt(ekf.x[0] ** 2 + ekf.x[2] ** 2)
            az = np.arctan2(ekf.x[2], ekf.x[0])
            z = np.array([r + rng.randn() * 5.0, az + rng.randn() * 0.01])
            ekf.update(z)
            np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-10)

    def test_covariance_stays_positive_definite(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x = np.array([1000.0, 5.0, 2000.0, -3.0])
        rng = np.random.RandomState(42)
        for _ in range(50):
            ekf.predict()
            r = np.sqrt(ekf.x[0] ** 2 + ekf.x[2] ** 2)
            az = np.arctan2(ekf.x[2], ekf.x[0])
            z = np.array([r + rng.randn() * 5.0, az + rng.randn() * 0.01])
            ekf.update(z)
            eigvals = np.linalg.eigvalsh(ekf.P)
            assert np.all(eigvals > 0), f"Non-PD covariance: eigenvalues={eigvals}"

    def test_position_property(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([100.0, 1.0, 200.0, 2.0])
        np.testing.assert_array_equal(ekf.position, [100.0, 200.0])

    def test_velocity_property(self):
        ekf = ExtendedKalmanFilter()
        ekf.x = np.array([100.0, 1.0, 200.0, 2.0])
        np.testing.assert_array_equal(ekf.velocity, [1.0, 2.0])
