"""Tests for Constant-Acceleration Kalman filters."""

import numpy as np
import pytest

from sentinel.tracking.filters import ConstantAccelerationEKF, ConstantAccelerationKF


class TestConstantAccelerationKF:
    """Test CA-KF in pixel space."""

    def test_init_state(self):
        kf = ConstantAccelerationKF(dt=0.1)
        assert kf.dim_state == 6
        assert kf.dim_meas == 2
        assert kf.x.shape == (6,)
        assert kf.P.shape == (6, 6)
        assert kf.F.shape == (6, 6)
        assert kf.H.shape == (2, 6)

    def test_transition_matrix_structure(self):
        dt = 0.1
        kf = ConstantAccelerationKF(dt=dt)
        # x-axis: F[0,1]=dt, F[0,2]=dt^2/2, F[1,2]=dt
        assert kf.F[0, 1] == pytest.approx(dt)
        assert kf.F[0, 2] == pytest.approx(dt**2 / 2)
        assert kf.F[1, 2] == pytest.approx(dt)
        # y-axis: same structure at [3:6, 3:6]
        assert kf.F[3, 4] == pytest.approx(dt)
        assert kf.F[3, 5] == pytest.approx(dt**2 / 2)
        assert kf.F[4, 5] == pytest.approx(dt)

    def test_predict_with_acceleration(self):
        kf = ConstantAccelerationKF(dt=1.0)
        kf.x = np.array([0.0, 10.0, 2.0, 0.0, 5.0, -1.0])
        kf.predict()
        # x: 0 + 10*1 + 2*0.5 = 11
        assert kf.x[0] == pytest.approx(11.0, abs=0.1)
        # vx: 10 + 2*1 = 12
        assert kf.x[1] == pytest.approx(12.0, abs=0.1)
        # ax unchanged: 2
        assert kf.x[2] == pytest.approx(2.0, abs=0.1)

    def test_position_property(self):
        kf = ConstantAccelerationKF()
        kf.x = np.array([100.0, 1.0, 0.0, 200.0, 2.0, 0.0])
        np.testing.assert_allclose(kf.position, [100.0, 200.0])

    def test_velocity_property(self):
        kf = ConstantAccelerationKF()
        kf.x = np.array([100.0, 10.0, 0.5, 200.0, 20.0, -0.5])
        np.testing.assert_allclose(kf.velocity, [10.0, 20.0])

    def test_acceleration_property(self):
        kf = ConstantAccelerationKF()
        kf.x = np.array([0.0, 0.0, 3.0, 0.0, 0.0, -2.0])
        np.testing.assert_allclose(kf.acceleration, [3.0, -2.0])

    def test_update_converges(self):
        """CA-KF should converge to true position with repeated measurements."""
        kf = ConstantAccelerationKF(dt=0.1)
        kf.x[0] = 100.0
        kf.x[3] = 200.0
        true_pos = np.array([150.0, 250.0])
        for _ in range(20):
            kf.predict()
            kf.update(true_pos + np.random.randn(2) * 2.0)
        assert abs(kf.position[0] - true_pos[0]) < 10.0
        assert abs(kf.position[1] - true_pos[1]) < 10.0

    def test_observation_matrix(self):
        kf = ConstantAccelerationKF()
        # H should pick x (index 0) and y (index 3)
        assert kf.H[0, 0] == 1.0
        assert kf.H[1, 3] == 1.0
        # All other entries zero
        assert kf.H[0, 1] == 0.0
        assert kf.H[0, 2] == 0.0
        assert kf.H[1, 4] == 0.0

    def test_gating_distance(self):
        kf = ConstantAccelerationKF()
        kf.x = np.array([100.0, 0.0, 0.0, 200.0, 0.0, 0.0])
        # Measurement at the predicted position should give ~0 distance
        dist = kf.gating_distance(np.array([100.0, 200.0]))
        assert dist < 1.0

    def test_set_process_noise(self):
        kf = ConstantAccelerationKF()
        Q_before = kf.Q.copy()
        kf.set_process_noise_std(5.0)
        # Q should change (scale with sigma^2)
        assert not np.allclose(kf.Q, Q_before)

    def test_set_measurement_noise(self):
        kf = ConstantAccelerationKF()
        kf.set_measurement_noise_std(3.0)
        np.testing.assert_allclose(kf.R[0, 0], 9.0)


class TestConstantAccelerationEKF:
    """Test CA-EKF for radar (polar measurements)."""

    def test_init_state(self):
        ekf = ConstantAccelerationEKF(dt=0.1)
        assert ekf.dim_state == 6
        assert ekf.dim_meas == 2
        assert ekf.x.shape == (6,)

    def test_measurement_function(self):
        ekf = ConstantAccelerationEKF()
        # Target at (3000, 0, 0, 4000, 0, 0) -> range=5000, az=atan2(4000,3000)
        ekf.x = np.array([3000.0, 0.0, 0.0, 4000.0, 0.0, 0.0])
        h = ekf.h(ekf.x)
        assert h[0] == pytest.approx(5000.0)
        assert h[1] == pytest.approx(np.arctan2(4000.0, 3000.0))

    def test_jacobian_shape(self):
        ekf = ConstantAccelerationEKF()
        ekf.x = np.array([1000.0, 10.0, 1.0, 500.0, 5.0, 0.5])
        H = ekf.H_jacobian(ekf.x)
        assert H.shape == (2, 6)

    def test_position_velocity_acceleration(self):
        ekf = ConstantAccelerationEKF()
        ekf.x = np.array([100.0, 10.0, 1.0, 200.0, 20.0, 2.0])
        np.testing.assert_allclose(ekf.position, [100.0, 200.0])
        np.testing.assert_allclose(ekf.velocity, [10.0, 20.0])
        np.testing.assert_allclose(ekf.acceleration, [1.0, 2.0])

    def test_predict_with_acceleration(self):
        dt = 1.0
        ekf = ConstantAccelerationEKF(dt=dt)
        ekf.x = np.array([1000.0, 50.0, 5.0, 2000.0, -30.0, 2.0])
        ekf.predict()
        # x: 1000 + 50*1 + 5*0.5 = 1052.5
        assert ekf.x[0] == pytest.approx(1052.5, abs=1.0)
        # vx: 50 + 5*1 = 55
        assert ekf.x[1] == pytest.approx(55.0, abs=1.0)

    def test_update_converges(self):
        """CA-EKF should converge to true target position."""
        ekf = ConstantAccelerationEKF(dt=0.1)
        # True target at (5000, 3000) moving slowly
        true_x, true_y = 5000.0, 3000.0
        ekf.x = np.array([4000.0, 0.0, 0.0, 2000.0, 0.0, 0.0])
        # Widen position uncertainty to reflect the large initial error
        ekf.P[0, 0] = 1e6
        ekf.P[3, 3] = 1e6
        true_range = np.sqrt(true_x**2 + true_y**2)
        true_az = np.arctan2(true_y, true_x)
        for _ in range(50):
            ekf.predict()
            z = np.array(
                [
                    true_range + np.random.randn() * 5.0,
                    true_az + np.random.randn() * np.radians(1.0),
                ]
            )
            ekf.update(z)
        assert abs(ekf.position[0] - true_x) < 200.0
        assert abs(ekf.position[1] - true_y) < 200.0
