"""Tests for Kalman filter implementation."""

import numpy as np

from sentinel.tracking.filters import KalmanFilter


class TestKalmanFilter:
    def test_init_state(self):
        kf = KalmanFilter()
        assert kf.dim_state == 4
        assert kf.dim_meas == 2
        np.testing.assert_array_equal(kf.x, np.zeros(4))

    def test_predict_constant_velocity(self):
        kf = KalmanFilter(dt=1.0)
        kf.x = np.array([0, 10, 0, 5])  # pos=(0,0), vel=(10,5)
        kf.P = np.eye(4) * 0.1  # Low uncertainty
        predicted = kf.predict()
        # After 1s: pos should be (10, 5)
        assert abs(predicted[0] - 10) < 1
        assert abs(predicted[2] - 5) < 1

    def test_predict_increases_uncertainty(self):
        kf = KalmanFilter()
        P_before = kf.P.copy()
        kf.predict()
        # Covariance should increase (more uncertainty after prediction)
        assert np.trace(kf.P) >= np.trace(P_before)

    def test_update_reduces_uncertainty(self):
        kf = KalmanFilter()
        kf.predict()
        P_before = kf.P.copy()
        kf.update(np.array([100, 200]))
        # Covariance should decrease after measurement
        assert np.trace(kf.P) < np.trace(P_before)

    def test_update_moves_state_toward_measurement(self):
        kf = KalmanFilter()
        kf.x = np.array([0, 0, 0, 0])
        kf.predict()
        z = np.array([100, 200])
        kf.update(z)
        # State should move toward measurement
        assert kf.x[0] > 0  # x moved toward 100
        assert kf.x[2] > 0  # y moved toward 200

    def test_convergence_on_static_target(self):
        """KF should converge on a stationary target with repeated measurements."""
        kf = KalmanFilter(dt=1 / 30)
        true_pos = np.array([500, 300])

        for _ in range(50):
            kf.predict()
            # Noisy measurement
            noise = np.random.randn(2) * 5
            kf.update(true_pos + noise)

        # Should be close to true position
        np.testing.assert_allclose(kf.position, true_pos, atol=20)

    def test_convergence_on_moving_target(self):
        """KF should track a constant-velocity target."""
        kf = KalmanFilter(dt=0.1)
        true_vel = np.array([50, 30])  # pixels per second

        for i in range(100):
            t = i * 0.1
            true_pos = np.array([100 + true_vel[0] * t, 200 + true_vel[1] * t])
            kf.predict()
            noise = np.random.randn(2) * 3
            kf.update(true_pos + noise)

        # Position should be close
        final_true = np.array([100 + true_vel[0] * 10, 200 + true_vel[1] * 10])
        np.testing.assert_allclose(kf.position, final_true, atol=30)
        # Velocity in state is pixels/second (F matrix: x += vx * dt)
        # So velocity estimates should be close to true_vel
        np.testing.assert_allclose(kf.velocity, true_vel, atol=10)

    def test_gating_distance_accepts_close(self):
        kf = KalmanFilter()
        kf.x = np.array([100, 0, 200, 0])
        kf.P = np.eye(4) * 10
        # Measurement close to predicted -> low distance
        dist = kf.gating_distance(np.array([102, 198]))
        assert dist < 9.21  # Within 99% gate

    def test_gating_distance_rejects_far(self):
        kf = KalmanFilter()
        kf.x = np.array([100, 0, 200, 0])
        kf.P = np.eye(4) * 1.0
        kf.R = np.eye(2) * 1.0
        # Measurement very far from predicted -> high distance
        dist = kf.gating_distance(np.array([500, 500]))
        assert dist > 9.21

    def test_position_property(self):
        kf = KalmanFilter()
        kf.x = np.array([10, 1, 20, 2])
        np.testing.assert_array_equal(kf.position, [10, 20])

    def test_velocity_property(self):
        kf = KalmanFilter()
        kf.x = np.array([10, 1, 20, 2])
        np.testing.assert_array_equal(kf.velocity, [1, 2])

    def test_set_noise(self):
        kf = KalmanFilter()
        kf.set_process_noise_std(10.0)
        assert kf.Q[0, 0] > 0
        kf.set_measurement_noise_std(20.0)
        assert kf.R[0, 0] == 400.0

    def test_covariance_stays_symmetric(self):
        """Joseph form should maintain symmetry."""
        kf = KalmanFilter(dt=1 / 30)
        for _ in range(20):
            kf.predict()
            kf.update(np.random.randn(2) * 100)
            # Check symmetry
            np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)

    def test_covariance_stays_positive_definite(self):
        kf = KalmanFilter(dt=1 / 30)
        for _ in range(50):
            kf.predict()
            kf.update(np.random.randn(2) * 100)
            eigvals = np.linalg.eigvalsh(kf.P)
            assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals}"
