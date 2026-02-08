"""Tests for the Interacting Multiple Model (IMM) filter."""

import numpy as np
import pytest

from sentinel.tracking.imm import IMMFilter


class TestIMMInit:
    """Test IMM initialization and basic properties."""

    def test_init_camera_mode(self):
        imm = IMMFilter(dt=0.033, mode="camera")
        assert imm.dim_state == 4
        assert imm.dim_meas == 2
        assert len(imm._filters) == 2
        assert imm.mu[0] == pytest.approx(0.9)
        assert imm.mu[1] == pytest.approx(0.1)

    def test_init_radar_mode(self):
        imm = IMMFilter(dt=0.1, mode="radar")
        assert imm.dim_state == 4
        assert imm.dim_meas == 2

    def test_initial_mode_probs(self):
        imm = IMMFilter()
        assert imm.mode_probabilities.sum() == pytest.approx(1.0)
        assert not imm.is_maneuvering  # initially CV-favored

    def test_transition_matrix(self):
        imm = IMMFilter(transition_prob=0.95)
        assert imm.TPM[0, 0] == pytest.approx(0.95)
        assert imm.TPM[0, 1] == pytest.approx(0.05)
        assert imm.TPM[1, 0] == pytest.approx(0.05)
        assert imm.TPM[1, 1] == pytest.approx(0.95)


class TestIMMStateDimConversion:
    """Test state vector expansion/contraction between 4D and 6D."""

    def test_expand_4_to_6(self):
        x4 = np.array([1.0, 2.0, 3.0, 4.0])
        x6 = IMMFilter._expand_state(x4, 4, 6)
        np.testing.assert_allclose(x6, [1.0, 2.0, 0.0, 3.0, 4.0, 0.0])

    def test_contract_6_to_4(self):
        x6 = np.array([1.0, 2.0, 0.5, 3.0, 4.0, -0.5])
        x4 = IMMFilter._expand_state(x6, 6, 4)
        np.testing.assert_allclose(x4, [1.0, 2.0, 3.0, 4.0])

    def test_same_dim_returns_copy(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = IMMFilter._expand_state(x, 4, 4)
        np.testing.assert_allclose(result, x)
        # Should be a copy, not same object
        result[0] = 999
        assert x[0] == 1.0

    def test_expand_covariance_4_to_6(self):
        P4 = np.eye(4) * 10.0
        P6 = IMMFilter._expand_covariance(P4, 4, 6)
        assert P6.shape == (6, 6)
        # Position and velocity entries should transfer
        assert P6[0, 0] == pytest.approx(10.0)
        assert P6[1, 1] == pytest.approx(10.0)
        assert P6[3, 3] == pytest.approx(10.0)
        assert P6[4, 4] == pytest.approx(10.0)
        # Acceleration diagonals filled with default uncertainty
        assert P6[2, 2] == pytest.approx(100.0)
        assert P6[5, 5] == pytest.approx(100.0)

    def test_contract_covariance_6_to_4(self):
        P6 = np.eye(6) * 10.0
        P4 = IMMFilter._expand_covariance(P6, 6, 4)
        assert P4.shape == (4, 4)
        assert P4[0, 0] == pytest.approx(10.0)


class TestIMMPredictUpdate:
    """Test IMM predict and update cycles."""

    def test_predict_returns_state(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm.x = np.array([100.0, 1.0, 200.0, 2.0])
        x = imm.predict()
        assert x.shape == (4,)

    def test_update_returns_state(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm.x = np.array([100.0, 1.0, 200.0, 2.0])
        imm.predict()
        x = imm.update(np.array([101.0, 202.0]))
        assert x.shape == (4,)

    def test_position_velocity(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm.x = np.array([100.0, 10.0, 200.0, 20.0])
        imm._combine()  # Need to recombine after setting state
        pos = imm.position
        vel = imm.velocity
        np.testing.assert_allclose(pos, [100.0, 200.0], atol=1.0)
        np.testing.assert_allclose(vel, [10.0, 20.0], atol=1.0)

    def test_mode_probs_sum_to_one(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm.x = np.array([100.0, 1.0, 200.0, 2.0])
        for _ in range(10):
            imm.predict()
            imm.update(np.array([100.0 + np.random.randn(), 200.0 + np.random.randn()]))
        assert imm.mode_probabilities.sum() == pytest.approx(1.0, abs=1e-6)


class TestIMMManeuverDetection:
    """Test that IMM detects maneuvering targets."""

    def test_constant_velocity_stays_cv(self):
        """Non-maneuvering target should keep CV dominant."""
        imm = IMMFilter(dt=0.1, mode="camera", transition_prob=0.98)
        # Target moving at constant velocity
        pos = np.array([100.0, 200.0])
        vel = np.array([5.0, 3.0])
        imm.x = np.array([pos[0], vel[0], pos[1], vel[1]])

        for i in range(50):
            pos = pos + vel * 0.1
            imm.predict()
            z = pos + np.random.randn(2) * 1.0
            imm.update(z)

        # CV should still dominate
        assert imm.mu[0] > 0.5  # CV probability
        assert not imm.is_maneuvering

    def test_accelerating_target_switches_to_ca(self):
        """Maneuvering target should shift probability toward CA."""
        imm = IMMFilter(dt=0.1, mode="camera", transition_prob=0.95)
        pos = np.array([100.0, 200.0])
        vel = np.array([5.0, 3.0])
        imm.x = np.array([pos[0], vel[0], pos[1], vel[1]])

        # Phase 1: constant velocity (10 steps)
        for _ in range(10):
            pos = pos + vel * 0.1
            imm.predict()
            imm.update(pos + np.random.randn(2) * 1.0)

        # Phase 2: sudden acceleration (20 steps)
        accel = np.array([20.0, -15.0])
        for _ in range(20):
            vel = vel + accel * 0.1
            pos = pos + vel * 0.1
            imm.predict()
            imm.update(pos + np.random.randn(2) * 1.0)

        # CA probability should increase
        assert imm.mu[1] > 0.3  # CA should gain weight


class TestIMMRadarMode:
    """Test IMM with radar (polar) measurements."""

    def test_radar_predict_update(self):
        imm = IMMFilter(dt=0.1, mode="radar")
        # Initialize at range=5000, azimuth ~0.9 rad
        imm.x = np.array([3000.0, -10.0, 4000.0, 5.0])
        for _ in range(10):
            imm.predict()
            r = np.sqrt(imm.position[0]**2 + imm.position[1]**2)
            az = np.arctan2(imm.position[1], imm.position[0])
            z = np.array([r + np.random.randn() * 5.0, az + np.random.randn() * 0.01])
            imm.update(z)
        assert imm.mode_probabilities.sum() == pytest.approx(1.0, abs=1e-6)

    def test_gating_distance(self):
        imm = IMMFilter(dt=0.1, mode="radar")
        imm.x = np.array([3000.0, 0.0, 4000.0, 0.0])
        # Measure at predicted location
        z = imm.predicted_measurement
        dist = imm.gating_distance(z)
        assert dist < 1.0


class TestIMMStateInterface:
    """Test state getter/setter for backward compatibility."""

    def test_x_getter(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm._filters[0].x = np.array([1.0, 2.0, 3.0, 4.0])
        imm._combine()
        x = imm.x
        assert x.shape == (4,)

    def test_x_setter_4d(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        imm.x = np.array([10.0, 1.0, 20.0, 2.0])
        # Both filters should get updated
        assert imm._filters[0].x[0] == pytest.approx(10.0)
        assert imm._filters[1].x[0] == pytest.approx(10.0)

    def test_P_getter(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        P = imm.P
        assert P.shape == (4, 4)

    def test_P_setter(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        P_new = np.eye(4) * 50.0
        imm.P = P_new
        np.testing.assert_allclose(imm._filters[0].P, P_new)

    def test_set_process_noise(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        # Should not raise
        imm.set_process_noise_std(2.0)

    def test_R_property(self):
        imm = IMMFilter(dt=0.1, mode="camera")
        R = imm.R
        assert R.shape == (2, 2)
