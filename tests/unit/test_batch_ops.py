"""Tests for batch operations (Python fallback path).

Verifies that batch operations produce identical results to per-element
implementations. These tests always exercise the pure-Python fallback
(by monkeypatching _HAS_CPP_BATCH = False), ensuring correctness
independent of C++ availability.
"""

from __future__ import annotations

import numpy as np
import pytest

import sentinel.tracking.batch_ops as batch_ops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def force_python_fallback(monkeypatch):
    """Force pure-Python path for all tests in this module."""
    monkeypatch.setattr(batch_ops, "_HAS_CPP_BATCH", False)


def _kf_state(dim=4):
    """Random KF state vector."""
    return np.random.randn(dim)


def _kf_cov(dim=4):
    """Random PSD covariance matrix."""
    A = np.random.randn(dim, dim)
    return A @ A.T + np.eye(dim) * 0.1


def _kf_H(m=2, n=4):
    """Simple observation matrix."""
    H = np.zeros((m, n))
    H[0, 0] = 1.0  # observe x
    H[1, 2] = 1.0  # observe y
    return H


def _kf_R(m=2):
    """Measurement noise."""
    return np.eye(m) * 10.0


def _bbox(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float64)


# ---------------------------------------------------------------------------
# batch_kf_cost_matrix
# ---------------------------------------------------------------------------

class TestBatchKFCostMatrix:
    def test_matches_per_element(self):
        """Batch result should match per-element gating_distance."""
        np.random.seed(42)
        T, D = 5, 8
        H = _kf_H()
        R = _kf_R()
        states = [_kf_state() for _ in range(T)]
        covs = [_kf_cov() for _ in range(T)]
        meas = [np.random.randn(2) * 100 for _ in range(D)]

        cost = batch_ops.batch_kf_cost_matrix(states, covs, meas, H, R, gate=1e5)

        # Compare with manual per-element computation
        for i in range(T):
            S = H @ covs[i] @ H.T + R
            S_inv = np.linalg.inv(S)
            z_pred = H @ states[i]
            for j in range(D):
                y = meas[j] - z_pred
                expected = float(y @ S_inv @ y)
                assert cost[i, j] == pytest.approx(expected, abs=1e-10)

    def test_gating(self):
        """Entries exceeding gate should be +inf."""
        np.random.seed(42)
        H = _kf_H()
        R = _kf_R()
        states = [np.zeros(4)]
        covs = [np.eye(4)]
        meas = [np.array([1000.0, 1000.0])]  # Very far away

        cost = batch_ops.batch_kf_cost_matrix(states, covs, meas, H, R, gate=1.0)
        assert cost[0, 0] == float("inf")

    def test_empty_tracks(self):
        H = _kf_H()
        R = _kf_R()
        cost = batch_ops.batch_kf_cost_matrix([], [], [np.zeros(2)], H, R)
        assert cost.shape == (0, 1)

    def test_empty_detections(self):
        H = _kf_H()
        R = _kf_R()
        cost = batch_ops.batch_kf_cost_matrix([np.zeros(4)], [np.eye(4)], [], H, R)
        assert cost.shape == (1, 0)

    def test_single_track_single_det(self):
        H = _kf_H()
        R = _kf_R()
        x = np.array([10.0, 0.0, 20.0, 0.0])
        P = np.eye(4)
        z = np.array([10.0, 20.0])  # Perfect match

        cost = batch_ops.batch_kf_cost_matrix([x], [P], [z], H, R, gate=1e5)
        assert cost.shape == (1, 1)
        assert cost[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_shape(self):
        np.random.seed(42)
        T, D = 10, 20
        H = _kf_H()
        R = _kf_R()
        states = [_kf_state() for _ in range(T)]
        covs = [_kf_cov() for _ in range(T)]
        meas = [np.random.randn(2) for _ in range(D)]
        cost = batch_ops.batch_kf_cost_matrix(states, covs, meas, H, R)
        assert cost.shape == (T, D)


# ---------------------------------------------------------------------------
# batch_iou_matrix
# ---------------------------------------------------------------------------

class TestBatchIoUMatrix:
    def test_matches_per_element(self):
        """Batch IoU should match single-pair computation."""
        from sentinel.tracking.cost_functions import iou_bbox

        A = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30],
        ], dtype=np.float64)
        B = np.array([
            [0, 0, 10, 10],
            [7, 7, 17, 17],
        ], dtype=np.float64)

        iou = batch_ops.batch_iou_matrix(A, B)

        assert iou.shape == (3, 2)
        for i in range(3):
            for j in range(2):
                expected = iou_bbox(A[i], B[j])
                assert iou[i, j] == pytest.approx(expected, abs=1e-10)

    def test_no_overlap(self):
        A = np.array([[0, 0, 5, 5]], dtype=np.float64)
        B = np.array([[10, 10, 15, 15]], dtype=np.float64)
        iou = batch_ops.batch_iou_matrix(A, B)
        assert iou[0, 0] == 0.0

    def test_perfect_overlap(self):
        A = np.array([[0, 0, 10, 10]], dtype=np.float64)
        B = np.array([[0, 0, 10, 10]], dtype=np.float64)
        iou = batch_ops.batch_iou_matrix(A, B)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_empty(self):
        A = np.zeros((0, 4))
        B = np.array([[0, 0, 10, 10]], dtype=np.float64)
        iou = batch_ops.batch_iou_matrix(A, B)
        assert iou.shape == (0, 1)


# ---------------------------------------------------------------------------
# batch_camera_cost_matrix
# ---------------------------------------------------------------------------

class TestBatchCameraCostMatrix:
    def test_matches_manual(self):
        """Combined cost should match alpha*maha + (1-alpha)*(1-IoU)."""
        np.random.seed(42)
        H = _kf_H()
        R = _kf_R()
        alpha = 0.5

        states = [np.array([5.0, 0.0, 5.0, 0.0])]
        covs = [np.eye(4)]
        meas = [np.array([5.0, 5.0])]
        bboxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        bboxes_b = np.array([[2, 2, 12, 12]], dtype=np.float64)

        cost = batch_ops.batch_camera_cost_matrix(
            states, covs, meas, H, R, bboxes_a, bboxes_b, alpha, gate=1e5
        )
        maha = batch_ops.batch_kf_cost_matrix(states, covs, meas, H, R, gate=1e5)
        iou = batch_ops.batch_iou_matrix(bboxes_a, bboxes_b)

        expected = alpha * maha[0, 0] + (1 - alpha) * (1 - iou[0, 0])
        assert cost[0, 0] == pytest.approx(expected, abs=1e-10)

    def test_gated_entry(self):
        H = _kf_H()
        R = _kf_R()
        states = [np.zeros(4)]
        covs = [np.eye(4)]
        meas = [np.array([1000.0, 1000.0])]
        bboxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        bboxes_b = np.array([[0, 0, 10, 10]], dtype=np.float64)

        cost = batch_ops.batch_camera_cost_matrix(
            states, covs, meas, H, R, bboxes_a, bboxes_b, 0.5, gate=1.0
        )
        assert cost[0, 0] == float("inf")


# ---------------------------------------------------------------------------
# batch_ekf_cost_matrix
# ---------------------------------------------------------------------------

class TestBatchEKFCostMatrix:
    def test_matches_per_element_polar(self):
        """EKF batch with polar Jacobians should match per-element."""
        np.random.seed(42)
        T, D = 3, 5

        states = []
        covs = []
        jacs = []
        h_preds = []
        for _ in range(T):
            x = np.random.randn(4) * np.array([1000, 10, 1000, 10])
            states.append(x)
            covs.append(_kf_cov())
            # Polar Jacobian: range and azimuth
            r = np.sqrt(x[0]**2 + x[2]**2) + 1e-6
            H = np.array([
                [x[0]/r, 0, x[2]/r, 0],
                [-x[2]/r**2, 0, x[0]/r**2, 0],
            ])
            jacs.append(H)
            h_preds.append(np.array([r, np.arctan2(x[2], x[0])]))

        R = np.diag([100.0, 0.01])
        meas = [np.random.randn(2) * np.array([5000, 1.0]) for _ in range(D)]

        cost = batch_ops.batch_ekf_cost_matrix(
            states, covs, meas, jacs, h_preds, R,
            angular_indices=[1], gate=1e5
        )

        # Per-element verification
        for i in range(T):
            S = jacs[i] @ covs[i] @ jacs[i].T + R
            S_inv = np.linalg.inv(S)
            for j in range(D):
                y = meas[j] - h_preds[i]
                y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
                expected = float(y @ S_inv @ y)
                if expected <= 1e5:
                    assert cost[i, j] == pytest.approx(expected, abs=1e-8)
                else:
                    assert cost[i, j] == float("inf")

    def test_angular_wrapping(self):
        """Innovations near +-pi should be wrapped correctly."""
        x = np.array([1000.0, 0.0, 0.0, 0.0])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        P = np.eye(4) * 100
        R = np.eye(2)

        # Measurement with azimuth near pi
        z_pred = np.array([1000.0, 0.0])
        z = np.array([1000.0, 3.1])  # Close to pi
        z2 = np.array([1000.0, -3.1])  # Close to -pi, should be close to z after wrapping

        cost = batch_ops.batch_ekf_cost_matrix(
            [x], [P], [z, z2], [H], [z_pred], R,
            angular_indices=[1], gate=1e5
        )
        # z and z2 are close modulo 2*pi wrapping
        assert cost.shape == (1, 2)
        # Both should have similar Mahalanobis distances
        assert abs(cost[0, 0] - cost[0, 1]) < 0.5

    def test_empty(self):
        R = np.eye(2)
        cost = batch_ops.batch_ekf_cost_matrix([], [], [], [], [], R)
        assert cost.shape == (0, 0)

    def test_gating(self):
        x = np.array([0.0, 0.0, 0.0, 0.0])
        P = np.eye(4) * 0.01  # Very tight covariance
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.01
        z_pred = np.array([0.0, 0.0])
        z = np.array([1000.0, 1000.0])  # Very far

        cost = batch_ops.batch_ekf_cost_matrix(
            [x], [P], [z], [H], [z_pred], R, gate=10.0
        )
        assert cost[0, 0] == float("inf")


# ---------------------------------------------------------------------------
# batch_gaussian_likelihood
# ---------------------------------------------------------------------------

class TestBatchGaussianLikelihood:
    def test_matches_per_element(self):
        """Batch likelihood should match per-element _gaussian_likelihood."""
        from sentinel.tracking.jpda import _gaussian_likelihood

        np.random.seed(42)
        S = np.array([[2.0, 0.5], [0.5, 3.0]])
        innovations = np.random.randn(10, 2)

        result = batch_ops.batch_gaussian_likelihood(innovations, S)

        for i in range(10):
            expected = _gaussian_likelihood(innovations[i], S)
            assert result[i] == pytest.approx(expected, rel=1e-10)

    def test_zero_innovation(self):
        """Zero innovation should give maximum likelihood."""
        S = np.eye(2)
        innovations = np.array([[0.0, 0.0]])
        result = batch_ops.batch_gaussian_likelihood(innovations, S)
        expected = 1.0 / (2 * np.pi)  # (2*pi)^{-d/2} * |I|^{-1/2}
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_empty(self):
        S = np.eye(2)
        result = batch_ops.batch_gaussian_likelihood(np.zeros((0, 2)), S)
        assert len(result) == 0

    def test_singular_S(self):
        """Singular covariance should return zeros."""
        S = np.zeros((2, 2))
        innovations = np.array([[1.0, 2.0]])
        result = batch_ops.batch_gaussian_likelihood(innovations, S)
        assert result[0] == 0.0

    def test_shape(self):
        np.random.seed(42)
        S = np.eye(3) * 5.0
        innovations = np.random.randn(20, 3)
        result = batch_ops.batch_gaussian_likelihood(innovations, S)
        assert result.shape == (20,)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# batch_kf_predict
# ---------------------------------------------------------------------------

class TestBatchKFPredict:
    def test_matches_per_element(self):
        """Batch predict should match individual KF predict."""
        np.random.seed(42)
        T = 5
        dt = 0.1
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])
        Q = np.eye(4) * 0.01

        states = [_kf_state() for _ in range(T)]
        covs = [_kf_cov() for _ in range(T)]

        s_pred, p_pred = batch_ops.batch_kf_predict(states, covs, F, Q)

        for i in range(T):
            expected_x = F @ states[i]
            expected_P = F @ covs[i] @ F.T + Q
            np.testing.assert_allclose(s_pred[i], expected_x, atol=1e-12)
            np.testing.assert_allclose(p_pred[i], expected_P, atol=1e-12)

    def test_empty(self):
        F = np.eye(4)
        Q = np.eye(4) * 0.01
        s_pred, p_pred = batch_ops.batch_kf_predict([], [], F, Q)
        assert len(s_pred) == 0
        assert len(p_pred) == 0

    def test_single_track(self):
        F = np.eye(4)
        Q = np.zeros((4, 4))
        x = np.array([1.0, 2.0, 3.0, 4.0])
        P = np.eye(4)

        s_pred, p_pred = batch_ops.batch_kf_predict([x], [P], F, Q)
        np.testing.assert_allclose(s_pred[0], x, atol=1e-12)
        np.testing.assert_allclose(p_pred[0], P, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry / properties
# ---------------------------------------------------------------------------

class TestBatchProperties:
    def test_kf_cost_non_negative(self):
        """Mahalanobis distance is always non-negative."""
        np.random.seed(42)
        H = _kf_H()
        R = _kf_R()
        states = [_kf_state() for _ in range(5)]
        covs = [_kf_cov() for _ in range(5)]
        meas = [np.random.randn(2) for _ in range(10)]
        cost = batch_ops.batch_kf_cost_matrix(states, covs, meas, H, R)
        assert np.all((cost >= 0) | np.isinf(cost))

    def test_iou_range(self):
        """IoU should be in [0, 1]."""
        np.random.seed(42)
        A = np.sort(np.random.rand(10, 4) * 100, axis=1).reshape(10, 4)
        B = np.sort(np.random.rand(8, 4) * 100, axis=1).reshape(8, 4)
        # Ensure x2 > x1, y2 > y1
        A[:, 2] = A[:, 0] + np.abs(A[:, 2] - A[:, 0]) + 1
        A[:, 3] = A[:, 1] + np.abs(A[:, 3] - A[:, 1]) + 1
        B[:, 2] = B[:, 0] + np.abs(B[:, 2] - B[:, 0]) + 1
        B[:, 3] = B[:, 1] + np.abs(B[:, 3] - B[:, 1]) + 1

        iou = batch_ops.batch_iou_matrix(A, B)
        assert np.all(iou >= 0)
        assert np.all(iou <= 1.0 + 1e-10)

    def test_likelihood_non_negative(self):
        """Gaussian likelihood should be non-negative."""
        np.random.seed(42)
        S = _kf_cov(2)
        innovations = np.random.randn(20, 2)
        L = batch_ops.batch_gaussian_likelihood(innovations, S)
        assert np.all(L >= 0)

    def test_predict_preserves_symmetry(self):
        """Predicted covariance should remain symmetric."""
        np.random.seed(42)
        F = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        states = [_kf_state() for _ in range(3)]
        covs = [_kf_cov() for _ in range(3)]

        _, p_pred = batch_ops.batch_kf_predict(states, covs, F, Q)
        for P in p_pred:
            np.testing.assert_allclose(P, P.T, atol=1e-12)
