"""Benchmarks comparing batch operations vs per-element equivalents.

These tests measure the performance of batch cost matrix building vs
individual per-element calls, both with Python fallback. When C++ extension
is compiled, the batch path uses a single boundary crossing instead of T*D.

Run with: pytest tests/benchmarks/test_batch_benchmarks.py -v --benchmark-enable
"""

from __future__ import annotations

import numpy as np
import pytest

import sentinel.tracking.batch_ops as batch_ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kf_data(T, D, n=4, m=2, seed=42):
    """Generate random KF data for benchmarking."""
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n))
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    R = np.eye(m) * 10.0
    states = [rng.standard_normal(n) * 100 for _ in range(T)]
    covs = []
    for _ in range(T):
        A = rng.standard_normal((n, n))
        covs.append(A @ A.T + np.eye(n) * 0.1)
    meas = [rng.standard_normal(m) * 100 for _ in range(D)]
    return states, covs, meas, H, R


def _make_ekf_data(T, D, n=4, m=2, seed=42):
    """Generate random EKF data for benchmarking."""
    rng = np.random.default_rng(seed)
    R = np.diag([100.0, 0.01])
    states = [rng.standard_normal(n) * np.array([5000, 10, 5000, 10]) for _ in range(T)]
    covs = []
    jacs = []
    h_preds = []
    for i in range(T):
        x = states[i]
        A = rng.standard_normal((n, n))
        covs.append(A @ A.T + np.eye(n))
        r = np.sqrt(x[0]**2 + x[2]**2) + 1e-6
        H = np.array([
            [x[0]/r, 0, x[2]/r, 0],
            [-x[2]/r**2, 0, x[0]/r**2, 0],
        ])
        jacs.append(H)
        h_preds.append(np.array([r, np.arctan2(x[2], x[0])]))
    meas = [rng.standard_normal(m) * np.array([5000, 1.0]) for _ in range(D)]
    return states, covs, meas, jacs, h_preds, R


def _make_bbox_data(T, D, seed=42):
    """Generate random bounding box data."""
    rng = np.random.default_rng(seed)
    bboxes_a = np.zeros((T, 4))
    bboxes_b = np.zeros((D, 4))
    for i in range(T):
        x, y = rng.random(2) * 500
        w, h = rng.random(2) * 50 + 10
        bboxes_a[i] = [x, y, x + w, y + h]
    for j in range(D):
        x, y = rng.random(2) * 500
        w, h = rng.random(2) * 50 + 10
        bboxes_b[j] = [x, y, x + w, y + h]
    return bboxes_a, bboxes_b


# ---------------------------------------------------------------------------
# KF cost matrix benchmarks
# ---------------------------------------------------------------------------

class TestKFCostMatrixBenchmarks:
    @pytest.mark.benchmark(group="kf_cost_matrix")
    def test_batch_kf_10x10(self, benchmark):
        """Batch KF cost matrix: 10 tracks x 10 detections."""
        states, covs, meas, H, R = _make_kf_data(10, 10)
        benchmark(batch_ops.batch_kf_cost_matrix, states, covs, meas, H, R, 1e5)

    @pytest.mark.benchmark(group="kf_cost_matrix")
    def test_batch_kf_50x30(self, benchmark):
        """Batch KF cost matrix: 50 tracks x 30 detections."""
        states, covs, meas, H, R = _make_kf_data(50, 30)
        benchmark(batch_ops.batch_kf_cost_matrix, states, covs, meas, H, R, 1e5)

    @pytest.mark.benchmark(group="kf_cost_matrix")
    def test_batch_kf_100x50(self, benchmark):
        """Batch KF cost matrix: 100 tracks x 50 detections."""
        states, covs, meas, H, R = _make_kf_data(100, 50)
        benchmark(batch_ops.batch_kf_cost_matrix, states, covs, meas, H, R, 1e5)

    @pytest.mark.benchmark(group="kf_cost_matrix")
    def test_per_element_kf_100x50(self, benchmark):
        """Per-element KF gating: 100 tracks x 50 detections (baseline)."""
        states, covs, meas, H, R = _make_kf_data(100, 50)

        def per_element():
            T, D = len(states), len(meas)
            cost = np.full((T, D), float("inf"))
            for i in range(T):
                S = H @ covs[i] @ H.T + R
                z_pred = H @ states[i]
                for j in range(D):
                    y = meas[j] - z_pred
                    d = float(y @ np.linalg.solve(S, y))
                    if d <= 1e5:
                        cost[i, j] = d
            return cost

        benchmark(per_element)


# ---------------------------------------------------------------------------
# IoU matrix benchmarks
# ---------------------------------------------------------------------------

class TestIoUMatrixBenchmarks:
    @pytest.mark.benchmark(group="iou_matrix")
    def test_batch_iou_50x30(self, benchmark):
        """Batch IoU matrix: 50 x 30."""
        A, B = _make_bbox_data(50, 30)
        benchmark(batch_ops.batch_iou_matrix, A, B)

    @pytest.mark.benchmark(group="iou_matrix")
    def test_batch_iou_100x50(self, benchmark):
        """Batch IoU matrix: 100 x 50."""
        A, B = _make_bbox_data(100, 50)
        benchmark(batch_ops.batch_iou_matrix, A, B)


# ---------------------------------------------------------------------------
# EKF cost matrix benchmarks
# ---------------------------------------------------------------------------

class TestEKFCostMatrixBenchmarks:
    @pytest.mark.benchmark(group="ekf_cost_matrix")
    def test_batch_ekf_10x10(self, benchmark):
        """Batch EKF cost matrix: 10 x 10."""
        states, covs, meas, jacs, h_preds, R = _make_ekf_data(10, 10)
        benchmark(
            batch_ops.batch_ekf_cost_matrix,
            states, covs, meas, jacs, h_preds, R,
            [1], 1e5,
        )

    @pytest.mark.benchmark(group="ekf_cost_matrix")
    def test_batch_ekf_50x30(self, benchmark):
        """Batch EKF cost matrix: 50 x 30."""
        states, covs, meas, jacs, h_preds, R = _make_ekf_data(50, 30)
        benchmark(
            batch_ops.batch_ekf_cost_matrix,
            states, covs, meas, jacs, h_preds, R,
            [1], 1e5,
        )

    @pytest.mark.benchmark(group="ekf_cost_matrix")
    def test_per_element_ekf_50x30(self, benchmark):
        """Per-element EKF gating: 50 x 30 (baseline)."""
        states, covs, meas, jacs, h_preds, R = _make_ekf_data(50, 30)

        def per_element():
            T, D = len(states), len(meas)
            cost = np.full((T, D), float("inf"))
            for i in range(T):
                S = jacs[i] @ covs[i] @ jacs[i].T + R
                for j in range(D):
                    y = meas[j] - h_preds[i]
                    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
                    d = float(y @ np.linalg.solve(S, y))
                    if d <= 1e5:
                        cost[i, j] = d
            return cost

        benchmark(per_element)


# ---------------------------------------------------------------------------
# Gaussian likelihood benchmarks
# ---------------------------------------------------------------------------

class TestLikelihoodBenchmarks:
    @pytest.mark.benchmark(group="likelihood")
    def test_batch_likelihood_30(self, benchmark):
        """Batch Gaussian likelihood: 30 innovations."""
        rng = np.random.default_rng(42)
        S = np.array([[2.0, 0.5], [0.5, 3.0]])
        innovations = rng.standard_normal((30, 2))
        benchmark(batch_ops.batch_gaussian_likelihood, innovations, S)

    @pytest.mark.benchmark(group="likelihood")
    def test_batch_likelihood_100(self, benchmark):
        """Batch Gaussian likelihood: 100 innovations."""
        rng = np.random.default_rng(42)
        S = np.array([[2.0, 0.5], [0.5, 3.0]])
        innovations = rng.standard_normal((100, 2))
        benchmark(batch_ops.batch_gaussian_likelihood, innovations, S)


# ---------------------------------------------------------------------------
# Batch predict benchmarks
# ---------------------------------------------------------------------------

class TestPredictBenchmarks:
    @pytest.mark.benchmark(group="predict")
    def test_batch_predict_50(self, benchmark):
        """Batch KF predict: 50 tracks."""
        states, covs, _, H, _ = _make_kf_data(50, 1)
        dt = 0.1
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        benchmark(batch_ops.batch_kf_predict, states, covs, F, Q)

    @pytest.mark.benchmark(group="predict")
    def test_per_element_predict_50(self, benchmark):
        """Per-element KF predict: 50 tracks (baseline)."""
        states, covs, _, H, _ = _make_kf_data(50, 1)
        dt = 0.1
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01

        def per_element():
            results = []
            for i in range(len(states)):
                x_new = F @ states[i]
                P_new = F @ covs[i] @ F.T + Q
                results.append((x_new, P_new))
            return results

        benchmark(per_element)
