"""Benchmarks for Kalman filter and tracking operations.

Measures both Python and C++ implementations to quantify speedup.
Run with: pytest tests/benchmarks/ --benchmark-enable -v
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.tracking._accel import has_cpp_acceleration
from sentinel.tracking.filters import (
    KalmanFilter,
    ExtendedKalmanFilter,
    BearingOnlyEKF,
    ConstantAccelerationKF,
    ConstantAccelerationEKF,
)


# ---------------------------------------------------------------
# KalmanFilter (4D linear, camera pixel space)
# ---------------------------------------------------------------

class TestKFPredictBenchmark:
    def test_predict_single(self, benchmark):
        kf = KalmanFilter(dt=0.1)
        kf.x = np.array([1000.0, 50.0, 2000.0, -30.0])
        benchmark(kf.predict)

    def test_predict_100_tracks(self, benchmark):
        rng = np.random.default_rng(42)
        filters = [KalmanFilter(dt=0.1) for _ in range(100)]
        for f in filters:
            f.x = rng.standard_normal(4) * 100

        def run():
            for f in filters:
                f.predict()

        benchmark(run)


class TestKFUpdateBenchmark:
    def test_update_single(self, benchmark):
        kf = KalmanFilter(dt=0.1)
        kf.x = np.array([1000.0, 50.0, 2000.0, -30.0])
        z = np.array([1005.0, 1995.0])
        kf.predict()
        benchmark(kf.update, z)


class TestKFGatingBenchmark:
    def test_gating_single(self, benchmark):
        kf = KalmanFilter(dt=0.1)
        kf.x = np.array([1000.0, 50.0, 2000.0, -30.0])
        z = np.array([1005.0, 1995.0])
        benchmark(kf.gating_distance, z)

    def test_gating_100x50(self, benchmark):
        """100 tracks × 50 detections — the critical inner loop."""
        rng = np.random.default_rng(42)
        filters = []
        for _ in range(100):
            kf = KalmanFilter(dt=0.1)
            kf.x = rng.standard_normal(4) * 100
            kf.P = np.eye(4) * 50.0
            filters.append(kf)
        dets = rng.normal(loc=0, scale=100, size=(50, 2))

        def run():
            for kf in filters:
                for z in dets:
                    kf.gating_distance(z)

        benchmark(run)


# ---------------------------------------------------------------
# ExtendedKalmanFilter (4D, polar measurements)
# ---------------------------------------------------------------

class TestEKFBenchmark:
    def test_predict_single(self, benchmark):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x = np.array([5000.0, 50.0, 3000.0, -20.0])
        benchmark(ekf.predict)

    def test_update_single(self, benchmark):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x = np.array([5000.0, 50.0, 3000.0, -20.0])
        r = np.sqrt(5000**2 + 3000**2)
        az = np.arctan2(3000, 5000)
        z = np.array([r + 5.0, az + 0.01])
        ekf.predict()
        benchmark(ekf.update, z)

    def test_gating_50x30(self, benchmark):
        """50 radar tracks × 30 detections."""
        rng = np.random.default_rng(42)
        filters = []
        for _ in range(50):
            ekf = ExtendedKalmanFilter(dt=0.1)
            ekf.x = np.array([
                rng.uniform(3000, 15000), rng.uniform(-50, 50),
                rng.uniform(-5000, 5000), rng.uniform(-50, 50),
            ])
            filters.append(ekf)
        dets = []
        for _ in range(30):
            r = rng.uniform(3000, 15000)
            az = rng.uniform(-1.0, 1.0)
            dets.append(np.array([r, az]))

        def run():
            for ekf in filters:
                for z in dets:
                    ekf.gating_distance(z)

        benchmark(run)


# ---------------------------------------------------------------
# Physics
# ---------------------------------------------------------------

class TestPhysicsBenchmark:
    def test_radar_snr_5000(self, benchmark):
        from sentinel.sensors.physics import radar_snr
        rng = np.random.default_rng(42)
        rcs = rng.uniform(0.1, 100, 5000)
        ranges = rng.uniform(1000, 100000, 5000)

        def run():
            for r, d in zip(rcs, ranges):
                radar_snr(r, d)

        benchmark(run)

    def test_qi_pd_1000(self, benchmark):
        from sentinel.sensors.physics import qi_practical_pd
        rng = np.random.default_rng(42)
        rcs = rng.uniform(0.1, 100, 1000)
        ranges = rng.uniform(1000, 100000, 1000)

        def run():
            for r, d in zip(rcs, ranges):
                qi_practical_pd(r, d, n_signal=0.01)

        benchmark(run)


# ---------------------------------------------------------------
# Acceleration availability
# ---------------------------------------------------------------

class TestAccelerationInfo:
    def test_report_status(self):
        """Report whether C++ acceleration is available (not a pass/fail test)."""
        status = "ENABLED" if has_cpp_acceleration() else "DISABLED (pure Python)"
        print(f"\n  C++ acceleration: {status}")
        # Always passes — just informational
        assert True
