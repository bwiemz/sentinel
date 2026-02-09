"""Scenario: Deterministic Time Verification.

Verifies that ScenarioRunner produces identical results across multiple runs
when using the same SimClock and seed. This validates the Phase 14 deterministic
simulation infrastructure.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.clock import SimClock
from sentinel.core.types import TargetType

from .conftest import ScenarioRunner, ScenarioTarget


def _standard_target() -> list[ScenarioTarget]:
    """A conventional target at moderate range for determinism checks."""
    return [
        ScenarioTarget(
            target_id="DET-1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([-100.0, 50.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=10.0,
            mach=0.5,
            expected_threat="MEDIUM",
        ),
    ]


def _two_target_scenario() -> list[ScenarioTarget]:
    """Two targets for multi-track determinism."""
    return [
        ScenarioTarget(
            target_id="DET-A",
            position=np.array([5000.0, 2000.0]),
            velocity=np.array([-80.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=15.0,
            mach=0.4,
            expected_threat="MEDIUM",
        ),
        ScenarioTarget(
            target_id="DET-B",
            position=np.array([12000.0, -3000.0]),
            velocity=np.array([-200.0, 100.0]),
            target_type=TargetType.HYPERSONIC,
            rcs_dbsm=5.0,
            mach=5.0,
            expected_threat="HIGH",
            expected_hypersonic=True,
        ),
    ]


class TestDeterministicScenarioRunner:
    """Two identical ScenarioRunner runs must produce byte-identical results."""

    def test_identical_detection_counts(self):
        """Same seed + same SimClock = identical detection counts per step."""
        logs = []
        for _ in range(2):
            runner = ScenarioRunner(
                _standard_target(), seed=42, n_steps=10, step_dt=0.1,
            )
            result = runner.run()
            logs.append([len(dets) for dets in result.multifreq_detection_log])
        assert logs[0] == logs[1]

    def test_identical_detection_values(self):
        """Detection range/azimuth/velocity values match exactly across runs."""
        results = []
        for _ in range(2):
            runner = ScenarioRunner(
                _standard_target(), seed=42, n_steps=10, step_dt=0.1,
            )
            results.append(runner.run())

        for step_idx in range(10):
            dets_a = results[0].multifreq_detection_log[step_idx]
            dets_b = results[1].multifreq_detection_log[step_idx]
            assert len(dets_a) == len(dets_b), f"Step {step_idx}: count mismatch"
            for da, db in zip(dets_a, dets_b):
                assert da.range_m == db.range_m
                assert da.azimuth_deg == db.azimuth_deg
                assert da.velocity_mps == db.velocity_mps

    def test_identical_thermal_detections(self):
        """Thermal detection logs are identical across runs."""
        results = []
        for _ in range(2):
            runner = ScenarioRunner(
                _standard_target(), seed=42, n_steps=10, step_dt=0.1,
            )
            results.append(runner.run())

        for step_idx in range(10):
            dets_a = results[0].thermal_detection_log[step_idx]
            dets_b = results[1].thermal_detection_log[step_idx]
            assert len(dets_a) == len(dets_b)
            for da, db in zip(dets_a, dets_b):
                assert da.azimuth_deg == db.azimuth_deg
                assert da.temperature_k == db.temperature_k

    def test_identical_track_counts(self):
        """Confirmed track counts are identical across runs."""
        counts = []
        for _ in range(2):
            runner = ScenarioRunner(
                _two_target_scenario(), seed=42, n_steps=15, step_dt=0.1,
            )
            result = runner.run()
            counts.append((
                result.radar_confirmed_count,
                result.thermal_confirmed_count,
            ))
        assert counts[0] == counts[1]

    def test_identical_fused_track_count(self):
        """Fused track count is identical across runs."""
        fused_counts = []
        for _ in range(2):
            runner = ScenarioRunner(
                _two_target_scenario(), seed=42, n_steps=15, step_dt=0.1,
            )
            result = runner.run()
            fused_counts.append(len(result.fused_tracks))
        assert fused_counts[0] == fused_counts[1]

    def test_different_step_dt_changes_positions(self):
        """Different step_dt should produce different target positions."""
        results = []
        for dt in [0.1, 0.5]:
            runner = ScenarioRunner(
                _standard_target(), seed=42, n_steps=10, step_dt=dt,
            )
            results.append(runner.run())

        # With different dt, the target moves different distances per step.
        # Last-step detections should differ (if both have detections).
        last_a = results[0].multifreq_detection_log[-1]
        last_b = results[1].multifreq_detection_log[-1]
        if last_a and last_b:
            # At least one detection should have different range
            ranges_a = sorted(d.range_m for d in last_a)
            ranges_b = sorted(d.range_m for d in last_b)
            assert ranges_a != ranges_b, "Different step_dt should yield different positions"

    def test_explicit_sim_clock_injection(self):
        """Passing an explicit SimClock produces deterministic results."""
        results = []
        for _ in range(2):
            clock = SimClock(start_epoch=5000.0)
            runner = ScenarioRunner(
                _standard_target(), seed=42, n_steps=10,
                clock=clock, step_dt=0.1,
            )
            results.append(runner.run())

        for step_idx in range(10):
            dets_a = results[0].multifreq_detection_log[step_idx]
            dets_b = results[1].multifreq_detection_log[step_idx]
            assert len(dets_a) == len(dets_b)
            for da, db in zip(dets_a, dets_b):
                assert da.timestamp == db.timestamp

    def test_sim_clock_timestamps_advance_correctly(self):
        """Frame timestamps should advance by step_dt each step."""
        clock = SimClock(start_epoch=2000.0)
        runner = ScenarioRunner(
            _standard_target(), seed=42, n_steps=5,
            clock=clock, step_dt=0.25,
        )
        result = runner.run()

        # Each step advances by 0.25s from epoch 2000.0
        for i, dets in enumerate(result.multifreq_detection_log):
            expected_ts = 2000.0 + (i + 1) * 0.25  # clock steps before read_frame
            if dets:
                assert dets[0].timestamp == expected_ts
