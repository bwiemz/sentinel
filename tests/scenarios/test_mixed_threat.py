"""Scenario 4: Mixed Threat Environment.

Simultaneous engagement with 1 stealth aircraft (HIGH), 1 hypersonic
missile (CRITICAL), and 2 conventional targets (MEDIUM/LOW).
Tests that the system correctly classifies each threat type differently
in a complex multi-target environment.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.types import TargetType

from .conftest import ScenarioRunner, ScenarioTarget


def _mixed_targets() -> list[ScenarioTarget]:
    return [
        ScenarioTarget(
            target_id="MIX-STEALTH",
            position=np.array([10000.0, 3000.0]),
            velocity=np.array([-100.0, -30.0]),
            target_type=TargetType.STEALTH,
            rcs_dbsm=-20.0,
            mach=0.85,
            expected_threat="HIGH",
            expected_stealth=True,
        ),
        ScenarioTarget(
            target_id="MIX-HYPER",
            position=np.array([12000.0, -4000.0]),
            velocity=np.array([-1600.0, 500.0]),
            target_type=TargetType.HYPERSONIC,
            rcs_dbsm=5.0,
            mach=5.0,
            expected_threat="CRITICAL",
            expected_hypersonic=True,
        ),
        ScenarioTarget(
            target_id="MIX-CONV-1",
            position=np.array([6000.0, 1000.0]),
            velocity=np.array([-20.0, 5.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=12.0,
            mach=0.7,
            expected_threat="MEDIUM",
        ),
        ScenarioTarget(
            target_id="MIX-CONV-2",
            position=np.array([7000.0, -2000.0]),
            velocity=np.array([-30.0, 10.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=10.0,
            mach=0.8,
            expected_threat="MEDIUM",
        ),
    ]


class TestMixedThreat:
    """Validate correct classification across different threat types simultaneously."""

    def test_all_targets_tracked(self):
        """Radar should confirm at least 3 of 4 targets."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        assert result.radar_confirmed_count >= 3, (
            f"Expected >= 3 confirmed radar tracks, got {result.radar_confirmed_count}"
        )

    def test_threat_level_diversity(self):
        """Fused tracks should span at least 2 different threat levels."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        threat_levels = {ft.threat_level for ft in result.fused_tracks}
        assert len(threat_levels) >= 2, (
            f"Expected at least 2 threat levels, got {threat_levels}"
        )

    def test_has_critical_threat(self):
        """At least one fused track should be CRITICAL (the hypersonic)."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        critical = [ft for ft in result.fused_tracks if ft.threat_level == "CRITICAL"]
        assert len(critical) >= 1, (
            f"Expected CRITICAL threat, got: "
            f"{[ft.threat_level for ft in result.fused_tracks]}"
        )

    def test_has_high_or_critical_for_stealth(self):
        """At least one fused track should be HIGH or CRITICAL (stealth candidate)."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        high_crit = [
            ft for ft in result.fused_tracks
            if ft.threat_level in ("HIGH", "CRITICAL")
        ]
        assert len(high_crit) >= 1, (
            f"Expected HIGH/CRITICAL threat, got: "
            f"{[ft.threat_level for ft in result.fused_tracks]}"
        )

    def test_has_medium_or_low(self):
        """At least one fused track should be MEDIUM or LOW (conventional)."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        medium_low = [
            ft for ft in result.fused_tracks
            if ft.threat_level in ("MEDIUM", "LOW")
        ]
        assert len(medium_low) >= 1, (
            f"Expected MEDIUM/LOW threat, got: "
            f"{[ft.threat_level for ft in result.fused_tracks]}"
        )

    def test_stealth_flag_present(self):
        """At least one fused track should have is_stealth_candidate set."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        stealth = [ft for ft in result.fused_tracks if ft.is_stealth_candidate]
        assert len(stealth) >= 1, (
            f"Expected at least one stealth candidate flag in fused tracks"
        )

    def test_thermal_extreme_for_hypersonic(self):
        """Thermal should detect the hypersonic target with temperature >1500 K."""
        runner = ScenarioRunner(
            _mixed_targets(), seed=42, n_steps=20, thermal_fov_deg=90.0,
        )
        result = runner.run()

        max_temp = 0.0
        for step_dets in result.thermal_detection_log:
            for d in step_dets:
                if d.temperature_k is not None and d.temperature_k > max_temp:
                    max_temp = d.temperature_k

        assert max_temp > 1500.0, (
            f"Expected thermal detection >1500 K for hypersonic target, "
            f"got max {max_temp:.0f} K"
        )
