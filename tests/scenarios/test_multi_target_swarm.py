"""Scenario 3: Multi-Target Swarm.

Four conventional targets at various ranges and azimuths approaching
simultaneously.  All have flat RCS across bands (~0-1 dB variation),
moderate thermal signatures (745-790 K), and strong radar returns.
Tests correct data association (no track swaps) and proper threat
classification (MEDIUM or LOW for conventional targets).
"""

from __future__ import annotations

import numpy as np

from sentinel.core.types import TargetType

from .conftest import ScenarioRunner, ScenarioTarget


def _swarm_targets() -> list[ScenarioTarget]:
    return [
        ScenarioTarget(
            target_id="SWARM-1",
            position=np.array([5000.0, 2000.0]),
            velocity=np.array([-30.0, -10.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=10.0,
            mach=0.8,
            expected_threat="MEDIUM",
        ),
        ScenarioTarget(
            target_id="SWARM-2",
            position=np.array([7000.0, -1000.0]),
            velocity=np.array([-25.0, 5.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=12.0,
            mach=0.7,
            expected_threat="MEDIUM",
        ),
        ScenarioTarget(
            target_id="SWARM-3",
            position=np.array([4000.0, -3000.0]),
            velocity=np.array([-20.0, 10.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=8.0,
            mach=0.6,
            expected_threat="LOW",
        ),
        ScenarioTarget(
            target_id="SWARM-4",
            position=np.array([8000.0, 4000.0]),
            velocity=np.array([-40.0, -20.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=15.0,
            mach=0.9,
            expected_threat="MEDIUM",
        ),
    ]


class TestMultiTargetSwarm:
    """Validate multi-target tracking and correct threat classification."""

    def test_all_targets_tracked(self):
        """Radar should confirm tracks for all 4 conventional targets."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15,
            multifreq_base_pd=0.95, thermal_fov_deg=90.0,
        )
        result = runner.run()

        assert result.radar_confirmed_count >= 4, (
            f"Expected >= 4 confirmed radar tracks, got {result.radar_confirmed_count}"
        )

    def test_thermal_tracks_multiple(self):
        """Thermal should detect most of the swarm targets."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15, thermal_fov_deg=90.0,
        )
        result = runner.run()

        assert result.thermal_confirmed_count >= 3, (
            f"Expected >= 3 confirmed thermal tracks, got {result.thermal_confirmed_count}"
        )

    def test_no_stealth_flags(self):
        """No conventional target should be flagged as stealth."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15, thermal_fov_deg=90.0,
            multifreq_base_pd=0.99,  # High pd to ensure multi-band detection
        )
        result = runner.run()

        stealth_flags = [ft for ft in result.fused_tracks if ft.is_stealth_candidate]
        assert len(stealth_flags) == 0, (
            f"Expected no stealth flags, got {len(stealth_flags)}"
        )

    def test_no_hypersonic_flags(self):
        """No conventional target should be flagged as hypersonic."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15, thermal_fov_deg=90.0,
        )
        result = runner.run()

        hyper_flags = [ft for ft in result.fused_tracks if ft.is_hypersonic_candidate]
        assert len(hyper_flags) == 0, (
            f"Expected no hypersonic flags, got {len(hyper_flags)}"
        )

    def test_all_medium_or_low(self):
        """All fused tracks should be classified as MEDIUM or LOW."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15, thermal_fov_deg=90.0,
            multifreq_base_pd=0.99,  # High pd to ensure multi-band detection
        )
        result = runner.run()

        assert len(result.fused_tracks) >= 1, "Expected at least one fused track"
        for ft in result.fused_tracks:
            assert ft.threat_level in ("MEDIUM", "LOW"), (
                f"Expected MEDIUM/LOW, got {ft.threat_level} for {ft.fused_id}"
            )

    def test_fused_count_matches_targets(self):
        """Fusion should produce tracks for most of the swarm."""
        runner = ScenarioRunner(
            _swarm_targets(), seed=42, n_steps=15, thermal_fov_deg=90.0,
        )
        result = runner.run()

        assert len(result.fused_tracks) >= 3, (
            f"Expected >= 3 fused tracks, got {len(result.fused_tracks)}"
        )
