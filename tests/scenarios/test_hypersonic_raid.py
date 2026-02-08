"""Scenario 2: Hypersonic Missile Raid.

A Mach 5.3 target approaching from ~16 km.  Plasma sheath attenuates
X-band radar (~17 dB) but VHF penetrates (~2 dB).  Thermal sees extreme
leading-edge temperature (~1616 K > 1500 K CRITICAL threshold).
Expected threat: CRITICAL.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.types import TargetType

from .conftest import ScenarioRunner, ScenarioTarget


def _hypersonic_target() -> list[ScenarioTarget]:
    return [
        ScenarioTarget(
            target_id="HYPER-1",
            position=np.array([15000.0, 5000.0]),
            velocity=np.array([-1715.0, -571.0]),
            target_type=TargetType.HYPERSONIC,
            rcs_dbsm=5.0,
            mach=5.3,
            expected_threat="CRITICAL",
            expected_hypersonic=True,
        ),
    ]


class TestHypersonicRaid:
    """Validate hypersonic detection through plasma-degraded radar and thermal."""

    def test_plasma_degrades_xband(self):
        """X-band detection count should be lower than VHF/UHF due to plasma."""
        runner = ScenarioRunner(_hypersonic_target(), seed=42, n_steps=15)
        result = runner.run()

        lowfreq_count = 0
        xband_count = 0
        for step_dets in result.multifreq_detection_log:
            for d in step_dets:
                band = d.radar_band
                if band in ("vhf", "uhf"):
                    lowfreq_count += 1
                elif band == "x_band":
                    xband_count += 1

        assert lowfreq_count > xband_count, (
            f"Low-freq ({lowfreq_count}) should exceed X-band ({xband_count}) "
            f"due to plasma sheath attenuation"
        )

    def test_lowfreq_detects_through_plasma(self):
        """VHF/UHF should reliably detect the target despite plasma sheath."""
        runner = ScenarioRunner(_hypersonic_target(), seed=42, n_steps=15)
        result = runner.run()

        steps_with_lowfreq = 0
        for step_dets in result.multifreq_detection_log:
            if any(d.radar_band in ("vhf", "uhf") for d in step_dets):
                steps_with_lowfreq += 1

        assert steps_with_lowfreq >= result.step_count // 2, (
            f"VHF/UHF detected in {steps_with_lowfreq}/{result.step_count} steps; "
            f"expected at least half"
        )

    def test_thermal_extreme_temperature(self):
        """Thermal detections should show temperatures exceeding 1500 K."""
        runner = ScenarioRunner(_hypersonic_target(), seed=42, n_steps=15)
        result = runner.run()

        max_temp = 0.0
        for step_dets in result.thermal_detection_log:
            for d in step_dets:
                if d.temperature_k is not None and d.temperature_k > max_temp:
                    max_temp = d.temperature_k

        assert max_temp > 1500.0, (
            f"Expected thermal detection >1500 K for Mach 5.3, got max {max_temp:.0f} K"
        )

    def test_radar_track_confirmed(self):
        """Low-frequency radar should confirm the hypersonic target."""
        runner = ScenarioRunner(_hypersonic_target(), seed=42, n_steps=15)
        result = runner.run()

        assert result.radar_confirmed_count >= 1, (
            f"Expected confirmed radar track, got {result.radar_confirmed_count}"
        )

    def test_fused_threat_critical(self):
        """Fused output should classify the hypersonic target as CRITICAL."""
        runner = ScenarioRunner(_hypersonic_target(), seed=42, n_steps=15)
        result = runner.run()

        assert len(result.fused_tracks) >= 1, "Expected at least one fused track"
        critical = [ft for ft in result.fused_tracks if ft.threat_level == "CRITICAL"]
        assert len(critical) >= 1, (
            f"Expected CRITICAL threat, got: "
            f"{[ft.threat_level for ft in result.fused_tracks]}"
        )
