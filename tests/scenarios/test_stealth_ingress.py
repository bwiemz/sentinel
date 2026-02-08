"""Scenario 1: Stealth Aircraft Ingress.

A low-RCS target (-20 dBsm at X-band) approaching from ~11 km.
Classical X-band radar should largely miss it.  VHF/UHF radar detects it
easily (+5 dBsm).  The multi-freq correlator flags the >15 dB RCS
variation as stealth.  Thermal sees the moderate engine signature (~590 K).
Quantum radar detects at X-band where classical cannot.
Expected threat: HIGH or CRITICAL.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.types import TargetType

from .conftest import ScenarioRunner, ScenarioTarget


def _stealth_target() -> list[ScenarioTarget]:
    return [
        ScenarioTarget(
            target_id="STEALTH-1",
            position=np.array([10000.0, 5000.0]),
            velocity=np.array([-150.0, -75.0]),
            target_type=TargetType.STEALTH,
            rcs_dbsm=-20.0,
            mach=0.9,
            expected_threat="HIGH",
            expected_stealth=True,
        ),
    ]


class TestStealthIngress:
    """Validate stealth detection through multi-frequency radar, thermal, and quantum."""

    def test_lowfreq_detects_more_than_highfreq(self):
        """VHF/UHF bands should produce significantly more detections than X-band."""
        runner = ScenarioRunner(_stealth_target(), seed=42, n_steps=15)
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

        # Low-freq should dominate for a stealth target
        assert lowfreq_count > xband_count, (
            f"Low-freq ({lowfreq_count}) should exceed X-band ({xband_count})"
        )

    def test_correlator_flags_stealth(self):
        """Multi-freq correlator should detect >15 dB RCS variation."""
        runner = ScenarioRunner(_stealth_target(), seed=42, n_steps=15)
        result = runner.run()

        assert len(result.correlated_detections) >= 1, "Expected at least one correlated detection"
        assert any(
            cd.is_stealth_candidate for cd in result.correlated_detections
        ), "Correlator should flag stealth candidate"

    def test_radar_track_confirmed(self):
        """Radar should confirm at least one track via low-frequency detections."""
        runner = ScenarioRunner(_stealth_target(), seed=42, n_steps=15)
        result = runner.run()

        assert result.radar_confirmed_count >= 1, (
            f"Expected confirmed radar track, got {result.radar_confirmed_count}"
        )

    def test_thermal_detects_stealth(self):
        """Thermal sensor should detect the stealth aircraft (~590 K signature)."""
        runner = ScenarioRunner(_stealth_target(), seed=42, n_steps=15)
        result = runner.run()

        assert result.thermal_confirmed_count >= 1, (
            f"Expected confirmed thermal track, got {result.thermal_confirmed_count}"
        )

    def test_quantum_detects_at_xband(self):
        """Quantum radar should detect the stealth target at X-band."""
        runner = ScenarioRunner(
            _stealth_target(), seed=42, n_steps=30, use_quantum=True,
            quantum_max_range_m=50000.0,
        )
        result = runner.run()

        assert result.quantum_confirmed_count >= 1, (
            f"Expected confirmed quantum track, got {result.quantum_confirmed_count}"
        )

    def test_fused_threat_high_or_critical(self):
        """Fused output should classify the stealth target as HIGH or CRITICAL."""
        runner = ScenarioRunner(_stealth_target(), seed=42, n_steps=15)
        result = runner.run()

        assert len(result.fused_tracks) >= 1, "Expected at least one fused track"
        high_or_crit = [
            ft for ft in result.fused_tracks
            if ft.threat_level in ("HIGH", "CRITICAL")
        ]
        assert len(high_or_crit) >= 1, (
            f"Expected HIGH/CRITICAL threat, got: "
            f"{[ft.threat_level for ft in result.fused_tracks]}"
        )
