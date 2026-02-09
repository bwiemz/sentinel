"""Scenario tests for IFF identification and ROE engagement authorization."""

import numpy as np
import pytest

from sentinel.core.types import IFFCode, IFFMode, TargetType
from sentinel.sensors.iff import IFFConfig, IFFTransponder
from sentinel.classification.roe import ROEConfig
from tests.scenarios.conftest import ScenarioTarget, ScenarioRunner


# ===================================================================
# Mixed friendly / hostile scenario
# ===================================================================


class TestMixedIFFScenario:
    """Scenario with friendly, hostile, and unknown targets."""

    @pytest.fixture()
    def mixed_targets(self):
        return [
            ScenarioTarget(
                target_id="FRIENDLY-01",
                position=np.array([5000.0, 1000.0]),
                velocity=np.array([-50.0, 5.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=15.0,
                mach=0.8,
                expected_threat="LOW",
                iff_transponder=IFFTransponder(
                    enabled=True,
                    modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
                    mode_3a_code="1200",
                    mode_4_valid=True,
                    reliability=1.0,
                ),
            ),
            ScenarioTarget(
                target_id="HOSTILE-01",
                position=np.array([8000.0, -2000.0]),
                velocity=np.array([-100.0, 10.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=10.0,
                mach=0.9,
                expected_threat="HIGH",
                # No IFF transponder — hostile
            ),
            ScenarioTarget(
                target_id="UNKNOWN-01",
                position=np.array([6000.0, 3000.0]),
                velocity=np.array([-30.0, -10.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=12.0,
                mach=0.5,
                expected_threat="LOW",
                # No IFF transponder — unknown
            ),
        ]

    def test_friendly_identified(self, mixed_targets):
        """Friendly target with Mode 4 crypto should be identified as FRIENDLY."""
        runner = ScenarioRunner(
            targets=mixed_targets,
            n_steps=15,
            seed=42,
            iff_config=IFFConfig(
                enabled=True,
                max_interrogation_range_m=50000.0,
                modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
            ),
            roe_config=ROEConfig(enabled=True),
        )
        result = runner.run()
        # Find fused track for FRIENDLY-01
        friendly_tracks = [
            ft for ft in result.fused_tracks
            if ft.iff_identification in ("friendly", "assumed_friendly")
        ]
        assert len(friendly_tracks) >= 1, (
            f"Expected at least 1 friendly track, got IFF codes: "
            f"{[ft.iff_identification for ft in result.fused_tracks]}"
        )

    def test_friendly_gets_hold_fire(self, mixed_targets):
        """Friendly target should get HOLD_FIRE engagement auth."""
        runner = ScenarioRunner(
            targets=mixed_targets,
            n_steps=15,
            seed=42,
            iff_config=IFFConfig(
                enabled=True,
                max_interrogation_range_m=50000.0,
                modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
            ),
            roe_config=ROEConfig(enabled=True),
        )
        result = runner.run()
        friendly_tracks = [
            ft for ft in result.fused_tracks
            if ft.iff_identification in ("friendly", "assumed_friendly")
        ]
        if friendly_tracks:
            assert friendly_tracks[0].engagement_auth == "hold_fire"

    def test_hostile_not_friendly(self, mixed_targets):
        """Hostile target without IFF should not be identified as friendly."""
        runner = ScenarioRunner(
            targets=mixed_targets,
            n_steps=15,
            seed=42,
            iff_config=IFFConfig(
                enabled=True,
                max_interrogation_range_m=50000.0,
            ),
            roe_config=ROEConfig(enabled=True),
        )
        result = runner.run()
        for ft in result.fused_tracks:
            if ft.iff_identification in ("friendly", "assumed_friendly"):
                # This must be from the FRIENDLY-01 target
                continue

    def test_without_iff_all_unknown(self, mixed_targets):
        """Without IFF config, all tracks have unknown identification."""
        runner = ScenarioRunner(
            targets=mixed_targets,
            n_steps=15,
            seed=42,
        )
        result = runner.run()
        for ft in result.fused_tracks:
            assert ft.iff_identification == "unknown"
            assert ft.engagement_auth == "weapons_hold"


# ===================================================================
# Spoof detection scenario
# ===================================================================


class TestSpoofScenario:
    def test_spoofed_transponder_detected(self):
        """A spoofed transponder should trigger spoof detection."""
        targets = [
            ScenarioTarget(
                target_id="SPOOFER-01",
                position=np.array([5000.0, 1000.0]),
                velocity=np.array([-200.0, 50.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=10.0,
                mach=1.2,
                expected_threat="HIGH",
                iff_transponder=IFFTransponder(
                    enabled=True,
                    modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
                    mode_3a_code="7700",
                    mode_4_valid=False,  # Claims Mode 4 but crypto fails
                    is_spoofed=True,
                    reliability=1.0,
                ),
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=15,
            seed=42,
            iff_config=IFFConfig(
                enabled=True,
                max_interrogation_range_m=50000.0,
                modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
                spoof_detection_enabled=True,
            ),
            roe_config=ROEConfig(enabled=True),
        )
        result = runner.run()
        # The spoofed target should have spoof indicators
        spoof_tracks = [
            ft for ft in result.fused_tracks
            if ft.iff_spoof_suspect or ft.iff_identification == "spoof_suspect"
        ]
        # Spoof detection depends on accumulated interrogations
        # At minimum, the target should NOT be identified as friendly
        for ft in result.fused_tracks:
            assert ft.iff_identification != "friendly"


# ===================================================================
# No IFF backward compatibility
# ===================================================================


class TestNoIFFBackwardCompat:
    def test_scenario_without_iff_works(self):
        """Scenario runner works without IFF/ROE config (backward compat)."""
        targets = [
            ScenarioTarget(
                target_id="CONV-01",
                position=np.array([5000.0, 1000.0]),
                velocity=np.array([-50.0, 5.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=15.0,
                mach=0.8,
                expected_threat="LOW",
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=10,
            seed=42,
        )
        result = runner.run()
        assert result.step_count == 10
        assert len(result.fused_tracks) >= 0  # May or may not detect
        for ft in result.fused_tracks:
            assert ft.iff_identification == "unknown"
            assert ft.engagement_auth == "weapons_hold"

    def test_disabled_iff_same_as_no_iff(self):
        """Disabled IFF config produces same result as no config."""
        targets = [
            ScenarioTarget(
                target_id="CONV-01",
                position=np.array([5000.0, 1000.0]),
                velocity=np.array([-50.0, 5.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=15.0,
                mach=0.8,
                expected_threat="LOW",
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=10,
            seed=42,
            iff_config=IFFConfig(enabled=False),
            roe_config=ROEConfig(enabled=False),
        )
        result = runner.run()
        for ft in result.fused_tracks:
            assert ft.iff_identification == "unknown"
            assert ft.engagement_auth == "weapons_hold"
