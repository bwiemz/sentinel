"""Integration tests for the engagement system with pipeline config and schema.

Verifies that the engagement system integrates correctly with OmegaConf
configuration loading, schema validation, and end-to-end plan generation.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.config import EngagementConfig
from sentinel.engagement.manager import EngagementManager


class MockTrack:
    """Lightweight mock of EnhancedFusedTrack for engagement testing."""

    def __init__(
        self,
        fused_id,
        position_m,
        velocity_mps,
        azimuth_deg,
        threat_level="HIGH",
        engagement_auth="weapons_free",
        iff_identification="hostile",
        is_jammed=False,
    ):
        self.fused_id = fused_id
        self.position_m = np.asarray(position_m, dtype=float)
        self.velocity_mps = velocity_mps
        self.azimuth_deg = azimuth_deg
        self.threat_level = threat_level
        self.engagement_auth = engagement_auth
        self.iff_identification = iff_identification
        self.is_jammed = is_jammed
        self.radar_track = None


@pytest.fixture
def default_config(config_path):
    """Load the default YAML configuration."""
    return OmegaConf.load(config_path)


def _engagement_yaml_config():
    """Build an OmegaConf config with engagement enabled, zones, and weapons."""
    return OmegaConf.create({
        "sentinel": {
            "engagement": {
                "enabled": True,
                "pk_weight": 0.4,
                "tti_weight": 0.3,
                "threat_weight": 0.3,
                "max_tti_s": 120.0,
                "default_zone_auth": "weapons_free",
                "zones": [
                    {
                        "type": "circle",
                        "zone_id": "ZONE-1",
                        "name": "Engagement Area Alpha",
                        "center_xy": [0.0, 0.0],
                        "radius_m": 50000.0,
                        "authorization": "weapons_free",
                        "priority": 0,
                    },
                    {
                        "type": "circle",
                        "zone_id": "NFZ-1",
                        "name": "Embassy No-Fire Zone",
                        "center_xy": [10000.0, 5000.0],
                        "radius_m": 2000.0,
                        "authorization": "no_fire",
                        "priority": 10,
                    },
                ],
                "weapons": [
                    {
                        "weapon_id": "SAM-1",
                        "name": "Patriot Battery Alpha",
                        "weapon_type": "sam_long",
                        "position_xy": [0.0, 0.0],
                        "min_range_m": 3000.0,
                        "max_range_m": 70000.0,
                        "optimal_range_m": 40000.0,
                        "weapon_speed_mps": 1700.0,
                        "pk_base": 0.85,
                        "max_target_speed_mps": 2000.0,
                        "max_simultaneous_engagements": 4,
                        "rounds_remaining": 16,
                    },
                ],
            },
        },
    })


class TestEngagementE2E:
    """Integration tests for the engagement system."""

    def test_config_schema_validates_engagement(self, config_path):
        """Loading default.yaml + engagement config should pass schema validation."""
        from sentinel.core.config_schema import validate_config

        cfg = OmegaConf.load(config_path)

        # Merge engagement config on top of defaults
        engagement_overlay = OmegaConf.create({
            "sentinel": {
                "engagement": {
                    "enabled": True,
                    "pk_weight": 0.4,
                    "tti_weight": 0.3,
                    "threat_weight": 0.3,
                    "max_tti_s": 120.0,
                    "default_zone_auth": "weapons_free",
                    "zones": [
                        {
                            "type": "circle",
                            "zone_id": "ZONE-TEST",
                            "name": "Test Zone",
                            "center_xy": [0.0, 0.0],
                            "radius_m": 10000.0,
                            "authorization": "weapons_free",
                            "priority": 0,
                        },
                    ],
                    "weapons": [
                        {
                            "weapon_id": "WPN-TEST",
                            "name": "Test Weapon",
                            "weapon_type": "sam_medium",
                            "position_xy": [0.0, 0.0],
                            "min_range_m": 500.0,
                            "max_range_m": 20000.0,
                        },
                    ],
                },
            },
        })
        merged = OmegaConf.merge(cfg, engagement_overlay)
        cfg_dict = OmegaConf.to_container(merged, resolve=True)

        # Should not raise
        schema = validate_config(cfg_dict)
        assert schema.sentinel.engagement.enabled is True
        assert len(schema.sentinel.engagement.zones) == 1
        assert len(schema.sentinel.engagement.weapons) == 1

    def test_engagement_disabled_by_default(self, config_path):
        """Default configuration should have engagement disabled."""
        cfg = OmegaConf.load(config_path)
        engagement_enabled = OmegaConf.select(cfg, "sentinel.engagement.enabled")
        assert engagement_enabled is False, (
            "Engagement should be disabled by default in default.yaml"
        )

    def test_engagement_manager_from_yaml_config(self):
        """EngagementManager can be built from an OmegaConf configuration.

        Verifies that zones and weapons are correctly parsed from the
        YAML-style config structure.
        """
        cfg = _engagement_yaml_config()
        engagement_cfg = OmegaConf.select(cfg, "sentinel.engagement")

        mgr = EngagementManager.from_config(engagement_cfg)
        assert mgr is not None, "Manager should be created when enabled=True"

        # Verify zones parsed
        zones = mgr.zone_manager.get_all_zones()
        assert len(zones) == 2, f"Expected 2 zones, got {len(zones)}"
        zone_ids = {z.zone_id for z in zones}
        assert "ZONE-1" in zone_ids
        assert "NFZ-1" in zone_ids

        # Verify weapons parsed
        weapons = mgr.weapons
        assert len(weapons) == 1
        assert weapons[0].weapon_id == "SAM-1"
        assert weapons[0].max_range_m == 70000.0

    def test_engagement_plan_serialization(self):
        """Engagement plan to_dict() should have expected structure."""
        cfg = _engagement_yaml_config()
        engagement_cfg = OmegaConf.select(cfg, "sentinel.engagement")
        mgr = EngagementManager.from_config(engagement_cfg)
        assert mgr is not None

        track = MockTrack(
            fused_id="TGT-SERIAL",
            position_m=[30000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([track], current_time=42.0)
        plan_dict = plan.to_dict()

        # Verify top-level structure
        assert "timestamp" in plan_dict
        assert plan_dict["timestamp"] == 42.0
        assert "assignment" in plan_dict
        assert "zone_statuses" in plan_dict

        # Verify assignment sub-structure
        assignment_dict = plan_dict["assignment"]
        assert "assignments" in assignment_dict
        assert "unassigned_weapons" in assignment_dict
        assert "unassigned_tracks" in assignment_dict
        assert "total_pk" in assignment_dict
        assert "assignment_count" in assignment_dict

        # Verify zone_statuses is a list of dicts
        assert isinstance(plan_dict["zone_statuses"], list)
        for zs in plan_dict["zone_statuses"]:
            assert "track_id" in zs
            assert "zone_authorization" in zs
            assert "containing_zones" in zs

    def test_engagement_with_roe_integration(self):
        """ROE engagement_auth should gate engagement decisions.

        WEAPONS_FREE zone + hostile + WEAPONS_FREE auth => assigned.
        Same setup with HOLD_FIRE auth => not assigned.
        """
        cfg = _engagement_yaml_config()
        engagement_cfg = OmegaConf.select(cfg, "sentinel.engagement")
        mgr = EngagementManager.from_config(engagement_cfg)
        assert mgr is not None

        # WEAPONS_FREE ROE -- should be assigned
        track_free = MockTrack(
            fused_id="TGT-ROE",
            position_m=[30000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan_free = mgr.evaluate([track_free], current_time=100.0)
        assigned_free = {tid for _, tid in plan_free.assignment.assignments}
        assert "TGT-ROE" in assigned_free, (
            "Hostile target with WEAPONS_FREE auth should be assigned"
        )

        # HOLD_FIRE ROE -- should NOT be assigned
        track_hold = MockTrack(
            fused_id="TGT-ROE-HOLD",
            position_m=[30000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="hold_fire",
            iff_identification="hostile",
        )

        plan_hold = mgr.evaluate([track_hold], current_time=100.0)
        assigned_hold = {tid for _, tid in plan_hold.assignment.assignments}
        assert "TGT-ROE-HOLD" not in assigned_hold, (
            "Hostile target with HOLD_FIRE auth should NOT be assigned"
        )
