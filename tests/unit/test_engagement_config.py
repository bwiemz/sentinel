"""Unit tests for EngagementConfig parsing and defaults."""

import pytest

from sentinel.core.types import ZoneAuth
from sentinel.engagement.config import EngagementConfig


class TestEngagementConfig:
    # ---------------------------------------------------------------
    # Default construction
    # ---------------------------------------------------------------

    def test_default_disabled(self):
        """Default EngagementConfig has enabled=False."""
        cfg = EngagementConfig()
        assert cfg.enabled is False
        assert cfg.default_zone_auth == ZoneAuth.WEAPONS_FREE
        assert cfg.zone_defs == []
        assert cfg.weapon_defs == []

    # ---------------------------------------------------------------
    # from_omegaconf — full config
    # ---------------------------------------------------------------

    def test_from_omegaconf_full(self):
        """Complete config dict is parsed with all fields."""
        raw = {
            "enabled": True,
            "pk_weight": 0.5,
            "tti_weight": 0.2,
            "threat_weight": 0.3,
            "max_tti_s": 90.0,
            "default_zone_auth": "restricted_fire",
            "zones": [
                {
                    "zone_id": "Z-1",
                    "type": "circle",
                    "center_xy": [100.0, 200.0],
                    "radius_m": 5000.0,
                    "authorization": "no_fire",
                },
            ],
            "weapons": [
                {
                    "weapon_id": "SAM-1",
                    "weapon_type": "sam_medium",
                    "position_xy": [0.0, 0.0],
                },
            ],
        }
        cfg = EngagementConfig.from_omegaconf(raw)
        assert cfg.enabled is True
        assert cfg.pk_weight == 0.5
        assert cfg.tti_weight == 0.2
        assert cfg.threat_weight == 0.3
        assert cfg.max_tti_s == 90.0
        assert cfg.default_zone_auth == ZoneAuth.RESTRICTED_FIRE
        assert len(cfg.zone_defs) == 1
        assert len(cfg.weapon_defs) == 1

    # ---------------------------------------------------------------
    # from_omegaconf — empty / None
    # ---------------------------------------------------------------

    def test_from_omegaconf_empty(self):
        """Empty dict produces sensible defaults."""
        cfg = EngagementConfig.from_omegaconf({})
        assert cfg.enabled is False
        assert cfg.pk_weight == 0.4
        assert cfg.max_tti_s == 120.0

    def test_from_omegaconf_none(self):
        """None input produces defaults (enabled=False)."""
        cfg = EngagementConfig.from_omegaconf(None)
        assert cfg.enabled is False
        assert cfg.zone_defs == []
        assert cfg.weapon_defs == []

    # ---------------------------------------------------------------
    # from_omegaconf — plain dict
    # ---------------------------------------------------------------

    def test_from_omegaconf_dict(self):
        """Plain dict (not OmegaConf) parses correctly."""
        cfg = EngagementConfig.from_omegaconf({"enabled": True, "pk_weight": 0.6})
        assert cfg.enabled is True
        assert cfg.pk_weight == 0.6

    # ---------------------------------------------------------------
    # Zone / weapon defs preserved as raw dicts
    # ---------------------------------------------------------------

    def test_zone_defs_preserved(self):
        """Zone definitions are stored as raw dicts."""
        zone = {"zone_id": "Z-A", "type": "circle", "radius_m": 1000.0}
        cfg = EngagementConfig.from_omegaconf({"zones": [zone]})
        assert len(cfg.zone_defs) == 1
        assert cfg.zone_defs[0]["zone_id"] == "Z-A"
        assert cfg.zone_defs[0]["radius_m"] == 1000.0

    def test_weapon_defs_preserved(self):
        """Weapon definitions are stored as raw dicts."""
        weapon = {"weapon_id": "GUN-1", "weapon_type": "gun"}
        cfg = EngagementConfig.from_omegaconf({"weapons": [weapon]})
        assert len(cfg.weapon_defs) == 1
        assert cfg.weapon_defs[0]["weapon_id"] == "GUN-1"
        assert cfg.weapon_defs[0]["weapon_type"] == "gun"

    # ---------------------------------------------------------------
    # Zone auth parsing
    # ---------------------------------------------------------------

    def test_default_zone_auth_parsing(self):
        """String 'no_fire' is parsed to ZoneAuth.NO_FIRE."""
        cfg = EngagementConfig.from_omegaconf({"default_zone_auth": "no_fire"})
        assert cfg.default_zone_auth == ZoneAuth.NO_FIRE

    def test_invalid_zone_auth_defaults(self):
        """Unknown auth string falls back to WEAPONS_FREE."""
        cfg = EngagementConfig.from_omegaconf(
            {"default_zone_auth": "totally_invalid"}
        )
        assert cfg.default_zone_auth == ZoneAuth.WEAPONS_FREE

    # ---------------------------------------------------------------
    # None zones / weapons
    # ---------------------------------------------------------------

    def test_none_zones_weapons(self):
        """Explicit None for zones/weapons produces empty lists."""
        cfg = EngagementConfig.from_omegaconf(
            {"zones": None, "weapons": None}
        )
        assert cfg.zone_defs == []
        assert cfg.weapon_defs == []

    # ---------------------------------------------------------------
    # Weight parsing
    # ---------------------------------------------------------------

    def test_weights(self):
        """pk_weight, tti_weight, threat_weight are parsed as floats."""
        cfg = EngagementConfig.from_omegaconf({
            "pk_weight": 0.1,
            "tti_weight": 0.2,
            "threat_weight": 0.7,
        })
        assert cfg.pk_weight == pytest.approx(0.1)
        assert cfg.tti_weight == pytest.approx(0.2)
        assert cfg.threat_weight == pytest.approx(0.7)
