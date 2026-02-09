"""Unit tests for Rules of Engagement engine."""

import pytest

from sentinel.core.types import EngagementAuth, IFFCode
from sentinel.classification.roe import ROEConfig, ROEEngine


# ===================================================================
# ROEConfig
# ===================================================================


class TestROEConfig:
    def test_default_disabled(self):
        cfg = ROEConfig()
        assert cfg.enabled is False
        assert cfg.default_posture == EngagementAuth.WEAPONS_HOLD

    def test_from_omegaconf(self):
        cfg_dict = {
            "enabled": True,
            "default_posture": "weapons_tight",
            "friendly_override": True,
            "hostile_attack_posture": "weapons_free",
            "spoof_posture": "weapons_free",
        }
        cfg = ROEConfig.from_omegaconf(cfg_dict)
        assert cfg.enabled is True
        assert cfg.default_posture == EngagementAuth.WEAPONS_TIGHT
        assert cfg.hostile_attack_posture == EngagementAuth.WEAPONS_FREE

    def test_from_omegaconf_defaults(self):
        cfg = ROEConfig.from_omegaconf({})
        assert cfg.enabled is False
        assert cfg.default_posture == EngagementAuth.WEAPONS_HOLD

    def test_from_omegaconf_invalid_posture(self):
        cfg = ROEConfig.from_omegaconf({"default_posture": "invalid_value"})
        assert cfg.default_posture == EngagementAuth.WEAPONS_HOLD


# ===================================================================
# ROEEngine — friendly override
# ===================================================================


class TestROEFriendly:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_friendly_iff_hold_fire(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.FRIENDLY, "HIGH", "attack")
        assert auth == EngagementAuth.HOLD_FIRE

    def test_assumed_friendly_hold_fire(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.ASSUMED_FRIENDLY, "CRITICAL", "attack")
        assert auth == EngagementAuth.HOLD_FIRE

    def test_friendly_override_disabled(self):
        engine = self._engine(friendly_override=False)
        auth = engine.evaluate(IFFCode.FRIENDLY, "LOW", "transit")
        # Without override, friendly falls through to default
        assert auth == EngagementAuth.WEAPONS_HOLD

    def test_friendly_string_input(self):
        engine = self._engine()
        auth = engine.evaluate("friendly", "HIGH", "attack")
        assert auth == EngagementAuth.HOLD_FIRE


# ===================================================================
# ROEEngine — hostile
# ===================================================================


class TestROEHostile:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_hostile_attack_weapons_free(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.HOSTILE, "HIGH", "attack")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_hostile_no_attack_weapons_tight(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.HOSTILE, "HIGH", "transit")
        assert auth == EngagementAuth.WEAPONS_TIGHT

    def test_assumed_hostile_attack(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.ASSUMED_HOSTILE, "MEDIUM", "attack")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_assumed_hostile_patrol(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.ASSUMED_HOSTILE, "MEDIUM", "patrol")
        assert auth == EngagementAuth.WEAPONS_TIGHT


# ===================================================================
# ROEEngine — spoof suspect
# ===================================================================


class TestROESpoof:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_spoof_weapons_free(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.SPOOF_SUSPECT, "HIGH", "approach")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_spoof_custom_posture(self):
        engine = self._engine(spoof_posture=EngagementAuth.WEAPONS_TIGHT)
        auth = engine.evaluate(IFFCode.SPOOF_SUSPECT, "LOW", "transit")
        assert auth == EngagementAuth.WEAPONS_TIGHT


# ===================================================================
# ROEEngine — critical threat
# ===================================================================


class TestROECritical:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_critical_unknown_weapons_free(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.UNKNOWN, "CRITICAL", "approach")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_critical_pending_weapons_free(self):
        engine = self._engine()
        auth = engine.evaluate(IFFCode.PENDING, "CRITICAL", "attack")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_critical_friendly_still_hold_fire(self):
        # Friendly override takes priority over critical threat
        engine = self._engine()
        auth = engine.evaluate(IFFCode.FRIENDLY, "CRITICAL", "attack")
        assert auth == EngagementAuth.HOLD_FIRE


# ===================================================================
# ROEEngine — controlled airspace
# ===================================================================


class TestROEControlledAirspace:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_unknown_in_controlled_airspace(self):
        engine = self._engine()
        auth = engine.evaluate(
            IFFCode.UNKNOWN, "LOW", "transit", controlled_airspace=True
        )
        assert auth == EngagementAuth.WEAPONS_TIGHT

    def test_pending_in_controlled_airspace(self):
        engine = self._engine()
        auth = engine.evaluate(
            IFFCode.PENDING, "LOW", "transit", controlled_airspace=True
        )
        assert auth == EngagementAuth.WEAPONS_TIGHT

    def test_friendly_in_controlled_airspace_still_hold_fire(self):
        engine = self._engine()
        auth = engine.evaluate(
            IFFCode.FRIENDLY, "LOW", "transit", controlled_airspace=True
        )
        assert auth == EngagementAuth.HOLD_FIRE

    def test_unknown_outside_controlled_default(self):
        engine = self._engine()
        auth = engine.evaluate(
            IFFCode.UNKNOWN, "LOW", "transit", controlled_airspace=False
        )
        assert auth == EngagementAuth.WEAPONS_HOLD


# ===================================================================
# ROEEngine — default posture
# ===================================================================


class TestROEDefault:
    def test_unknown_no_special_conditions(self):
        engine = ROEEngine(ROEConfig(enabled=True))
        auth = engine.evaluate(IFFCode.UNKNOWN, "LOW", "transit")
        assert auth == EngagementAuth.WEAPONS_HOLD

    def test_none_iff(self):
        engine = ROEEngine(ROEConfig(enabled=True))
        auth = engine.evaluate(None, "LOW", "transit")
        assert auth == EngagementAuth.WEAPONS_HOLD

    def test_invalid_string_iff(self):
        engine = ROEEngine(ROEConfig(enabled=True))
        auth = engine.evaluate("garbage_value", "LOW", "transit")
        assert auth == EngagementAuth.WEAPONS_HOLD

    def test_custom_default_posture(self):
        engine = ROEEngine(ROEConfig(
            enabled=True,
            default_posture=EngagementAuth.WEAPONS_TIGHT,
        ))
        auth = engine.evaluate(IFFCode.UNKNOWN, "MEDIUM", "patrol")
        assert auth == EngagementAuth.WEAPONS_TIGHT


# ===================================================================
# ROEEngine — priority ordering
# ===================================================================


class TestROEPriority:
    def _engine(self, **kwargs):
        return ROEEngine(ROEConfig(enabled=True, **kwargs))

    def test_friendly_beats_hostile_attack(self):
        """Friendly override takes highest priority."""
        engine = self._engine()
        auth = engine.evaluate(IFFCode.FRIENDLY, "CRITICAL", "attack")
        assert auth == EngagementAuth.HOLD_FIRE

    def test_spoof_beats_hostile(self):
        """Spoof detection beats normal hostile rules."""
        engine = self._engine()
        auth = engine.evaluate(IFFCode.SPOOF_SUSPECT, "LOW", "transit")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_hostile_beats_critical(self):
        """Hostile + attack triggers before generic critical rule."""
        engine = self._engine()
        auth = engine.evaluate(IFFCode.HOSTILE, "CRITICAL", "attack")
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_critical_beats_controlled_airspace(self):
        """Critical threat triggers before controlled airspace rule."""
        engine = self._engine()
        auth = engine.evaluate(
            IFFCode.UNKNOWN, "CRITICAL", "transit", controlled_airspace=True
        )
        assert auth == EngagementAuth.WEAPONS_FREE

    def test_config_property(self):
        cfg = ROEConfig(enabled=True)
        engine = ROEEngine(cfg)
        assert engine.config is cfg
