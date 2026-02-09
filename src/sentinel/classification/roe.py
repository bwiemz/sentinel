"""Rules of Engagement (ROE) engine.

Combines IFF identification, threat level, and intent to determine
engagement authorization. ROE is purely rule-based and deterministic —
policy decisions, not technical classification.

All ROE features default OFF via ``ROEConfig.enabled = False``.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from sentinel.core.types import EngagementAuth, IFFCode


@dataclass
class ROEConfig:
    """Rules of Engagement configuration."""

    enabled: bool = False
    default_posture: EngagementAuth = EngagementAuth.WEAPONS_HOLD
    friendly_override: bool = True  # FRIENDLY IFF always → HOLD_FIRE
    hostile_attack_posture: EngagementAuth = EngagementAuth.WEAPONS_FREE
    hostile_no_attack_posture: EngagementAuth = EngagementAuth.WEAPONS_TIGHT
    unknown_controlled_posture: EngagementAuth = EngagementAuth.WEAPONS_TIGHT
    spoof_posture: EngagementAuth = EngagementAuth.WEAPONS_FREE
    critical_threat_posture: EngagementAuth = EngagementAuth.WEAPONS_FREE

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig | dict) -> ROEConfig:
        """Parse from OmegaConf config dict."""
        def _parse_auth(val: str, default: EngagementAuth) -> EngagementAuth:
            try:
                return EngagementAuth(val)
            except ValueError:
                return default

        return cls(
            enabled=cfg.get("enabled", False),
            default_posture=_parse_auth(
                cfg.get("default_posture", "weapons_hold"),
                EngagementAuth.WEAPONS_HOLD,
            ),
            friendly_override=cfg.get("friendly_override", True),
            hostile_attack_posture=_parse_auth(
                cfg.get("hostile_attack_posture", "weapons_free"),
                EngagementAuth.WEAPONS_FREE,
            ),
            hostile_no_attack_posture=_parse_auth(
                cfg.get("hostile_no_attack_posture", "weapons_tight"),
                EngagementAuth.WEAPONS_TIGHT,
            ),
            unknown_controlled_posture=_parse_auth(
                cfg.get("unknown_controlled_posture", "weapons_tight"),
                EngagementAuth.WEAPONS_TIGHT,
            ),
            spoof_posture=_parse_auth(
                cfg.get("spoof_posture", "weapons_free"),
                EngagementAuth.WEAPONS_FREE,
            ),
            critical_threat_posture=_parse_auth(
                cfg.get("critical_threat_posture", "weapons_free"),
                EngagementAuth.WEAPONS_FREE,
            ),
        )


class ROEEngine:
    """Computes engagement authorization from IFF + threat + intent.

    Decision priority (highest first):
    1. FRIENDLY IFF with valid crypto → HOLD_FIRE (if friendly_override=True)
    2. SPOOF_SUSPECT → spoof_posture (default WEAPONS_FREE)
    3. HOSTILE/ASSUMED_HOSTILE + ATTACK intent → hostile_attack_posture
    4. HOSTILE/ASSUMED_HOSTILE + other intent → hostile_no_attack_posture
    5. CRITICAL threat + non-friendly → critical_threat_posture
    6. UNKNOWN + controlled airspace → unknown_controlled_posture
    7. Default → default_posture (WEAPONS_HOLD)
    """

    def __init__(self, config: ROEConfig):
        self._config = config

    @property
    def config(self) -> ROEConfig:
        return self._config

    def evaluate(
        self,
        iff_identification: IFFCode | str | None = None,
        threat_level: str = "LOW",
        intent: str = "unknown",
        controlled_airspace: bool = False,
    ) -> EngagementAuth:
        """Determine engagement authorization.

        Args:
            iff_identification: IFF status (IFFCode enum or string value).
            threat_level: Threat level string ("LOW", "MEDIUM", "HIGH", "CRITICAL").
            intent: Intent string ("attack", "approach", "transit", etc.).
            controlled_airspace: Whether target is in controlled airspace.

        Returns:
            EngagementAuth authorization level.
        """
        cfg = self._config

        # Normalize IFF identification to IFFCode
        if iff_identification is None:
            iff = IFFCode.UNKNOWN
        elif isinstance(iff_identification, str):
            try:
                iff = IFFCode(iff_identification)
            except ValueError:
                iff = IFFCode.UNKNOWN
        else:
            iff = iff_identification

        is_friendly = iff in (IFFCode.FRIENDLY, IFFCode.ASSUMED_FRIENDLY)
        is_hostile = iff in (IFFCode.HOSTILE, IFFCode.ASSUMED_HOSTILE)
        is_spoof = iff == IFFCode.SPOOF_SUSPECT

        # Priority 1: Friendly override
        if cfg.friendly_override and is_friendly:
            return EngagementAuth.HOLD_FIRE

        # Priority 2: Spoof suspect
        if is_spoof:
            return cfg.spoof_posture

        # Priority 3-4: Hostile
        if is_hostile:
            if intent == "attack":
                return cfg.hostile_attack_posture
            return cfg.hostile_no_attack_posture

        # Priority 5: Critical threat (non-friendly)
        if threat_level == "CRITICAL" and not is_friendly:
            return cfg.critical_threat_posture

        # Priority 6: Unknown in controlled airspace
        if iff in (IFFCode.UNKNOWN, IFFCode.PENDING) and controlled_airspace:
            return cfg.unknown_controlled_posture

        # Default
        return cfg.default_posture
