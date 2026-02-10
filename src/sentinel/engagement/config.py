"""Engagement system configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sentinel.core.types import ZoneAuth

logger = logging.getLogger(__name__)


@dataclass
class EngagementConfig:
    """Engagement system configuration."""

    enabled: bool = False

    # Assignment weights
    pk_weight: float = 0.4
    tti_weight: float = 0.3
    threat_weight: float = 0.3
    max_tti_s: float = 120.0

    # Default zone settings
    default_zone_auth: ZoneAuth = ZoneAuth.WEAPONS_FREE

    # Raw zone definitions (parsed by ZoneManager.from_config)
    zone_defs: list[dict] = field(default_factory=list)

    # Raw weapon definitions (parsed by WeaponProfile.from_config)
    weapon_defs: list[dict] = field(default_factory=list)

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> EngagementConfig:
        """Build from OmegaConf dict or plain dict."""
        if cfg is None:
            return cls()

        # Handle OmegaConf containers
        try:
            from omegaconf import OmegaConf
            if hasattr(cfg, '_metadata'):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass

        if not isinstance(cfg, dict):
            cfg = dict(cfg)

        # Parse default zone auth
        auth_str = cfg.get("default_zone_auth", "weapons_free")
        try:
            default_auth = ZoneAuth(auth_str)
        except ValueError:
            default_auth = ZoneAuth.WEAPONS_FREE

        # Zone and weapon definitions stay as raw dicts for factory methods
        zones = cfg.get("zones", [])
        if zones is None:
            zones = []
        zone_defs = [dict(z) if not isinstance(z, dict) else z for z in zones]

        weapons = cfg.get("weapons", [])
        if weapons is None:
            weapons = []
        weapon_defs = [dict(w) if not isinstance(w, dict) else w for w in weapons]

        return cls(
            enabled=bool(cfg.get("enabled", False)),
            pk_weight=float(cfg.get("pk_weight", 0.4)),
            tti_weight=float(cfg.get("tti_weight", 0.3)),
            threat_weight=float(cfg.get("threat_weight", 0.3)),
            max_tti_s=float(cfg.get("max_tti_s", 120.0)),
            default_zone_auth=default_auth,
            zone_defs=zone_defs,
            weapon_defs=weapon_defs,
        )
