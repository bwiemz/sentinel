"""Engagement manager — top-level orchestrator for engagement zone management.

Ties together: zones + weapons + feasibility + assignment.
Called optionally after fusion step in the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sentinel.core.types import EngagementAuth, ZoneAuth
from sentinel.engagement.assignment import EngagementAssignment, WeaponTargetAssigner
from sentinel.engagement.config import EngagementConfig
from sentinel.engagement.feasibility import FeasibilityCalculator
from sentinel.engagement.weapons import WEZCalculator, WeaponProfile
from sentinel.engagement.zones import EngagementZone, ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class EngagementPlan:
    """Complete engagement plan for current tactical picture."""

    timestamp: float
    assignment: EngagementAssignment
    zone_statuses: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "assignment": self.assignment.to_dict(),
            "zone_statuses": self.zone_statuses,
        }


class EngagementManager:
    """Top-level orchestrator for engagement zone management.

    Ties together: zones + weapons + feasibility + assignment.
    Called optionally after fusion step in the pipeline.
    """

    def __init__(
        self,
        config: EngagementConfig,
        zone_manager: ZoneManager | None = None,
        geo_context: Any = None,
    ):
        self._config = config

        # Build zone manager from config if not provided
        if zone_manager is not None:
            self._zone_manager = zone_manager
        else:
            self._zone_manager = ZoneManager.from_config(
                config.zone_defs,
                geo_context=geo_context,
                default_authorization=config.default_zone_auth,
            )

        # Build weapons from config
        self._weapons: list[WeaponProfile] = []
        for wd in config.weapon_defs:
            self._weapons.append(WeaponProfile.from_config(wd))

        # Sub-components
        self._wez = WEZCalculator()
        self._feasibility = FeasibilityCalculator(
            pk_weight=config.pk_weight,
            tti_weight=config.tti_weight,
            threat_weight=config.threat_weight,
            max_tti_s=config.max_tti_s,
        )
        self._assigner = WeaponTargetAssigner(
            wez_calculator=self._wez,
            feasibility_calculator=self._feasibility,
            zone_manager=self._zone_manager,
        )

    @property
    def zone_manager(self) -> ZoneManager:
        return self._zone_manager

    @property
    def weapons(self) -> list[WeaponProfile]:
        return list(self._weapons)

    def add_weapon(self, weapon: WeaponProfile) -> None:
        self._weapons.append(weapon)

    def remove_weapon(self, weapon_id: str) -> None:
        self._weapons = [w for w in self._weapons if w.weapon_id != weapon_id]

    def evaluate(
        self,
        tracks: list[Any],
        current_time: float,
    ) -> EngagementPlan:
        """Run full engagement evaluation cycle.

        Args:
            tracks: List of EnhancedFusedTrack objects (or any objects
                    with the required attributes).
            current_time: Current simulation/wall time.

        Returns:
            EngagementPlan with assignments and zone statuses.
        """
        # Convert tracks to assignment-compatible dicts
        track_dicts = []
        zone_statuses = []

        for t in tracks:
            td = self._extract_track_dict(t)
            if td is None:
                continue

            # Resolve zone authorization for this track
            zone_auth = self._zone_manager.resolve_authorization(td["position"])
            td["zone_auth"] = zone_auth

            zone_statuses.append({
                "track_id": td["track_id"],
                "zone_authorization": zone_auth.value,
                "containing_zones": [
                    z.zone_id for z in
                    self._zone_manager.get_containing_zones(td["position"])
                ],
            })

            track_dicts.append(td)

        # Run assignment
        assignment = self._assigner.assign(
            self._weapons,
            track_dicts,
            default_zone_auth=self._config.default_zone_auth,
        )

        return EngagementPlan(
            timestamp=current_time,
            assignment=assignment,
            zone_statuses=zone_statuses,
        )

    def get_track_zone_auth(self, position: np.ndarray) -> ZoneAuth:
        """Resolve effective zone authorization for a position."""
        return self._zone_manager.resolve_authorization(position)

    @staticmethod
    def _extract_track_dict(track: Any) -> dict | None:
        """Extract assignment-compatible dict from a track object."""
        # Handle EnhancedFusedTrack objects
        position = None
        velocity = np.array([0.0, 0.0])

        # Try to get position from various track attributes
        if hasattr(track, "position_m") and track.position_m is not None:
            position = np.asarray(track.position_m, dtype=float)
        elif hasattr(track, "position") and track.position is not None:
            position = np.asarray(track.position, dtype=float)

        if position is None:
            # Try radar track
            if hasattr(track, "radar_track") and track.radar_track is not None:
                rt = track.radar_track
                if hasattr(rt, "position") and rt.position is not None:
                    position = np.asarray(rt.position, dtype=float)

        if position is None:
            return None

        # Extract velocity
        if hasattr(track, "velocity_mps") and track.velocity_mps is not None:
            # Scalar velocity — use azimuth to decompose
            v = float(track.velocity_mps)
            if hasattr(track, "azimuth_deg") and track.azimuth_deg is not None:
                az_rad = np.radians(float(track.azimuth_deg))
                velocity = np.array([v * np.sin(az_rad), v * np.cos(az_rad)])
            else:
                velocity = np.array([v, 0.0])
        elif hasattr(track, "radar_track") and track.radar_track is not None:
            rt = track.radar_track
            if hasattr(rt, "ekf") and hasattr(rt.ekf, "x"):
                # EKF state: [x, vx, y, vy, ...]
                velocity = np.array([float(rt.ekf.x[1]), float(rt.ekf.x[3])])

        # Track ID
        track_id = ""
        if hasattr(track, "fused_id"):
            track_id = str(track.fused_id)
        elif hasattr(track, "track_id"):
            track_id = str(track.track_id)

        # Engagement auth
        eng_auth = EngagementAuth.WEAPONS_HOLD
        if hasattr(track, "engagement_auth"):
            auth_val = track.engagement_auth
            if isinstance(auth_val, EngagementAuth):
                eng_auth = auth_val
            elif isinstance(auth_val, str):
                try:
                    eng_auth = EngagementAuth(auth_val)
                except ValueError:
                    pass

        return {
            "track_id": track_id,
            "position": position,
            "velocity": velocity,
            "threat_level": getattr(track, "threat_level", "LOW") or "LOW",
            "engagement_auth": eng_auth,
            "iff_identification": getattr(track, "iff_identification", "unknown") or "unknown",
            "is_jammed": getattr(track, "is_jammed", False),
        }

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        geo_context: Any = None,
    ) -> EngagementManager | None:
        """Build from OmegaConf or dict. Returns None if disabled."""
        config = EngagementConfig.from_omegaconf(cfg)
        if not config.enabled:
            return None
        return cls(config=config, geo_context=geo_context)
