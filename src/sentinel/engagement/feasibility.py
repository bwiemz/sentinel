"""Engagement feasibility scoring — Pk, TTI, quality, permission checks.

Computes probability of kill (Pk) via parametric model, time to intercept
(TTI) via kinematic geometry, and a combined quality score. Integrates
with ROE engine and zone authorization for engagement permission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinel.core.types import EngagementAuth, ZoneAuth
from sentinel.engagement.weapons import WEZResult, WeaponProfile


# Threat priority weights for quality scoring
_THREAT_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.75,
    "MEDIUM": 0.5,
    "LOW": 0.25,
}


@dataclass
class EngagementFeasibility:
    """Complete feasibility assessment for one weapon-target pair."""

    weapon_id: str
    track_id: str
    wez: WEZResult

    pk: float = 0.0
    tti_s: float = float("inf")
    quality_score: float = 0.0

    zone_authorization: ZoneAuth = ZoneAuth.WEAPONS_FREE
    roe_authorization: EngagementAuth = EngagementAuth.WEAPONS_HOLD
    engagement_permitted: bool = False

    def to_dict(self) -> dict:
        return {
            "weapon_id": self.weapon_id,
            "track_id": self.track_id,
            "pk": round(self.pk, 3),
            "tti_s": round(self.tti_s, 1) if math.isfinite(self.tti_s) else None,
            "quality_score": round(self.quality_score, 3),
            "zone_authorization": self.zone_authorization.value,
            "roe_authorization": self.roe_authorization.value,
            "engagement_permitted": self.engagement_permitted,
            "wez": self.wez.to_dict(),
        }


class FeasibilityCalculator:
    """Computes Pk, TTI, and quality score for weapon-target pairs."""

    def __init__(
        self,
        pk_weight: float = 0.4,
        tti_weight: float = 0.3,
        threat_weight: float = 0.3,
        max_tti_s: float = 120.0,
    ):
        # Normalize weights so they sum to 1.0
        total = pk_weight + tti_weight + threat_weight
        if total > 0 and abs(total - 1.0) > 1e-6:
            pk_weight /= total
            tti_weight /= total
            threat_weight /= total
        self._pk_weight = pk_weight
        self._tti_weight = tti_weight
        self._threat_weight = threat_weight
        self._max_tti_s = max(max_tti_s, 1.0)

    def evaluate(
        self,
        weapon: WeaponProfile,
        wez: WEZResult,
        threat_level: str = "LOW",
        engagement_auth: EngagementAuth = EngagementAuth.WEAPONS_HOLD,
        zone_auth: ZoneAuth = ZoneAuth.WEAPONS_FREE,
        is_jammed: bool = False,
    ) -> EngagementFeasibility:
        """Full feasibility evaluation for a weapon-target pair."""
        pk = self.compute_pk(weapon, wez, is_jammed)
        tti = self.compute_tti(weapon, wez)
        permitted = self.check_engagement_permitted(zone_auth, engagement_auth)
        quality = self.compute_quality_score(
            pk, tti, threat_level, engagement_auth, zone_auth,
        )

        return EngagementFeasibility(
            weapon_id=weapon.weapon_id,
            track_id=wez.track_id,
            wez=wez,
            pk=pk,
            tti_s=tti,
            quality_score=quality,
            zone_authorization=zone_auth,
            roe_authorization=engagement_auth,
            engagement_permitted=permitted,
        )

    def compute_pk(
        self,
        weapon: WeaponProfile,
        wez: WEZResult,
        is_jammed: bool = False,
    ) -> float:
        """Parametric Pk estimation.

        Pk = pk_base * range_factor * speed_factor * aspect_factor * ecm_factor

        Returns 0.0 if target is outside WEZ or weapon has no rounds.
        """
        if not wez.feasible or weapon.rounds_remaining <= 0:
            return 0.0

        # Range factor: 1.0 at optimal, quadratic falloff beyond
        range_factor = self._range_factor(
            wez.slant_range_m,
            weapon.min_range_m,
            weapon.optimal_range_m,
            weapon.max_range_m,
            weapon.pk_range_falloff,
        )

        # Speed factor: penalty increases with target speed
        speed_ratio = wez.target_speed_mps / max(weapon.max_target_speed_mps, 1.0)
        speed_factor = max(0.0, 1.0 - weapon.pk_speed_penalty * speed_ratio * speed_ratio)

        # Aspect factor: 1.0 for all-aspect, reduced for limited weapons
        aspect_factor = 1.0
        if weapon.max_aspect_angle_deg < 180.0:
            # Linear degradation toward weapon's aspect limits
            aspect_range = weapon.max_aspect_angle_deg - weapon.min_aspect_angle_deg
            if aspect_range > 1e-3:
                center = (weapon.min_aspect_angle_deg + weapon.max_aspect_angle_deg) / 2.0
                deviation = abs(wez.aspect_angle_deg - center) / (aspect_range / 2.0)
                aspect_factor = max(0.0, 1.0 - 0.3 * deviation)

        # ECM/jamming factor
        ecm_factor = 1.0 - weapon.pk_ecm_penalty if is_jammed else 1.0

        pk = weapon.pk_base * range_factor * speed_factor * aspect_factor * ecm_factor
        return max(0.0, min(1.0, pk))

    def compute_tti(self, weapon: WeaponProfile, wez: WEZResult) -> float:
        """Time to intercept via simple kinematic model.

        Head-on: TTI = range / (weapon_speed + closing_speed)
        Pursuit: TTI = range / (weapon_speed - |opening_speed|)
        """
        if not wez.feasible or wez.slant_range_m < 1.0 or weapon.weapon_speed_mps <= 0:
            return 0.0 if (wez.feasible and weapon.weapon_speed_mps > 0) else float("inf")

        if wez.closing_speed_mps > 0:
            # Target approaching — head-on geometry
            effective_speed = weapon.weapon_speed_mps + wez.closing_speed_mps
        else:
            # Target receding — pursuit geometry
            effective_speed = weapon.weapon_speed_mps - abs(wez.closing_speed_mps)

        if effective_speed <= 0:
            return float("inf")  # Can't catch target

        return wez.slant_range_m / effective_speed

    def compute_quality_score(
        self,
        pk: float,
        tti_s: float,
        threat_level: str,
        roe_auth: EngagementAuth,
        zone_auth: ZoneAuth,
    ) -> float:
        """Combined engagement quality score [0, 1].

        quality = w_pk * pk + w_tti * tti_factor + w_threat * threat_weight
        Modified by zone/ROE constraints.
        """
        # Base quality from Pk, TTI, and threat
        tti_factor = max(0.0, 1.0 - tti_s / self._max_tti_s) if math.isfinite(tti_s) else 0.0
        threat_w = _THREAT_WEIGHTS.get(threat_level.upper(), 0.25)

        quality = (
            self._pk_weight * pk
            + self._tti_weight * tti_factor
            + self._threat_weight * threat_w
        )

        # Zone/ROE modifiers
        if zone_auth == ZoneAuth.NO_FIRE:
            return 0.0
        if roe_auth == EngagementAuth.HOLD_FIRE:
            return 0.0
        if zone_auth == ZoneAuth.RESTRICTED_FIRE:
            quality *= 0.3
        if zone_auth == ZoneAuth.SELF_DEFENSE_ONLY:
            quality *= 0.5
        if roe_auth == EngagementAuth.WEAPONS_HOLD:
            quality *= 0.3

        return max(0.0, min(1.0, quality))

    @staticmethod
    def check_engagement_permitted(
        zone_auth: ZoneAuth,
        roe_auth: EngagementAuth,
    ) -> bool:
        """Check if both zone and ROE permit engagement.

        Engagement requires:
        - Zone is not NO_FIRE
        - ROE is not HOLD_FIRE
        - At least one of: zone=WEAPONS_FREE + ROE allows, or
          zone=SELF_DEFENSE_ONLY + ROE=WEAPONS_FREE
        """
        if zone_auth == ZoneAuth.NO_FIRE:
            return False
        if roe_auth == EngagementAuth.HOLD_FIRE:
            return False
        if zone_auth == ZoneAuth.WEAPONS_FREE:
            return roe_auth in (
                EngagementAuth.WEAPONS_FREE,
                EngagementAuth.WEAPONS_TIGHT,
            )
        if zone_auth == ZoneAuth.SELF_DEFENSE_ONLY:
            return roe_auth == EngagementAuth.WEAPONS_FREE
        if zone_auth == ZoneAuth.RESTRICTED_FIRE:
            return roe_auth == EngagementAuth.WEAPONS_FREE
        return False

    @staticmethod
    def _range_factor(
        slant_range: float,
        min_range: float,
        optimal_range: float,
        max_range: float,
        falloff: float,
    ) -> float:
        """Compute range-dependent Pk factor.

        1.0 between min_range and optimal_range.
        Quadratic falloff from optimal_range to max_range.
        Degraded inside min_range (too close for guidance).
        """
        if slant_range < min_range:
            # Too close — linear degradation
            if min_range <= 0:
                return 1.0
            return max(0.0, slant_range / min_range)
        if slant_range <= optimal_range:
            return 1.0
        if slant_range >= max_range:
            return 0.0
        # Quadratic falloff beyond optimal
        denom = max_range - optimal_range
        if denom <= 0:
            return 0.0  # No falloff zone — beyond envelope
        ratio = (slant_range - optimal_range) / denom
        return max(0.0, 1.0 - ratio ** falloff)
