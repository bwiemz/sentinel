"""Weapon-target assignment via Hungarian algorithm.

Optimally assigns weapons to threatening tracks based on feasibility
scores, respecting weapon capacity, zone constraints, and ROE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import EngagementAuth, ZoneAuth
from sentinel.engagement.feasibility import EngagementFeasibility, FeasibilityCalculator
from sentinel.engagement.weapons import WEZCalculator, WEZResult, WeaponProfile
from sentinel.engagement.zones import ZoneManager

logger = logging.getLogger(__name__)

_INFEASIBLE_COST = 1e6


@dataclass
class EngagementAssignment:
    """Result of weapon-target assignment optimization."""

    assignments: list[tuple[str, str]]  # (weapon_id, track_id) pairs
    feasibility_map: dict[tuple[str, str], EngagementFeasibility] = field(
        default_factory=dict,
    )
    unassigned_weapons: list[str] = field(default_factory=list)
    unassigned_tracks: list[str] = field(default_factory=list)

    @property
    def total_pk(self) -> float:
        """Product of individual assignment Pk values (salvo Pk)."""
        if not self.assignments:
            return 0.0
        pk = 1.0
        for wid, tid in self.assignments:
            f = self.feasibility_map.get((wid, tid))
            if f:
                pk *= (1.0 - f.pk)
        return 1.0 - pk  # 1 - product of miss probabilities

    def to_dict(self) -> dict:
        return {
            "assignments": [
                {"weapon_id": w, "track_id": t} for w, t in self.assignments
            ],
            "unassigned_weapons": self.unassigned_weapons,
            "unassigned_tracks": self.unassigned_tracks,
            "total_pk": round(self.total_pk, 3),
            "assignment_count": len(self.assignments),
        }


class WeaponTargetAssigner:
    """Optimal weapon-target assignment using Hungarian algorithm.

    Respects max simultaneous engagements, zone constraints,
    ROE constraints, and weapon capacity (rounds remaining).
    """

    def __init__(
        self,
        wez_calculator: WEZCalculator | None = None,
        feasibility_calculator: FeasibilityCalculator | None = None,
        zone_manager: ZoneManager | None = None,
    ):
        self._wez = wez_calculator or WEZCalculator()
        self._feasibility = feasibility_calculator or FeasibilityCalculator()
        self._zone_manager = zone_manager

    def assign(
        self,
        weapons: list[WeaponProfile],
        tracks: list[dict],
        default_zone_auth: ZoneAuth = ZoneAuth.WEAPONS_FREE,
    ) -> EngagementAssignment:
        """Compute optimal weapon-target assignment.

        Args:
            weapons: Available weapon systems.
            tracks: List of track dicts with keys:
                - track_id: str
                - position: np.ndarray [x, y] or [x, y, z]
                - velocity: np.ndarray [vx, vy] or [vx, vy, vz]
                - threat_level: str (LOW/MEDIUM/HIGH/CRITICAL)
                - engagement_auth: EngagementAuth
                - iff_identification: str (optional)
                - is_jammed: bool (optional)
        """
        if not weapons or not tracks:
            return EngagementAssignment(
                assignments=[],
                unassigned_weapons=[w.weapon_id for w in weapons],
                unassigned_tracks=[t["track_id"] for t in tracks],
            )

        # Filter out friendly tracks
        engageable_tracks = [
            t for t in tracks
            if t.get("iff_identification", "unknown").lower()
            not in ("friendly", "assumed_friendly")
        ]
        friendly_ids = [
            t["track_id"] for t in tracks
            if t["track_id"] not in {et["track_id"] for et in engageable_tracks}
        ]

        if not engageable_tracks:
            return EngagementAssignment(
                assignments=[],
                unassigned_weapons=[w.weapon_id for w in weapons],
                unassigned_tracks=friendly_ids,
            )

        # Expand weapons into individual engagement slots
        slots = self._expand_weapon_slots(weapons)

        # Build cost matrix and feasibility map
        cost, feas_map = self._build_cost_matrix(
            slots, engageable_tracks, default_zone_auth,
        )

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # Extract valid assignments (filter infeasible)
        assignments: list[tuple[str, str]] = []
        assigned_weapon_ids: set[str] = set()
        assigned_track_ids: set[str] = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= _INFEASIBLE_COST:
                continue
            wid = slots[r][0].weapon_id
            tid = engageable_tracks[c]["track_id"]
            feas = feas_map.get((wid, tid))
            if feas and feas.engagement_permitted:
                assignments.append((wid, tid))
                assigned_weapon_ids.add(wid)
                assigned_track_ids.add(tid)

        unassigned_weapons = [
            w.weapon_id for w in weapons if w.weapon_id not in assigned_weapon_ids
        ]
        unassigned_tracks = [
            t["track_id"] for t in engageable_tracks
            if t["track_id"] not in assigned_track_ids
        ] + friendly_ids

        return EngagementAssignment(
            assignments=assignments,
            feasibility_map=feas_map,
            unassigned_weapons=unassigned_weapons,
            unassigned_tracks=unassigned_tracks,
        )

    def _build_cost_matrix(
        self,
        slots: list[tuple[WeaponProfile, int]],
        tracks: list[dict],
        default_zone_auth: ZoneAuth,
    ) -> tuple[np.ndarray, dict[tuple[str, str], EngagementFeasibility]]:
        """Build cost matrix and feasibility map."""
        n_slots = len(slots)
        n_tracks = len(tracks)
        cost = np.full((n_slots, n_tracks), _INFEASIBLE_COST)
        feas_map: dict[tuple[str, str], EngagementFeasibility] = {}

        for i, (weapon, slot_idx) in enumerate(slots):
            if weapon.rounds_remaining <= slot_idx * weapon.salvo_size:
                continue  # This slot has no ammo

            for j, track in enumerate(tracks):
                tid = track["track_id"]
                pos = np.asarray(track["position"], dtype=float)
                vel = np.asarray(track["velocity"], dtype=float)

                # WEZ evaluation
                wez = self._wez.evaluate(weapon, pos, vel, tid)

                # Zone authorization
                zone_auth = default_zone_auth
                if self._zone_manager is not None:
                    zone_auth = self._zone_manager.resolve_authorization(pos)

                # Feasibility
                roe_auth = track.get(
                    "engagement_auth", EngagementAuth.WEAPONS_HOLD,
                )
                if isinstance(roe_auth, str):
                    try:
                        roe_auth = EngagementAuth(roe_auth)
                    except ValueError:
                        roe_auth = EngagementAuth.WEAPONS_HOLD

                feas = self._feasibility.evaluate(
                    weapon, wez,
                    threat_level=track.get("threat_level", "LOW"),
                    engagement_auth=roe_auth,
                    zone_auth=zone_auth,
                    is_jammed=track.get("is_jammed", False),
                )

                key = (weapon.weapon_id, tid)
                # Keep best feasibility per weapon-track pair across slots
                if key not in feas_map or feas.quality_score > feas_map[key].quality_score:
                    feas_map[key] = feas

                if feas.engagement_permitted and feas.quality_score > 0:
                    cost[i, j] = 1.0 - feas.quality_score

        return cost, feas_map

    @staticmethod
    def _expand_weapon_slots(
        weapons: list[WeaponProfile],
    ) -> list[tuple[WeaponProfile, int]]:
        """Expand weapons into individual engagement slots.

        A weapon with max_simultaneous_engagements=2 becomes 2 rows.
        """
        slots: list[tuple[WeaponProfile, int]] = []
        for w in weapons:
            for s in range(w.max_simultaneous_engagements):
                slots.append((w, s))
        return slots
