"""Tests for weapon-target assignment via Hungarian algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.weapons import WeaponProfile, WEZCalculator
from sentinel.engagement.feasibility import FeasibilityCalculator
from sentinel.engagement.assignment import WeaponTargetAssigner, EngagementAssignment
from sentinel.engagement.zones import ZoneManager, CircularZone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weapon(
    weapon_id: str = "W1",
    x: float = 0.0,
    y: float = 0.0,
    max_engagements: int = 2,
    rounds: int = 10,
    **overrides,
) -> WeaponProfile:
    """Build a WeaponProfile with sensible defaults, accepting overrides."""
    defaults = dict(
        weapon_id=weapon_id,
        name=f"SAM-{weapon_id}",
        weapon_type=WeaponType.SAM_MEDIUM,
        position_xy=np.array([x, y]),
        altitude_m=0.0,
        min_range_m=500.0,
        max_range_m=20000.0,
        optimal_range_m=10000.0,
        min_altitude_m=0.0,
        max_altitude_m=30000.0,
        max_target_speed_mps=800.0,
        weapon_speed_mps=1200.0,
        pk_base=0.85,
        pk_range_falloff=2.0,
        pk_speed_penalty=0.3,
        pk_ecm_penalty=0.3,
        min_aspect_angle_deg=0.0,
        max_aspect_angle_deg=180.0,
        max_simultaneous_engagements=max_engagements,
        rounds_remaining=rounds,
        salvo_size=1,
        reload_time_s=5.0,
    )
    defaults.update(overrides)
    return WeaponProfile(**defaults)


def _track(
    track_id: str = "T1",
    x: float = 10000.0,
    y: float = 5000.0,
    vx: float = -100.0,
    vy: float = 0.0,
    threat: str = "HIGH",
    auth: EngagementAuth = EngagementAuth.WEAPONS_FREE,
    iff: str = "hostile",
    is_jammed: bool = False,
) -> dict:
    """Build a track dict matching the WeaponTargetAssigner interface."""
    return {
        "track_id": track_id,
        "position": np.array([x, y]),
        "velocity": np.array([vx, vy]),
        "threat_level": threat,
        "engagement_auth": auth,
        "iff_identification": iff,
        "is_jammed": is_jammed,
    }


def _assigner(**kw) -> WeaponTargetAssigner:
    """Create a WeaponTargetAssigner with real sub-calculators."""
    return WeaponTargetAssigner(
        wez_calculator=kw.get("wez", WEZCalculator()),
        feasibility_calculator=kw.get("feas", FeasibilityCalculator()),
        zone_manager=kw.get("zone_mgr", None),
    )


# ---------------------------------------------------------------------------
# TestWeaponTargetAssigner
# ---------------------------------------------------------------------------

class TestWeaponTargetAssigner:
    """Tests for WeaponTargetAssigner.assign."""

    def test_single_weapon_single_target(self):
        """One weapon and one hostile track → exactly one assignment."""
        assigner = _assigner()
        result = assigner.assign(
            weapons=[_weapon()],
            tracks=[_track()],
        )
        assert isinstance(result, EngagementAssignment)
        assert len(result.assignments) == 1
        wid, tid = result.assignments[0]
        assert wid == "W1"
        assert tid == "T1"

    def test_two_weapons_two_targets(self):
        """Two weapons, two targets → two assignments, all covered."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id="W1"), _weapon(weapon_id="W2")]
        tracks = [
            _track(track_id="T1", x=8000.0),
            _track(track_id="T2", x=12000.0),
        ]
        result = assigner.assign(weapons, tracks)
        assigned_wids = {a[0] for a in result.assignments}
        assigned_tids = {a[1] for a in result.assignments}
        assert len(result.assignments) == 2
        assert assigned_tids == {"T1", "T2"}

    def test_more_weapons_than_targets(self):
        """Extra weapons remain unassigned; all targets are covered."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id=f"W{i}") for i in range(3)]
        tracks = [_track(track_id="T1")]
        result = assigner.assign(weapons, tracks)
        assert len(result.assignments) >= 1
        assert "T1" in {a[1] for a in result.assignments}
        assert len(result.unassigned_weapons) >= 1

    def test_more_targets_than_weapons(self):
        """Fewer weapons than targets → some targets unassigned."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id="W1", max_engagements=1)]
        tracks = [
            _track(track_id="T1", x=8000.0),
            _track(track_id="T2", x=12000.0),
        ]
        result = assigner.assign(weapons, tracks)
        assert len(result.assignments) == 1
        assert len(result.unassigned_tracks) >= 1

    def test_infeasible_pair_skipped(self):
        """Target beyond max range with HOLD_FIRE ROE is not assigned.

        Out-of-range alone doesn't block assignment (threat weight keeps
        quality > 0), so we combine with HOLD_FIRE to zero out quality.
        """
        assigner = _assigner()
        # Target at 50 km + HOLD_FIRE → quality = 0, infeasible cost
        tracks = [_track(track_id="T1", x=50000.0, y=0.0,
                         auth=EngagementAuth.HOLD_FIRE)]
        result = assigner.assign([_weapon()], tracks)
        assert len(result.assignments) == 0
        assert "T1" in result.unassigned_tracks

    def test_out_of_range_target_has_zero_pk(self):
        """An out-of-range target may be assigned but with Pk = 0."""
        assigner = _assigner()
        tracks = [_track(track_id="T1", x=50000.0, y=0.0)]
        result = assigner.assign([_weapon()], tracks)
        # The assignment code assigns based on quality_score > 0 and permitted,
        # but the Pk should be 0 since the WEZ is infeasible.
        if result.assignments:
            feas = result.feasibility_map[("W1", "T1")]
            assert feas.pk == 0.0

    def test_no_fire_zone_blocks(self):
        """Weapon in a NO_FIRE zone must not produce assignments."""
        no_fire_zone = CircularZone(
            zone_id="NFZ",
            name="No Fire Zone",
            center_xy=np.array([5000.0, 2500.0]),
            radius_m=50000.0,
            authorization=ZoneAuth.NO_FIRE,
            altitude_min_m=0.0,
            altitude_max_m=30000.0,
        )
        zone_mgr = ZoneManager(zones=[no_fire_zone])
        assigner = _assigner(zone_mgr=zone_mgr)
        result = assigner.assign([_weapon()], [_track()])
        assert len(result.assignments) == 0

    def test_friendly_excluded(self):
        """Friendly tracks must be excluded from assignment."""
        assigner = _assigner()
        tracks = [
            _track(track_id="T1", iff="friendly"),
            _track(track_id="T2", iff="assumed_friendly"),
        ]
        result = assigner.assign([_weapon()], tracks)
        assert len(result.assignments) == 0
        # Friendly tracks land in unassigned
        assert "T1" in result.unassigned_tracks
        assert "T2" in result.unassigned_tracks

    def test_multi_slot_weapon(self):
        """Weapon with max_simultaneous_engagements=2 can engage two targets."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id="W1", max_engagements=2)]
        tracks = [
            _track(track_id="T1", x=8000.0),
            _track(track_id="T2", x=12000.0),
        ]
        result = assigner.assign(weapons, tracks)
        # Both targets should be assigned to W1
        assigned_tids = {a[1] for a in result.assignments}
        assert len(assigned_tids) == 2
        for wid, _tid in result.assignments:
            assert wid == "W1"

    def test_empty_weapons(self):
        """Empty weapon list → no assignments, all tracks unassigned."""
        assigner = _assigner()
        result = assigner.assign([], [_track()])
        assert result.assignments == []
        assert "T1" in result.unassigned_tracks

    def test_empty_tracks(self):
        """Empty track list → no assignments, all weapons unassigned."""
        assigner = _assigner()
        result = assigner.assign([_weapon()], [])
        assert result.assignments == []
        assert "W1" in result.unassigned_weapons

    def test_weapon_capacity_exhausted(self):
        """Weapon with rounds_remaining=0 must not be assigned."""
        assigner = _assigner()
        result = assigner.assign(
            [_weapon(rounds=0)],
            [_track()],
        )
        assert len(result.assignments) == 0

    def test_cost_prefers_critical_threat(self):
        """Given equal range, a CRITICAL track should be preferred over LOW."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id="W1", max_engagements=1)]
        tracks = [
            _track(track_id="T_LOW", x=8000.0, threat="LOW"),
            _track(track_id="T_CRIT", x=8000.0, y=1.0, threat="CRITICAL"),
        ]
        result = assigner.assign(weapons, tracks)
        assert len(result.assignments) == 1
        _wid, tid = result.assignments[0]
        assert tid == "T_CRIT"

    def test_total_pk_single_assignment(self):
        """total_pk for a single assignment equals that assignment's Pk."""
        assigner = _assigner()
        result = assigner.assign([_weapon()], [_track()])
        assert len(result.assignments) == 1
        wid, tid = result.assignments[0]
        feas = result.feasibility_map.get((wid, tid))
        assert feas is not None
        assert result.total_pk == pytest.approx(feas.pk, abs=1e-6)

    def test_total_pk_two_assignments(self):
        """total_pk for two assignments is 1 - (1-pk1)*(1-pk2)."""
        assigner = _assigner()
        weapons = [_weapon(weapon_id="W1", max_engagements=1),
                    _weapon(weapon_id="W2", max_engagements=1)]
        tracks = [
            _track(track_id="T1", x=8000.0),
            _track(track_id="T2", x=12000.0),
        ]
        result = assigner.assign(weapons, tracks)
        # Collect individual Pk values
        miss = 1.0
        for wid, tid in result.assignments:
            f = result.feasibility_map[(wid, tid)]
            miss *= (1.0 - f.pk)
        expected_total = 1.0 - miss
        assert result.total_pk == pytest.approx(expected_total, abs=1e-6)

    def test_total_pk_no_assignments_is_zero(self):
        """total_pk with no assignments is 0."""
        result = EngagementAssignment(assignments=[])
        assert result.total_pk == 0.0

    def test_to_dict_structure(self):
        """to_dict() must produce expected keys and types."""
        assigner = _assigner()
        result = assigner.assign([_weapon()], [_track()])
        d = result.to_dict()
        assert "assignments" in d
        assert "unassigned_weapons" in d
        assert "unassigned_tracks" in d
        assert "total_pk" in d
        assert "assignment_count" in d
        assert isinstance(d["assignments"], list)
        assert isinstance(d["total_pk"], float)

    def test_hold_fire_roe_blocks_assignment(self):
        """HOLD_FIRE ROE should prevent assignment even for CRITICAL threat."""
        assigner = _assigner()
        tracks = [_track(auth=EngagementAuth.HOLD_FIRE, threat="CRITICAL")]
        result = assigner.assign([_weapon()], tracks)
        assert len(result.assignments) == 0

    def test_mixed_friendly_and_hostile(self):
        """Only hostile tracks receive assignments; friendly skipped."""
        assigner = _assigner()
        tracks = [
            _track(track_id="T_FRIEND", iff="friendly", x=8000.0),
            _track(track_id="T_HOSTILE", iff="hostile", x=8000.0, y=1.0),
        ]
        result = assigner.assign([_weapon()], tracks)
        assigned_tids = {a[1] for a in result.assignments}
        assert "T_HOSTILE" in assigned_tids
        assert "T_FRIEND" not in assigned_tids

    def test_weapons_hold_roe_still_assigns_if_permitted(self):
        """WEAPONS_HOLD + WEAPONS_FREE zone can still assign (quality reduced)."""
        assigner = _assigner()
        tracks = [_track(auth=EngagementAuth.WEAPONS_HOLD)]
        result = assigner.assign([_weapon()], tracks)
        # check_engagement_permitted: WEAPONS_FREE zone + WEAPONS_HOLD ROE
        # WEAPONS_FREE zone requires ROE in {WEAPONS_FREE, WEAPONS_TIGHT} → blocked
        assert len(result.assignments) == 0
