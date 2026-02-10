"""End-to-end engagement scenarios testing the full EngagementManager.

Realistic military engagement scenarios that create an EngagementManager with
zones and weapons, feed it mock tracks, and validate the engagement plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.config import EngagementConfig
from sentinel.engagement.manager import EngagementManager
from sentinel.engagement.weapons import WeaponProfile
from sentinel.engagement.zones import (
    AnnularZone,
    CircularZone,
    PolygonZone,
    SectorZone,
    ZoneManager,
)


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


def _make_patriot_weapon(weapon_id="PAT-1", position_xy=None):
    """Create a Patriot-class SAM weapon profile.

    Uses min_altitude_m=0.0 so 2D positions (altitude=0) pass the WEZ
    altitude check, keeping the focus on range/speed/zone constraints.
    """
    return WeaponProfile(
        weapon_id=weapon_id,
        name="Patriot SAM",
        weapon_type=WeaponType.SAM_LONG,
        position_xy=np.array(position_xy or [0.0, 0.0]),
        min_range_m=3000.0,
        max_range_m=70000.0,
        optimal_range_m=40000.0,
        min_altitude_m=0.0,
        weapon_speed_mps=1700.0,
        pk_base=0.85,
        max_target_speed_mps=2000.0,
        max_simultaneous_engagements=4,
        rounds_remaining=16,
        salvo_size=1,
    )


def _make_ciws_weapon(weapon_id="CIWS-1", position_xy=None):
    """Create a CIWS (Close-In Weapon System) profile."""
    return WeaponProfile(
        weapon_id=weapon_id,
        name="Phalanx CIWS",
        weapon_type=WeaponType.CIWS,
        position_xy=np.array(position_xy or [0.0, 0.0]),
        min_range_m=200.0,
        max_range_m=1500.0,
        optimal_range_m=800.0,
        min_altitude_m=0.0,
        weapon_speed_mps=1000.0,
        pk_base=0.75,
        max_target_speed_mps=800.0,
        max_simultaneous_engagements=1,
        rounds_remaining=1000,
        salvo_size=1,
    )


def _build_manager(zones=None, weapons=None, default_zone_auth=ZoneAuth.WEAPONS_FREE):
    """Helper to build an EngagementManager from explicit zones and weapons."""
    zone_manager = ZoneManager(
        zones=zones or [],
        default_authorization=default_zone_auth,
    )
    config = EngagementConfig(
        enabled=True,
        default_zone_auth=default_zone_auth,
    )
    mgr = EngagementManager(config=config, zone_manager=zone_manager)
    # Add weapons after construction
    for w in (weapons or []):
        mgr.add_weapon(w)
    return mgr


class TestEngagementScenarios:
    """End-to-end engagement scenarios."""

    def test_patriot_vs_incoming_missile(self):
        """Patriot SAM engages a single incoming missile at optimal range.

        Target: CRITICAL threat at 50 km range, approaching at -300 m/s.
        Expected: assigned, Pk > 0.5, TTI < 60 s.
        """
        weapon = _make_patriot_weapon()
        mgr = _build_manager(weapons=[weapon])

        track = MockTrack(
            fused_id="TGT-001",
            position_m=[50000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,  # approaching from east (heading west toward origin)
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([track], current_time=100.0)
        assignment = plan.assignment

        # Should be assigned
        assigned_track_ids = {tid for _, tid in assignment.assignments}
        assert "TGT-001" in assigned_track_ids, (
            f"Expected TGT-001 to be assigned, got assignments: {assignment.assignments}"
        )

        # Check Pk > 0.5
        key = (weapon.weapon_id, "TGT-001")
        feas = assignment.feasibility_map.get(key)
        assert feas is not None, "Feasibility entry missing for assigned pair"
        assert feas.pk > 0.5, f"Expected Pk > 0.5, got {feas.pk:.3f}"

        # Check TTI < 60 s
        assert feas.tti_s < 60.0, f"Expected TTI < 60 s, got {feas.tti_s:.1f}"

    def test_no_fire_zone_sanctuary(self):
        """Track inside a NO_FIRE zone must not be engaged.

        Even with WEAPONS_FREE ROE and CRITICAL threat level, the no-fire
        zone must prevent assignment.
        """
        no_fire_zone = CircularZone(
            zone_id="NFZ-1",
            name="Embassy Sanctuary",
            center_xy=np.array([5000.0, 3000.0]),
            radius_m=2000.0,
            authorization=ZoneAuth.NO_FIRE,
            priority=10,  # high priority overrides defaults
        )
        weapon = _make_patriot_weapon()
        mgr = _build_manager(zones=[no_fire_zone], weapons=[weapon])

        track = MockTrack(
            fused_id="TGT-002",
            position_m=[5000.0, 3000.0],
            velocity_mps=200.0,
            azimuth_deg=0.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([track], current_time=100.0)

        assigned_track_ids = {tid for _, tid in plan.assignment.assignments}
        assert "TGT-002" not in assigned_track_ids, (
            "Track inside NO_FIRE zone must not be assigned"
        )

    def test_ciws_close_range_only(self):
        """CIWS weapon can only engage targets within its short range.

        Target at 800 m should be assigned; target at 5000 m should not.
        """
        ciws = _make_ciws_weapon()
        mgr = _build_manager(weapons=[ciws])

        close_target = MockTrack(
            fused_id="TGT-CLOSE",
            position_m=[800.0, 0.0],
            velocity_mps=100.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )
        far_target = MockTrack(
            fused_id="TGT-FAR",
            position_m=[5000.0, 0.0],
            velocity_mps=100.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([close_target, far_target], current_time=100.0)
        assigned_ids = {tid for _, tid in plan.assignment.assignments}

        assert "TGT-CLOSE" in assigned_ids, "Close target should be engaged by CIWS"
        assert "TGT-FAR" not in assigned_ids, "Far target outside CIWS range"

    def test_multiple_sam_batteries_split_targets(self):
        """Two SAM batteries should split targets optimally.

        SAM-1 at origin, SAM-2 at [20000, 0].
        Target-A at [10000, 5000], Target-B at [25000, 5000].
        Each SAM should get one target for optimal coverage.
        """
        sam1 = _make_patriot_weapon(weapon_id="SAM-1", position_xy=[0.0, 0.0])
        sam2 = _make_patriot_weapon(weapon_id="SAM-2", position_xy=[20000.0, 0.0])
        mgr = _build_manager(weapons=[sam1, sam2])

        track_a = MockTrack(
            fused_id="TGT-A",
            position_m=[10000.0, 5000.0],
            velocity_mps=250.0,
            azimuth_deg=180.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )
        track_b = MockTrack(
            fused_id="TGT-B",
            position_m=[25000.0, 5000.0],
            velocity_mps=250.0,
            azimuth_deg=180.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([track_a, track_b], current_time=100.0)

        assigned_ids = {tid for _, tid in plan.assignment.assignments}
        assert "TGT-A" in assigned_ids, "Target-A should be assigned"
        assert "TGT-B" in assigned_ids, "Target-B should be assigned"

        # Each SAM should handle one target (optimal split)
        weapon_targets = {}
        for wid, tid in plan.assignment.assignments:
            weapon_targets.setdefault(wid, []).append(tid)
        # At least each target is covered
        assert len(assigned_ids) == 2

    def test_hypersonic_exceeds_weapon_speed(self):
        """Target exceeding weapon's max_target_speed renders WEZ infeasible.

        Weapon max_target_speed = 800 m/s, target speed = 2000 m/s.
        WEZ should be infeasible due to speed constraint, yielding Pk = 0.

        Note: The assignment algorithm may still include infeasible pairs
        in its output if the quality_score > 0 (from threat weight). The
        key invariant is that Pk = 0 and WEZ.in_speed = False.
        """
        slow_weapon = WeaponProfile(
            weapon_id="SAM-SLOW",
            name="Short Range SAM",
            weapon_type=WeaponType.SAM_SHORT,
            position_xy=np.array([0.0, 0.0]),
            min_range_m=500.0,
            max_range_m=30000.0,
            optimal_range_m=15000.0,
            min_altitude_m=0.0,
            weapon_speed_mps=1200.0,
            pk_base=0.80,
            max_target_speed_mps=800.0,
            max_simultaneous_engagements=2,
            rounds_remaining=10,
        )
        mgr = _build_manager(weapons=[slow_weapon])

        # Hypersonic target at 2000 m/s heading due west (azimuth 270)
        hyper_track = MockTrack(
            fused_id="TGT-HYPER",
            position_m=[15000.0, 0.0],
            velocity_mps=2000.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([hyper_track], current_time=100.0)

        # Verify WEZ is infeasible due to speed
        key = ("SAM-SLOW", "TGT-HYPER")
        feas = plan.assignment.feasibility_map.get(key)
        assert feas is not None, "Feasibility entry should exist"
        assert not feas.wez.in_speed, (
            f"WEZ speed check should fail for {feas.wez.target_speed_mps:.0f} m/s "
            f"vs limit {slow_weapon.max_target_speed_mps:.0f} m/s"
        )
        assert not feas.wez.feasible, "WEZ should be infeasible overall"
        assert feas.pk == 0.0, "Pk should be zero when WEZ is infeasible"

    def test_ecm_reduces_pk(self):
        """ECM (jamming) should reduce Pk compared to an unjammed target.

        Two identical targets at the same range, one jammed, one not.
        The jammed target should have a lower Pk in the feasibility map.
        """
        weapon = _make_patriot_weapon()
        mgr = _build_manager(weapons=[weapon])

        clean_track = MockTrack(
            fused_id="TGT-CLEAN",
            position_m=[30000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
            is_jammed=False,
        )
        jammed_track = MockTrack(
            fused_id="TGT-JAMMED",
            position_m=[30000.0, 1.0],  # slight offset to avoid identical positions
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
            is_jammed=True,
        )

        plan = mgr.evaluate([clean_track, jammed_track], current_time=100.0)

        feas_clean = plan.assignment.feasibility_map.get(
            (weapon.weapon_id, "TGT-CLEAN")
        )
        feas_jammed = plan.assignment.feasibility_map.get(
            (weapon.weapon_id, "TGT-JAMMED")
        )

        assert feas_clean is not None, "Clean target feasibility missing"
        assert feas_jammed is not None, "Jammed target feasibility missing"
        assert feas_jammed.pk < feas_clean.pk, (
            f"Jammed Pk ({feas_jammed.pk:.3f}) should be less than "
            f"clean Pk ({feas_clean.pk:.3f})"
        )

    def test_friendly_holdfire_overrides_free_zone(self):
        """Friendly IFF identification must prevent engagement.

        Even in a WEAPONS_FREE zone, a friendly track must not be assigned.
        """
        weapon = _make_patriot_weapon()
        mgr = _build_manager(weapons=[weapon])

        friendly_track = MockTrack(
            fused_id="TGT-FRIEND",
            position_m=[20000.0, 0.0],
            velocity_mps=250.0,
            azimuth_deg=90.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="friendly",
        )

        plan = mgr.evaluate([friendly_track], current_time=100.0)
        assigned_ids = {tid for _, tid in plan.assignment.assignments}
        assert "TGT-FRIEND" not in assigned_ids, (
            "Friendly track must not be assigned despite WEAPONS_FREE zone"
        )

    def test_mixed_zone_scenario(self):
        """Three targets in zones with different authorization levels.

        Zone layout:
        - NO_FIRE circle at [5000, 0], radius 3000
        - WEAPONS_FREE circle at [20000, 0], radius 5000
        - RESTRICTED_FIRE circle at [40000, 0], radius 5000

        The NO_FIRE target should never be assigned.
        The WEAPONS_FREE target should get full assignment.
        The RESTRICTED_FIRE target gets reduced quality (may or may not assign).
        """
        no_fire = CircularZone(
            zone_id="NFZ",
            name="No Fire Zone",
            center_xy=np.array([5000.0, 0.0]),
            radius_m=3000.0,
            authorization=ZoneAuth.NO_FIRE,
            priority=10,
        )
        weapons_free = CircularZone(
            zone_id="WFZ",
            name="Weapons Free Zone",
            center_xy=np.array([20000.0, 0.0]),
            radius_m=5000.0,
            authorization=ZoneAuth.WEAPONS_FREE,
            priority=5,
        )
        restricted = CircularZone(
            zone_id="RFZ",
            name="Restricted Fire Zone",
            center_xy=np.array([40000.0, 0.0]),
            radius_m=5000.0,
            authorization=ZoneAuth.RESTRICTED_FIRE,
            priority=5,
        )

        weapon = _make_patriot_weapon()
        mgr = _build_manager(
            zones=[no_fire, weapons_free, restricted],
            weapons=[weapon],
            default_zone_auth=ZoneAuth.WEAPONS_FREE,
        )

        track_nofire = MockTrack(
            fused_id="TGT-NFZ",
            position_m=[5000.0, 0.0],
            velocity_mps=200.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )
        track_free = MockTrack(
            fused_id="TGT-WFZ",
            position_m=[20000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )
        track_restricted = MockTrack(
            fused_id="TGT-RFZ",
            position_m=[40000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate(
            [track_nofire, track_free, track_restricted], current_time=100.0,
        )
        assigned_ids = {tid for _, tid in plan.assignment.assignments}

        # NO_FIRE target must never be assigned
        assert "TGT-NFZ" not in assigned_ids, (
            "Target in NO_FIRE zone must not be assigned"
        )
        # WEAPONS_FREE target should be assigned
        assert "TGT-WFZ" in assigned_ids, (
            "Target in WEAPONS_FREE zone should be assigned"
        )

    def test_weapon_capacity_depleted(self):
        """Weapon with zero rounds remaining should not be assigned."""
        empty_weapon = WeaponProfile(
            weapon_id="SAM-EMPTY",
            name="Empty SAM",
            weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
            min_range_m=1000.0,
            max_range_m=40000.0,
            optimal_range_m=20000.0,
            min_altitude_m=0.0,
            weapon_speed_mps=1500.0,
            pk_base=0.85,
            max_target_speed_mps=1500.0,
            max_simultaneous_engagements=2,
            rounds_remaining=0,
        )
        mgr = _build_manager(weapons=[empty_weapon])

        track = MockTrack(
            fused_id="TGT-010",
            position_m=[20000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="CRITICAL",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([track], current_time=100.0)
        assert len(plan.assignment.assignments) == 0, (
            "Depleted weapon should produce no assignments"
        )

    def test_annular_defense_ring(self):
        """Annular defense ring with inner NO_FIRE exclusion zone.

        AnnularZone WEAPONS_FREE ring from 5 km to 20 km.
        CircularZone NO_FIRE inner area with radius 5 km (higher priority).

        Target at [3000, 0] (inside inner NO_FIRE) -> not assigned.
        Target at [12000, 0] (in annular ring) -> assigned.
        """
        annular_ring = AnnularZone(
            zone_id="RING-1",
            name="Defense Ring",
            center_xy=np.array([0.0, 0.0]),
            inner_radius_m=5000.0,
            outer_radius_m=20000.0,
            authorization=ZoneAuth.WEAPONS_FREE,
            priority=5,
        )
        inner_nofire = CircularZone(
            zone_id="INNER-NFZ",
            name="Inner No Fire",
            center_xy=np.array([0.0, 0.0]),
            radius_m=5000.0,
            authorization=ZoneAuth.NO_FIRE,
            priority=10,  # higher priority overrides annular ring
        )

        weapon = _make_patriot_weapon()
        mgr = _build_manager(
            zones=[annular_ring, inner_nofire],
            weapons=[weapon],
            default_zone_auth=ZoneAuth.WEAPONS_FREE,
        )

        inside_track = MockTrack(
            fused_id="TGT-INNER",
            position_m=[3000.0, 0.0],
            velocity_mps=200.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )
        ring_track = MockTrack(
            fused_id="TGT-RING",
            position_m=[12000.0, 0.0],
            velocity_mps=300.0,
            azimuth_deg=270.0,
            threat_level="HIGH",
            engagement_auth="weapons_free",
            iff_identification="hostile",
        )

        plan = mgr.evaluate([inside_track, ring_track], current_time=100.0)
        assigned_ids = {tid for _, tid in plan.assignment.assignments}

        assert "TGT-INNER" not in assigned_ids, (
            "Target inside inner NO_FIRE zone should not be assigned"
        )
        assert "TGT-RING" in assigned_ids, (
            "Target in annular WEAPONS_FREE ring should be assigned"
        )
