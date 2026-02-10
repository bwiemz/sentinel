"""Unit tests for EngagementManager orchestration."""

import numpy as np
import pytest

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.config import EngagementConfig
from sentinel.engagement.manager import EngagementManager, EngagementPlan
from sentinel.engagement.weapons import WeaponProfile
from sentinel.engagement.zones import CircularZone, ZoneManager


# -------------------------------------------------------------------
# Helper mock track
# -------------------------------------------------------------------


class MockTrack:
    """Lightweight stand-in for EnhancedFusedTrack."""

    def __init__(
        self,
        fused_id="T-1",
        position_m=(5000.0, 5000.0),
        velocity_mps=300.0,
        azimuth_deg=180.0,
        threat_level="HIGH",
        engagement_auth=EngagementAuth.WEAPONS_FREE,
        iff_identification="hostile",
    ):
        self.fused_id = fused_id
        self.position_m = position_m
        self.velocity_mps = velocity_mps
        self.azimuth_deg = azimuth_deg
        self.threat_level = threat_level
        self.engagement_auth = engagement_auth
        self.iff_identification = iff_identification
        self.radar_track = None
        self.is_jammed = False


# -------------------------------------------------------------------
# Helpers for building configs
# -------------------------------------------------------------------


def _make_weapon_def(
    weapon_id="SAM-1",
    weapon_type="sam_medium",
    position_xy=None,
    max_range_m=30000.0,
):
    """Return a raw weapon dict consumable by WeaponProfile.from_config."""
    return {
        "weapon_id": weapon_id,
        "name": weapon_id,
        "weapon_type": weapon_type,
        "position_xy": position_xy or [0.0, 0.0],
        "max_range_m": max_range_m,
        "min_range_m": 200.0,
        "optimal_range_m": 10000.0,
        "min_altitude_m": 0.0,
        "max_altitude_m": 50000.0,
        "max_target_speed_mps": 1500.0,
        "weapon_speed_mps": 1200.0,
        "pk_base": 0.90,
        "rounds_remaining": 10,
        "max_simultaneous_engagements": 2,
    }


def _make_zone_def(
    zone_id="Z-NFZ",
    zone_type="circle",
    center_xy=None,
    radius_m=5000.0,
    authorization="no_fire",
    priority=10,
):
    return {
        "zone_id": zone_id,
        "type": zone_type,
        "name": zone_id,
        "center_xy": center_xy or [20000.0, 20000.0],
        "radius_m": radius_m,
        "authorization": authorization,
        "priority": priority,
    }


def _enabled_config(
    weapon_defs=None,
    zone_defs=None,
    default_zone_auth=ZoneAuth.WEAPONS_FREE,
):
    """Return an EngagementConfig with enabled=True."""
    return EngagementConfig(
        enabled=True,
        pk_weight=0.4,
        tti_weight=0.3,
        threat_weight=0.3,
        max_tti_s=120.0,
        default_zone_auth=default_zone_auth,
        zone_defs=zone_defs if zone_defs is not None else [],
        weapon_defs=weapon_defs if weapon_defs is not None else [_make_weapon_def()],
    )


# ===================================================================
# TestEngagementManager
# ===================================================================


class TestEngagementManager:
    # ---------------------------------------------------------------
    # from_config factory
    # ---------------------------------------------------------------

    def test_from_config_disabled(self):
        """from_config returns None when engagement is disabled."""
        result = EngagementManager.from_config({"enabled": False})
        assert result is None

    def test_from_config_enabled(self):
        """from_config returns an EngagementManager with zones and weapons."""
        cfg = {
            "enabled": True,
            "default_zone_auth": "weapons_free",
            "zones": [_make_zone_def()],
            "weapons": [_make_weapon_def()],
        }
        mgr = EngagementManager.from_config(cfg)
        assert isinstance(mgr, EngagementManager)
        # One weapon loaded
        assert len(mgr.weapons) == 1
        assert mgr.weapons[0].weapon_id == "SAM-1"
        # One zone loaded
        assert len(mgr.zone_manager.get_all_zones()) == 1

    # ---------------------------------------------------------------
    # evaluate — basic cases
    # ---------------------------------------------------------------

    def test_evaluate_empty_tracks(self):
        """No tracks produces an empty plan with no assignments."""
        mgr = EngagementManager(config=_enabled_config())
        plan = mgr.evaluate([], current_time=100.0)
        assert isinstance(plan, EngagementPlan)
        assert plan.timestamp == 100.0
        assert len(plan.assignment.assignments) == 0
        assert len(plan.zone_statuses) == 0

    def test_evaluate_single_engagement(self):
        """One weapon, one hostile track within range produces an assignment."""
        mgr = EngagementManager(config=_enabled_config())
        track = MockTrack(
            fused_id="T-1",
            position_m=(5000.0, 5000.0),
            velocity_mps=300.0,
            azimuth_deg=180.0,
            threat_level="HIGH",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            iff_identification="hostile",
        )
        plan = mgr.evaluate([track], current_time=50.0)
        assert plan.timestamp == 50.0
        # Should have at least one assignment
        assigned_track_ids = [tid for _, tid in plan.assignment.assignments]
        assert "T-1" in assigned_track_ids
        # Zone statuses populated
        assert len(plan.zone_statuses) == 1
        assert plan.zone_statuses[0]["track_id"] == "T-1"

    # ---------------------------------------------------------------
    # Zone constraints
    # ---------------------------------------------------------------

    def test_zone_blocks_engagement(self):
        """Track inside a NO_FIRE zone is not assigned to any weapon."""
        # Place the no-fire zone around the track position
        nfz = _make_zone_def(
            zone_id="NFZ-1",
            center_xy=[5000.0, 5000.0],
            radius_m=10000.0,
            authorization="no_fire",
            priority=100,
        )
        config = _enabled_config(zone_defs=[nfz])
        mgr = EngagementManager(config=config)

        track = MockTrack(
            fused_id="T-NFZ",
            position_m=(5000.0, 5000.0),
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            iff_identification="hostile",
        )
        plan = mgr.evaluate([track], current_time=1.0)
        assigned_track_ids = [tid for _, tid in plan.assignment.assignments]
        assert "T-NFZ" not in assigned_track_ids

    # ---------------------------------------------------------------
    # IFF / friendly exclusion
    # ---------------------------------------------------------------

    def test_friendly_exclusion(self):
        """Track with iff_identification='friendly' is never assigned."""
        mgr = EngagementManager(config=_enabled_config())
        track = MockTrack(
            fused_id="T-FRIEND",
            position_m=(5000.0, 5000.0),
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            iff_identification="friendly",
        )
        plan = mgr.evaluate([track], current_time=1.0)
        assigned_track_ids = [tid for _, tid in plan.assignment.assignments]
        assert "T-FRIEND" not in assigned_track_ids

    # ---------------------------------------------------------------
    # Weapon management
    # ---------------------------------------------------------------

    def test_add_remove_weapon(self):
        """add_weapon and remove_weapon dynamically modify the weapon list."""
        mgr = EngagementManager(config=_enabled_config(weapon_defs=[]))
        assert len(mgr.weapons) == 0

        wp = WeaponProfile(
            weapon_id="GUN-1",
            name="GUN-1",
            weapon_type=WeaponType.GUN,
            position_xy=np.array([0.0, 0.0]),
        )
        mgr.add_weapon(wp)
        assert len(mgr.weapons) == 1
        assert mgr.weapons[0].weapon_id == "GUN-1"

        mgr.remove_weapon("GUN-1")
        assert len(mgr.weapons) == 0

    # ---------------------------------------------------------------
    # Zone authorization query
    # ---------------------------------------------------------------

    def test_track_zone_auth(self):
        """get_track_zone_auth returns correct auth for a given position."""
        nfz = CircularZone(
            zone_id="NFZ-A",
            name="No-Fire Alpha",
            center_xy=np.array([1000.0, 1000.0]),
            radius_m=500.0,
            authorization=ZoneAuth.NO_FIRE,
            priority=10,
        )
        zm = ZoneManager(zones=[nfz], default_authorization=ZoneAuth.WEAPONS_FREE)
        mgr = EngagementManager(config=_enabled_config(), zone_manager=zm)

        # Inside the no-fire zone
        auth_in = mgr.get_track_zone_auth(np.array([1000.0, 1000.0]))
        assert auth_in == ZoneAuth.NO_FIRE

        # Outside all zones — default
        auth_out = mgr.get_track_zone_auth(np.array([99999.0, 99999.0]))
        assert auth_out == ZoneAuth.WEAPONS_FREE

    # ---------------------------------------------------------------
    # Multi-weapon, multi-target
    # ---------------------------------------------------------------

    def test_evaluate_multi_weapon(self):
        """Multiple weapons and targets produce optimal assignments."""
        weapon_defs = [
            _make_weapon_def(weapon_id="SAM-A", position_xy=[0.0, 0.0]),
            _make_weapon_def(weapon_id="SAM-B", position_xy=[10000.0, 0.0]),
        ]
        config = _enabled_config(weapon_defs=weapon_defs)
        mgr = EngagementManager(config=config)

        tracks = [
            MockTrack(
                fused_id="T-X",
                position_m=(3000.0, 3000.0),
                threat_level="HIGH",
                engagement_auth=EngagementAuth.WEAPONS_FREE,
            ),
            MockTrack(
                fused_id="T-Y",
                position_m=(12000.0, 3000.0),
                threat_level="CRITICAL",
                engagement_auth=EngagementAuth.WEAPONS_FREE,
            ),
        ]
        plan = mgr.evaluate(tracks, current_time=10.0)
        assigned_track_ids = {tid for _, tid in plan.assignment.assignments}
        # Both targets should be assigned (enough weapons)
        assert "T-X" in assigned_track_ids
        assert "T-Y" in assigned_track_ids

    # ---------------------------------------------------------------
    # EngagementPlan serialization
    # ---------------------------------------------------------------

    def test_plan_to_dict(self):
        """EngagementPlan.to_dict includes timestamp, assignment, zone_statuses."""
        mgr = EngagementManager(config=_enabled_config())
        track = MockTrack(
            fused_id="T-SER",
            position_m=(5000.0, 5000.0),
            engagement_auth=EngagementAuth.WEAPONS_FREE,
        )
        plan = mgr.evaluate([track], current_time=77.0)
        d = plan.to_dict()
        assert d["timestamp"] == 77.0
        assert "assignment" in d
        assert "zone_statuses" in d
        assert isinstance(d["assignment"], dict)
        assert isinstance(d["zone_statuses"], list)

    # ---------------------------------------------------------------
    # _extract_track_dict internals
    # ---------------------------------------------------------------

    def test_extract_track_dict_returns_none_without_position(self):
        """Tracks lacking any position attribute produce None."""
        track = MockTrack()
        track.position_m = None
        result = EngagementManager._extract_track_dict(track)
        assert result is None

    def test_extract_track_dict_uses_position_m(self):
        """_extract_track_dict reads position_m when available."""
        track = MockTrack(position_m=(1000.0, 2000.0))
        td = EngagementManager._extract_track_dict(track)
        assert td is not None
        np.testing.assert_allclose(td["position"], [1000.0, 2000.0])
        assert td["track_id"] == "T-1"
        assert td["engagement_auth"] == EngagementAuth.WEAPONS_FREE

    def test_extract_track_dict_velocity_decomposition(self):
        """Velocity is decomposed from scalar + azimuth."""
        track = MockTrack(velocity_mps=100.0, azimuth_deg=90.0)
        td = EngagementManager._extract_track_dict(track)
        # azimuth 90 deg -> sin(90)=1, cos(90)=~0
        np.testing.assert_allclose(td["velocity"][0], 100.0, atol=1e-6)
        np.testing.assert_allclose(td["velocity"][1], 0.0, atol=1e-6)

    def test_extract_track_dict_engagement_auth_string(self):
        """String engagement_auth is converted to EngagementAuth enum."""
        track = MockTrack()
        track.engagement_auth = "weapons_tight"
        td = EngagementManager._extract_track_dict(track)
        assert td["engagement_auth"] == EngagementAuth.WEAPONS_TIGHT

    def test_evaluate_hold_fire_not_assigned(self):
        """Track with HOLD_FIRE engagement auth is not assigned."""
        mgr = EngagementManager(config=_enabled_config())
        track = MockTrack(
            fused_id="T-HOLD",
            position_m=(5000.0, 5000.0),
            engagement_auth=EngagementAuth.HOLD_FIRE,
            iff_identification="hostile",
        )
        plan = mgr.evaluate([track], current_time=1.0)
        assigned_track_ids = [tid for _, tid in plan.assignment.assignments]
        assert "T-HOLD" not in assigned_track_ids
