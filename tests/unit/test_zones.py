"""Unit tests for engagement zone types and ZoneManager."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.core.types import ZoneAuth
from sentinel.engagement.zones import (
    AnnularZone,
    CircularZone,
    PolygonZone,
    SectorZone,
    ZoneManager,
)


# ===================================================================
# CircularZone
# ===================================================================


class TestCircularZone:
    def _zone(self, **kwargs):
        defaults = dict(
            zone_id="C1",
            name="Circle-1",
            center_xy=np.array([0.0, 0.0]),
            radius_m=1000.0,
            authorization=ZoneAuth.WEAPONS_FREE,
        )
        defaults.update(kwargs)
        return CircularZone(**defaults)

    def test_point_inside(self):
        z = self._zone()
        assert z.contains(np.array([500.0, 0.0])) is True

    def test_point_at_center(self):
        z = self._zone()
        assert z.contains(np.array([0.0, 0.0])) is True

    def test_point_outside(self):
        z = self._zone()
        assert z.contains(np.array([1500.0, 0.0])) is False

    def test_point_on_boundary(self):
        z = self._zone()
        # Exactly at radius distance
        assert z.contains(np.array([1000.0, 0.0])) is True

    def test_point_just_outside_boundary(self):
        z = self._zone()
        assert z.contains(np.array([1000.1, 0.0])) is False

    def test_diagonal_inside(self):
        z = self._zone()
        # 707.1 < 1000
        assert z.contains(np.array([500.0, 500.0])) is True

    def test_diagonal_outside(self):
        z = self._zone()
        # sqrt(800^2 + 800^2) ~ 1131 > 1000
        assert z.contains(np.array([800.0, 800.0])) is False

    def test_altitude_inside(self):
        z = self._zone(altitude_min_m=100.0, altitude_max_m=5000.0)
        assert z.contains(np.array([0.0, 0.0, 3000.0])) is True

    def test_altitude_below_minimum(self):
        z = self._zone(altitude_min_m=100.0, altitude_max_m=5000.0)
        assert z.contains(np.array([0.0, 0.0, 50.0])) is False

    def test_altitude_above_maximum(self):
        z = self._zone(altitude_min_m=100.0, altitude_max_m=5000.0)
        assert z.contains(np.array([0.0, 0.0, 6000.0])) is False

    def test_2d_position_skips_altitude_check(self):
        z = self._zone(altitude_min_m=5000.0, altitude_max_m=10000.0)
        # 2D position should bypass altitude filter
        assert z.contains(np.array([0.0, 0.0])) is True

    def test_offset_center(self):
        z = self._zone(center_xy=np.array([5000.0, 5000.0]))
        assert z.contains(np.array([5500.0, 5000.0])) is True
        assert z.contains(np.array([0.0, 0.0])) is False

    def test_to_dict(self):
        z = self._zone()
        d = z.to_dict()
        assert d["type"] == "circle"
        assert d["zone_id"] == "C1"
        assert d["radius_m"] == 1000.0
        assert d["authorization"] == "weapons_free"
        assert d["center_xy"] == [0.0, 0.0]

    def test_equality_by_zone_id(self):
        z1 = self._zone(zone_id="X")
        z2 = self._zone(zone_id="X", radius_m=9999.0)
        assert z1 == z2

    def test_inequality_by_zone_id(self):
        z1 = self._zone(zone_id="A")
        z2 = self._zone(zone_id="B")
        assert z1 != z2


# ===================================================================
# PolygonZone
# ===================================================================


class TestPolygonZone:
    def _square_zone(self, **kwargs):
        """Unit square [0,1000] x [0,1000]."""
        defaults = dict(
            zone_id="P1",
            name="Polygon-1",
            vertices=np.array([
                [0.0, 0.0],
                [1000.0, 0.0],
                [1000.0, 1000.0],
                [0.0, 1000.0],
            ]),
            authorization=ZoneAuth.NO_FIRE,
        )
        defaults.update(kwargs)
        return PolygonZone(**defaults)

    def test_point_inside_square(self):
        z = self._square_zone()
        assert z.contains(np.array([500.0, 500.0])) is True

    def test_point_outside_square(self):
        z = self._square_zone()
        assert z.contains(np.array([1500.0, 500.0])) is False

    def test_point_outside_negative(self):
        z = self._square_zone()
        assert z.contains(np.array([-100.0, 500.0])) is False

    def test_triangle(self):
        tri = PolygonZone(
            zone_id="T1",
            name="Triangle",
            vertices=np.array([[0.0, 0.0], [1000.0, 0.0], [500.0, 1000.0]]),
            authorization=ZoneAuth.RESTRICTED_FIRE,
        )
        assert tri.contains(np.array([500.0, 400.0])) is True
        assert tri.contains(np.array([900.0, 900.0])) is False

    def test_altitude_filter(self):
        z = self._square_zone(altitude_min_m=500.0, altitude_max_m=3000.0)
        assert z.contains(np.array([500.0, 500.0, 1000.0])) is True
        assert z.contains(np.array([500.0, 500.0, 100.0])) is False

    def test_to_dict(self):
        z = self._square_zone()
        d = z.to_dict()
        assert d["type"] == "polygon"
        assert d["zone_id"] == "P1"
        assert len(d["vertices"]) == 4
        assert d["authorization"] == "no_fire"


# ===================================================================
# AnnularZone
# ===================================================================


class TestAnnularZone:
    def _zone(self, **kwargs):
        defaults = dict(
            zone_id="A1",
            name="Annular-1",
            center_xy=np.array([0.0, 0.0]),
            inner_radius_m=500.0,
            outer_radius_m=2000.0,
            authorization=ZoneAuth.SELF_DEFENSE_ONLY,
        )
        defaults.update(kwargs)
        return AnnularZone(**defaults)

    def test_inside_annulus(self):
        z = self._zone()
        assert z.contains(np.array([1000.0, 0.0])) is True

    def test_inside_hole(self):
        z = self._zone()
        assert z.contains(np.array([100.0, 0.0])) is False

    def test_center_excluded(self):
        z = self._zone()
        assert z.contains(np.array([0.0, 0.0])) is False

    def test_outside_outer(self):
        z = self._zone()
        assert z.contains(np.array([3000.0, 0.0])) is False

    def test_on_inner_boundary(self):
        z = self._zone()
        # Exactly at inner_radius — should not be excluded (dist == inner is <, so excluded)
        # Code uses `dist < inner_radius_m`, so dist == inner is NOT less, so included
        assert z.contains(np.array([500.0, 0.0])) is True

    def test_on_outer_boundary(self):
        z = self._zone()
        assert z.contains(np.array([2000.0, 0.0])) is True

    def test_altitude_filter(self):
        z = self._zone(altitude_min_m=1000.0, altitude_max_m=8000.0)
        assert z.contains(np.array([1000.0, 0.0, 5000.0])) is True
        assert z.contains(np.array([1000.0, 0.0, 500.0])) is False

    def test_to_dict(self):
        z = self._zone()
        d = z.to_dict()
        assert d["type"] == "annular"
        assert d["inner_radius_m"] == 500.0
        assert d["outer_radius_m"] == 2000.0
        assert d["authorization"] == "self_defense_only"


# ===================================================================
# SectorZone
# ===================================================================


class TestSectorZone:
    def _zone(self, **kwargs):
        defaults = dict(
            zone_id="S1",
            name="Sector-1",
            center_xy=np.array([0.0, 0.0]),
            radius_m=5000.0,
            azimuth_min_deg=0.0,
            azimuth_max_deg=90.0,
            authorization=ZoneAuth.WEAPONS_FREE,
        )
        defaults.update(kwargs)
        return SectorZone(**defaults)

    def test_north_inside(self):
        """Point due north (bearing 0) with azimuth 0-90 is inside."""
        z = self._zone()
        assert z.contains(np.array([0.0, 1000.0])) is True

    def test_east_inside(self):
        """Point due east (bearing 90) with azimuth 0-90 is inside."""
        z = self._zone()
        assert z.contains(np.array([1000.0, 0.0])) is True

    def test_south_outside(self):
        """Point due south (bearing 180) is outside 0-90 sector."""
        z = self._zone()
        assert z.contains(np.array([0.0, -1000.0])) is False

    def test_west_outside(self):
        """Point due west (bearing 270) is outside 0-90 sector."""
        z = self._zone()
        assert z.contains(np.array([-1000.0, 0.0])) is False

    def test_northeast_inside(self):
        """Point at bearing 45 is inside 0-90 sector."""
        z = self._zone()
        assert z.contains(np.array([1000.0, 1000.0])) is True

    def test_beyond_radius(self):
        z = self._zone()
        # Bearing 45 but too far
        assert z.contains(np.array([4000.0, 4000.0])) is False

    def test_wrap_around_north(self):
        """Sector 350-10 wraps around North."""
        z = self._zone(azimuth_min_deg=350.0, azimuth_max_deg=10.0)
        # Due north (bearing 0) should be inside
        assert z.contains(np.array([0.0, 1000.0])) is True
        # Bearing 355 should be inside
        # dx=sin(355)=-0.087, dy=cos(355)=0.996 => point at (-87, 996)
        assert z.contains(np.array([-87.0, 996.0])) is True
        # Bearing 180 should be outside
        assert z.contains(np.array([0.0, -1000.0])) is False

    def test_altitude_filter(self):
        z = self._zone(altitude_min_m=1000.0, altitude_max_m=5000.0)
        assert z.contains(np.array([0.0, 1000.0, 3000.0])) is True
        assert z.contains(np.array([0.0, 1000.0, 500.0])) is False

    def test_to_dict(self):
        z = self._zone()
        d = z.to_dict()
        assert d["type"] == "sector"
        assert d["azimuth_min_deg"] == 0.0
        assert d["azimuth_max_deg"] == 90.0
        assert d["radius_m"] == 5000.0


# ===================================================================
# ZoneManager
# ===================================================================


class TestZoneManager:
    def test_empty_returns_default(self):
        mgr = ZoneManager([], default_authorization=ZoneAuth.WEAPONS_FREE)
        auth = mgr.resolve_authorization(np.array([100.0, 200.0]))
        assert auth == ZoneAuth.WEAPONS_FREE

    def test_single_zone_inside(self):
        zone = CircularZone(
            zone_id="Z1", name="Z1",
            center_xy=np.array([0.0, 0.0]), radius_m=1000.0,
            authorization=ZoneAuth.NO_FIRE,
        )
        mgr = ZoneManager([zone])
        assert mgr.resolve_authorization(np.array([500.0, 0.0])) == ZoneAuth.NO_FIRE

    def test_single_zone_outside_returns_default(self):
        zone = CircularZone(
            zone_id="Z1", name="Z1",
            center_xy=np.array([0.0, 0.0]), radius_m=1000.0,
            authorization=ZoneAuth.NO_FIRE,
        )
        mgr = ZoneManager([zone], default_authorization=ZoneAuth.WEAPONS_FREE)
        assert mgr.resolve_authorization(np.array([5000.0, 0.0])) == ZoneAuth.WEAPONS_FREE

    def test_higher_priority_wins(self):
        low = CircularZone(
            zone_id="LOW", name="Low",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.WEAPONS_FREE, priority=1,
        )
        high = CircularZone(
            zone_id="HIGH", name="High",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.NO_FIRE, priority=10,
        )
        mgr = ZoneManager([low, high])
        assert mgr.resolve_authorization(np.array([100.0, 0.0])) == ZoneAuth.NO_FIRE

    def test_equal_priority_most_restrictive_wins(self):
        z1 = CircularZone(
            zone_id="Z1", name="Z1",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.WEAPONS_FREE, priority=5,
        )
        z2 = CircularZone(
            zone_id="Z2", name="Z2",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.RESTRICTED_FIRE, priority=5,
        )
        mgr = ZoneManager([z1, z2])
        assert mgr.resolve_authorization(np.array([100.0, 0.0])) == ZoneAuth.RESTRICTED_FIRE

    def test_equal_priority_no_fire_is_most_restrictive(self):
        z1 = CircularZone(
            zone_id="Z1", name="Z1",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.SELF_DEFENSE_ONLY, priority=5,
        )
        z2 = CircularZone(
            zone_id="Z2", name="Z2",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.NO_FIRE, priority=5,
        )
        mgr = ZoneManager([z1, z2])
        assert mgr.resolve_authorization(np.array([100.0, 0.0])) == ZoneAuth.NO_FIRE

    def test_only_containing_zones_apply(self):
        """A higher-priority zone that does NOT contain the point is ignored."""
        inner = CircularZone(
            zone_id="INNER", name="Inner",
            center_xy=np.array([0.0, 0.0]), radius_m=100.0,
            authorization=ZoneAuth.NO_FIRE, priority=10,
        )
        outer = CircularZone(
            zone_id="OUTER", name="Outer",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.WEAPONS_FREE, priority=1,
        )
        mgr = ZoneManager([inner, outer])
        # Point is in outer but not inner
        assert mgr.resolve_authorization(np.array([1000.0, 0.0])) == ZoneAuth.WEAPONS_FREE

    def test_add_zone(self):
        mgr = ZoneManager([])
        assert len(mgr.get_all_zones()) == 0
        zone = CircularZone(
            zone_id="NEW", name="New",
            center_xy=np.array([0.0, 0.0]), radius_m=1000.0,
            authorization=ZoneAuth.NO_FIRE,
        )
        mgr.add_zone(zone)
        assert len(mgr.get_all_zones()) == 1

    def test_remove_zone(self):
        zone = CircularZone(
            zone_id="RM", name="Remove-me",
            center_xy=np.array([0.0, 0.0]), radius_m=1000.0,
            authorization=ZoneAuth.NO_FIRE,
        )
        mgr = ZoneManager([zone])
        assert len(mgr.get_all_zones()) == 1
        mgr.remove_zone("RM")
        assert len(mgr.get_all_zones()) == 0

    def test_remove_nonexistent_zone_no_error(self):
        mgr = ZoneManager([])
        mgr.remove_zone("NOPE")  # Should not raise

    def test_get_containing_zones(self):
        z1 = CircularZone(
            zone_id="Z1", name="Z1",
            center_xy=np.array([0.0, 0.0]), radius_m=5000.0,
            authorization=ZoneAuth.WEAPONS_FREE,
        )
        z2 = CircularZone(
            zone_id="Z2", name="Z2",
            center_xy=np.array([10000.0, 0.0]), radius_m=1000.0,
            authorization=ZoneAuth.NO_FIRE,
        )
        mgr = ZoneManager([z1, z2])
        containing = mgr.get_containing_zones(np.array([0.0, 0.0]))
        assert len(containing) == 1
        assert containing[0].zone_id == "Z1"

    def test_from_config_circle(self):
        cfg = [
            {
                "type": "circle",
                "zone_id": "CFG1",
                "name": "Config Circle",
                "center_xy": [100.0, 200.0],
                "radius_m": 3000.0,
                "authorization": "no_fire",
                "priority": 5,
            }
        ]
        mgr = ZoneManager.from_config(cfg)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        assert zones[0].zone_id == "CFG1"
        assert isinstance(zones[0], CircularZone)
        assert zones[0].authorization == ZoneAuth.NO_FIRE
        assert zones[0].priority == 5

    def test_from_config_polygon(self):
        cfg = [
            {
                "type": "polygon",
                "zone_id": "POLY1",
                "vertices": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                "authorization": "restricted_fire",
            }
        ]
        mgr = ZoneManager.from_config(cfg)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        assert isinstance(zones[0], PolygonZone)

    def test_from_config_annular(self):
        cfg = [
            {
                "type": "annular",
                "zone_id": "ANN1",
                "center_xy": [0, 0],
                "inner_radius_m": 500,
                "outer_radius_m": 3000,
                "authorization": "self_defense_only",
            }
        ]
        mgr = ZoneManager.from_config(cfg)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        assert isinstance(zones[0], AnnularZone)

    def test_from_config_sector(self):
        cfg = [
            {
                "type": "sector",
                "zone_id": "SEC1",
                "center_xy": [0, 0],
                "radius_m": 5000,
                "azimuth_min_deg": 45,
                "azimuth_max_deg": 135,
                "authorization": "weapons_free",
            }
        ]
        mgr = ZoneManager.from_config(cfg)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        assert isinstance(zones[0], SectorZone)

    def test_from_config_mixed_types(self):
        cfg = [
            {"type": "circle", "zone_id": "Z1", "radius_m": 1000},
            {"type": "polygon", "zone_id": "Z2"},
            {"type": "annular", "zone_id": "Z3"},
            {"type": "sector", "zone_id": "Z4"},
        ]
        mgr = ZoneManager.from_config(cfg)
        assert len(mgr.get_all_zones()) == 4

    def test_from_config_unknown_type_skipped(self):
        cfg = [{"type": "hexagonal", "zone_id": "BAD"}]
        mgr = ZoneManager.from_config(cfg)
        assert len(mgr.get_all_zones()) == 0

    def test_from_config_empty_list(self):
        mgr = ZoneManager.from_config([])
        assert len(mgr.get_all_zones()) == 0

    def test_from_config_default_authorization(self):
        mgr = ZoneManager.from_config(
            [], default_authorization=ZoneAuth.NO_FIRE
        )
        assert mgr.resolve_authorization(np.array([0.0, 0.0])) == ZoneAuth.NO_FIRE

    def test_from_config_geodetic_center(self):
        """Geodetic center_geo should be converted via geo_context."""
        geo = MagicMock()
        geo.geodetic_to_enu.return_value = (1000.0, 2000.0, 0.0)
        cfg = [
            {
                "type": "circle",
                "zone_id": "GEO1",
                "center_geo": [38.0, -77.0],
                "radius_m": 5000.0,
                "authorization": "weapons_free",
            }
        ]
        mgr = ZoneManager.from_config(cfg, geo_context=geo)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        np.testing.assert_allclose(zones[0].center_xy, [1000.0, 2000.0])
        geo.geodetic_to_enu.assert_called_once_with(38.0, -77.0, 0.0)

    def test_from_config_geodetic_vertices(self):
        """Geodetic vertices_geo should be converted via geo_context."""
        geo = MagicMock()
        geo.geodetic_to_enu.side_effect = [
            (0.0, 0.0, 0.0),
            (1000.0, 0.0, 0.0),
            (1000.0, 1000.0, 0.0),
            (0.0, 1000.0, 0.0),
        ]
        cfg = [
            {
                "type": "polygon",
                "zone_id": "GPOLY",
                "vertices_geo": [
                    [38.0, -77.0],
                    [38.0, -76.9],
                    [38.1, -76.9],
                    [38.1, -77.0],
                ],
                "authorization": "no_fire",
            }
        ]
        mgr = ZoneManager.from_config(cfg, geo_context=geo)
        zones = mgr.get_all_zones()
        assert len(zones) == 1
        assert zones[0].vertices.shape == (4, 2)
        assert geo.geodetic_to_enu.call_count == 4
