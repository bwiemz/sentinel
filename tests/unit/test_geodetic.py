"""Tests for WGS84 geodetic coordinate conversions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.utils.geodetic import (
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_bearing,
    geodetic_to_ecef,
    geodetic_to_enu,
    haversine_distance,
)


# ── Geodetic ↔ ECEF ────────────────────────────────────────────────


class TestGeodeticToECEF:
    """Test geodetic_to_ecef against known values."""

    def test_equator_prime_meridian(self):
        """(0, 0, 0) → on equator at prime meridian → X=A, Y=0, Z=0."""
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert abs(x - WGS84_A) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_north_pole(self):
        """(90, 0, 0) → X=0, Y=0, Z=B."""
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z - WGS84_B) < 0.01

    def test_south_pole(self):
        """(-90, 0, 0) → X=0, Y=0, Z=-B."""
        x, y, z = geodetic_to_ecef(-90.0, 0.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z + WGS84_B) < 0.01

    def test_equator_90_east(self):
        """(0, 90, 0) → X=0, Y=A, Z=0."""
        x, y, z = geodetic_to_ecef(0.0, 90.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y - WGS84_A) < 0.01
        assert abs(z) < 0.01

    def test_equator_90_west(self):
        """(0, -90, 0) → X=0, Y=-A, Z=0."""
        x, y, z = geodetic_to_ecef(0.0, -90.0, 0.0)
        assert abs(x) < 0.01
        assert abs(y + WGS84_A) < 0.01
        assert abs(z) < 0.01

    def test_altitude_increases_radius(self):
        """Point at altitude should be farther from center."""
        x0, y0, z0 = geodetic_to_ecef(0.0, 0.0, 0.0)
        x1, y1, z1 = geodetic_to_ecef(0.0, 0.0, 1000.0)
        r0 = math.sqrt(x0**2 + y0**2 + z0**2)
        r1 = math.sqrt(x1**2 + y1**2 + z1**2)
        assert r1 - r0 == pytest.approx(1000.0, abs=0.1)

    def test_london(self):
        """London (51.5074°N, 0.1278°W) approximate ECEF check."""
        x, y, z = geodetic_to_ecef(51.5074, -0.1278, 11.0)
        r = math.sqrt(x**2 + y**2 + z**2)
        # Radius should be close to ~6370-6378 km
        assert 6_350_000 < r < 6_390_000
        # X should be positive (roughly east of anti-meridian)
        assert x > 0

    def test_equator_180(self):
        """(0, 180, 0) → X=-A, Y≈0, Z=0."""
        x, y, z = geodetic_to_ecef(0.0, 180.0, 0.0)
        assert abs(x + WGS84_A) < 0.01
        assert abs(y) < 0.5  # numerical noise from sin(pi)
        assert abs(z) < 0.01


class TestECEFToGeodetic:
    """Test ecef_to_geodetic against known values."""

    def test_equator_prime_meridian(self):
        lat, lon, alt = ecef_to_geodetic(WGS84_A, 0.0, 0.0)
        assert lat == pytest.approx(0.0, abs=1e-8)
        assert lon == pytest.approx(0.0, abs=1e-8)
        assert alt == pytest.approx(0.0, abs=0.01)

    def test_north_pole(self):
        lat, lon, alt = ecef_to_geodetic(0.0, 0.0, WGS84_B)
        assert lat == pytest.approx(90.0, abs=1e-8)
        assert alt == pytest.approx(0.0, abs=0.01)

    def test_south_pole(self):
        lat, lon, alt = ecef_to_geodetic(0.0, 0.0, -WGS84_B)
        assert lat == pytest.approx(-90.0, abs=1e-8)
        assert alt == pytest.approx(0.0, abs=0.01)

    def test_roundtrip_equator(self):
        """geodetic → ECEF → geodetic roundtrip at equator."""
        lat0, lon0, alt0 = 0.0, 45.0, 500.0
        x, y, z = geodetic_to_ecef(lat0, lon0, alt0)
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(lat0, abs=1e-8)
        assert lon == pytest.approx(lon0, abs=1e-8)
        assert alt == pytest.approx(alt0, abs=0.01)

    def test_roundtrip_high_latitude(self):
        """Roundtrip at high latitude."""
        lat0, lon0, alt0 = 72.0, -40.0, 10000.0
        x, y, z = geodetic_to_ecef(lat0, lon0, alt0)
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(lat0, abs=1e-8)
        assert lon == pytest.approx(lon0, abs=1e-8)
        assert alt == pytest.approx(alt0, abs=0.01)

    @pytest.mark.parametrize("lat0,lon0,alt0", [
        (0.0, 0.0, 0.0),
        (51.5, -0.13, 11.0),
        (35.68, 139.69, 40.0),
        (-33.87, 151.21, 58.0),
        (90.0, 0.0, 0.0),
        (-90.0, 0.0, 0.0),
        (0.0, 180.0, 0.0),
        (0.0, -180.0, 0.0),
        (45.0, 90.0, 35000.0),
    ])
    def test_roundtrip_various(self, lat0, lon0, alt0):
        """Roundtrip at various global positions."""
        x, y, z = geodetic_to_ecef(lat0, lon0, alt0)
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(lat0, abs=1e-6)
        # Longitude wraps at ±180
        if abs(abs(lon0) - 180) < 0.01:
            assert abs(abs(lon) - 180) < 1e-6
        else:
            assert lon == pytest.approx(lon0, abs=1e-6)
        assert alt == pytest.approx(alt0, abs=0.1)


# ── Geodetic ↔ ENU ────────────────────────────────────────────────


class TestGeodeticToENU:
    """Test geodetic_to_enu conversions."""

    def test_same_point_is_zero(self):
        """Converting reference point to ENU gives (0, 0, 0)."""
        e, n, u = geodetic_to_enu(51.5, -0.13, 11.0, 51.5, -0.13, 11.0)
        assert abs(e) < 1e-6
        assert abs(n) < 1e-6
        assert abs(u) < 1e-6

    def test_point_north(self):
        """Point directly north of reference should have +N, ~0 E."""
        # ~111 km per degree at equator
        e, n, u = geodetic_to_enu(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert abs(e) < 10  # nearly zero east
        assert n > 110_000  # ~111 km north
        assert n < 112_000

    def test_point_east(self):
        """Point directly east of reference should have +E, ~0 N."""
        e, n, u = geodetic_to_enu(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert e > 110_000  # ~111 km east at equator
        assert e < 112_000
        assert abs(n) < 10

    def test_point_above(self):
        """Point directly above reference should have +U only."""
        e, n, u = geodetic_to_enu(0.0, 0.0, 1000.0, 0.0, 0.0, 0.0)
        assert abs(e) < 0.1
        assert abs(n) < 0.1
        assert u == pytest.approx(1000.0, abs=0.1)

    def test_short_range_accuracy(self):
        """For nearby points (<10 km), verify sub-meter accuracy."""
        lat0, lon0 = 38.8977, -77.0365  # Washington DC
        # Point ~1 km north
        lat1 = lat0 + 1.0 / 111.0  # approx 1 km per 0.009 deg
        e, n, u = geodetic_to_enu(lat1, lon0, 0.0, lat0, lon0, 0.0)
        assert abs(e) < 1.0  # should be ~0
        assert abs(n - 1000.0) < 20  # ~1 km north, allow some curvature error

    def test_roundtrip(self):
        """ENU → geodetic → ENU roundtrip."""
        lat0, lon0, alt0 = 38.8977, -77.0365, 0.0
        lat1, lon1, alt1 = 38.91, -77.02, 100.0
        e, n, u = geodetic_to_enu(lat1, lon1, alt1, lat0, lon0, alt0)
        lat_back, lon_back, alt_back = enu_to_geodetic(e, n, u, lat0, lon0, alt0)
        assert lat_back == pytest.approx(lat1, abs=1e-7)
        assert lon_back == pytest.approx(lon1, abs=1e-7)
        assert alt_back == pytest.approx(alt1, abs=0.1)


class TestENUToGeodetic:
    """Test enu_to_geodetic conversions."""

    def test_origin_returns_reference(self):
        """(0, 0, 0) ENU → reference geodetic point."""
        lat, lon, alt = enu_to_geodetic(0.0, 0.0, 0.0, 38.8977, -77.0365, 50.0)
        assert lat == pytest.approx(38.8977, abs=1e-7)
        assert lon == pytest.approx(-77.0365, abs=1e-7)
        assert alt == pytest.approx(50.0, abs=0.1)

    def test_east_1km(self):
        """1 km east from reference at equator."""
        lat, lon, alt = enu_to_geodetic(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert lat == pytest.approx(0.0, abs=0.001)
        assert lon > 0.0  # east means positive longitude
        dist = haversine_distance(0.0, 0.0, lat, lon)
        # Haversine uses mean sphere, ENU uses ellipsoid — allow ~10m tolerance
        assert dist == pytest.approx(1000.0, abs=10.0)

    def test_north_1km(self):
        """1 km north from reference at equator."""
        lat, lon, alt = enu_to_geodetic(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0)
        assert lat > 0.0  # north means positive latitude
        assert lon == pytest.approx(0.0, abs=0.001)
        dist = haversine_distance(0.0, 0.0, lat, lon)
        # Haversine uses mean sphere, ENU uses ellipsoid — allow ~10m tolerance
        assert dist == pytest.approx(1000.0, abs=10.0)

    @pytest.mark.parametrize("e,n,u", [
        (0.0, 0.0, 0.0),
        (1000.0, 0.0, 0.0),
        (0.0, 1000.0, 0.0),
        (0.0, 0.0, 500.0),
        (5000.0, 3000.0, 100.0),
        (-2000.0, -4000.0, 0.0),
        (50000.0, 50000.0, 0.0),  # 50 km
        (100000.0, 100000.0, 10000.0),  # 100 km + 10 km alt
    ])
    def test_roundtrip_various(self, e, n, u):
        """ENU → geodetic → ENU roundtrip at various offsets."""
        lat0, lon0, alt0 = 40.0, -74.0, 0.0  # NYC area
        lat, lon, alt = enu_to_geodetic(e, n, u, lat0, lon0, alt0)
        e2, n2, u2 = geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
        assert e2 == pytest.approx(e, abs=0.001)  # sub-mm
        assert n2 == pytest.approx(n, abs=0.001)
        assert u2 == pytest.approx(u, abs=0.001)


# ── Haversine distance ────────────────────────────────────────────


class TestHaversineDistance:
    """Test haversine_distance against known values."""

    def test_same_point_is_zero(self):
        d = haversine_distance(51.5, -0.13, 51.5, -0.13)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_one_degree_latitude_at_equator(self):
        """1° latitude at equator ≈ 111.19 km."""
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111_195, rel=0.001)  # within 0.1%

    def test_one_degree_longitude_at_equator(self):
        """1° longitude at equator ≈ 111.19 km."""
        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111_195, rel=0.001)

    def test_london_to_paris(self):
        """London to Paris ≈ 344 km."""
        d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        assert d == pytest.approx(344_000, rel=0.01)  # within 1%

    def test_new_york_to_london(self):
        """NYC to London ≈ 5,570 km."""
        d = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert d == pytest.approx(5_570_000, rel=0.01)

    def test_antipodal_points(self):
        """Maximum distance: half Earth circumference ≈ 20,004 km."""
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert d == pytest.approx(20_004_000, rel=0.001)

    def test_symmetric(self):
        """Distance is symmetric: d(A,B) = d(B,A)."""
        d1 = haversine_distance(51.5, -0.13, 48.86, 2.35)
        d2 = haversine_distance(48.86, 2.35, 51.5, -0.13)
        assert d1 == pytest.approx(d2, abs=0.01)


# ── Geodetic bearing ──────────────────────────────────────────────


class TestGeodeticBearing:
    """Test geodetic_bearing against known values."""

    def test_due_north(self):
        """Point directly north → bearing 0°."""
        b = geodetic_bearing(0.0, 0.0, 1.0, 0.0)
        assert b == pytest.approx(0.0, abs=0.01)

    def test_due_east(self):
        """Point directly east at equator → bearing 90°."""
        b = geodetic_bearing(0.0, 0.0, 0.0, 1.0)
        assert b == pytest.approx(90.0, abs=0.01)

    def test_due_south(self):
        """Point directly south → bearing 180°."""
        b = geodetic_bearing(1.0, 0.0, 0.0, 0.0)
        assert b == pytest.approx(180.0, abs=0.01)

    def test_due_west(self):
        """Point directly west at equator → bearing 270°."""
        b = geodetic_bearing(0.0, 1.0, 0.0, 0.0)
        assert b == pytest.approx(270.0, abs=0.01)

    def test_northeast(self):
        """Point northeast → bearing ~45°."""
        b = geodetic_bearing(0.0, 0.0, 1.0, 1.0)
        assert 40 < b < 50  # approximately 45°

    def test_bearing_range(self):
        """Bearing is always in [0, 360)."""
        b = geodetic_bearing(0.0, 0.0, -1.0, -1.0)
        assert 0 <= b < 360

    def test_london_to_paris(self):
        """London to Paris ≈ ~150° (roughly SSE)."""
        b = geodetic_bearing(51.5074, -0.1278, 48.8566, 2.3522)
        assert 140 < b < 160
