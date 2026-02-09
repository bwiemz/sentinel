"""Tests for GeoContext coordinator object."""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.utils.geo_context import GeoContext
from sentinel.utils.geodetic import haversine_distance


class TestGeoContextCreation:
    def test_basic_construction(self):
        gc = GeoContext(lat0_deg=38.8977, lon0_deg=-77.0365, alt0_m=50.0, name="DC")
        assert gc.lat0_deg == 38.8977
        assert gc.lon0_deg == -77.0365
        assert gc.alt0_m == 50.0
        assert gc.name == "DC"

    def test_defaults(self):
        gc = GeoContext(lat0_deg=0.0, lon0_deg=0.0)
        assert gc.alt0_m == 0.0
        assert gc.name == ""

    def test_frozen(self):
        gc = GeoContext(lat0_deg=0.0, lon0_deg=0.0)
        with pytest.raises(AttributeError):
            gc.lat0_deg = 1.0


class TestGeoContextFromConfig:
    def test_enabled_config(self):
        cfg = OmegaConf.create({
            "enabled": True,
            "lat": 38.8977,
            "lon": -77.0365,
            "alt": 50.0,
            "name": "Pentagon",
        })
        gc = GeoContext.from_config(cfg)
        assert gc is not None
        assert gc.lat0_deg == 38.8977
        assert gc.lon0_deg == -77.0365
        assert gc.alt0_m == 50.0
        assert gc.name == "Pentagon"

    def test_disabled_config(self):
        cfg = OmegaConf.create({"enabled": False, "lat": 51.5, "lon": -0.13})
        gc = GeoContext.from_config(cfg)
        assert gc is None

    def test_missing_enabled(self):
        cfg = OmegaConf.create({"lat": 51.5, "lon": -0.13})
        gc = GeoContext.from_config(cfg)
        assert gc is None  # defaults to disabled

    def test_none_config(self):
        gc = GeoContext.from_config(None)
        assert gc is None

    def test_dict_config(self):
        cfg = {"enabled": True, "lat": 35.68, "lon": 139.69, "alt": 40.0}
        gc = GeoContext.from_config(cfg)
        assert gc is not None
        assert gc.lat0_deg == 35.68


class TestGeoContextConversions:
    @pytest.fixture
    def dc_context(self):
        return GeoContext(lat0_deg=38.8977, lon0_deg=-77.0365, alt0_m=0.0)

    def test_geodetic_to_enu_self(self, dc_context):
        enu = dc_context.geodetic_to_enu(38.8977, -77.0365, 0.0)
        assert isinstance(enu, np.ndarray)
        assert enu.shape == (3,)
        assert abs(enu[0]) < 1e-6
        assert abs(enu[1]) < 1e-6
        assert abs(enu[2]) < 1e-6

    def test_target_geodetic_to_xy(self, dc_context):
        xy = dc_context.target_geodetic_to_xy(38.91, -77.02, 0.0)
        assert isinstance(xy, np.ndarray)
        assert xy.shape == (2,)
        # Should be ~1.4 km NE
        dist = np.linalg.norm(xy)
        assert 1000 < dist < 3000

    def test_xy_to_geodetic(self, dc_context):
        lat, lon, alt = dc_context.xy_to_geodetic(0.0, 0.0, 0.0)
        assert lat == pytest.approx(38.8977, abs=1e-6)
        assert lon == pytest.approx(-77.0365, abs=1e-6)

    def test_roundtrip_target(self, dc_context):
        """geodetic → xy → geodetic roundtrip."""
        lat1, lon1 = 38.91, -77.02
        xy = dc_context.target_geodetic_to_xy(lat1, lon1)
        lat_back, lon_back, _ = dc_context.xy_to_geodetic(xy[0], xy[1])
        assert lat_back == pytest.approx(lat1, abs=1e-6)
        assert lon_back == pytest.approx(lon1, abs=1e-6)

    def test_enu_to_geodetic(self, dc_context):
        # 1 km east, 2 km north
        lat, lon, alt = dc_context.enu_to_geodetic(1000.0, 2000.0, 0.0)
        assert lat > 38.8977  # north
        assert lon > -77.0365  # east (less negative)

    def test_distance_m(self, dc_context):
        d = dc_context.distance_m(38.8977, -77.0365, 38.91, -77.02)
        assert 1000 < d < 3000

    def test_bearing_deg(self, dc_context):
        # Point directly north
        b = dc_context.bearing_deg(38.8977, -77.0365, 38.91, -77.0365)
        assert b == pytest.approx(0.0, abs=1.0)  # ~north
