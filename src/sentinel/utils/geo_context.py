"""Geodetic reference context for ENU coordinate conversions.

GeoContext holds the WGS84 reference point (typically the sensor/radar site
location). All ENU coordinates in SENTINEL are relative to this point.

When GeoContext is None throughout the system, SENTINEL operates in raw
Cartesian mode exactly as before (full backward compatibility).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.utils.geodetic import (
    enu_to_geodetic,
    geodetic_bearing,
    geodetic_to_enu,
    haversine_distance,
)


@dataclass(frozen=True)
class GeoContext:
    """Immutable geodetic reference context.

    Attributes:
        lat0_deg: Reference latitude in degrees.
        lon0_deg: Reference longitude in degrees.
        alt0_m: Reference altitude in meters above WGS84 ellipsoid.
        name: Optional human-readable site name.
    """

    lat0_deg: float
    lon0_deg: float
    alt0_m: float = 0.0
    name: str = ""

    def geodetic_to_enu(
        self, lat_deg: float, lon_deg: float, alt_m: float = 0.0,
    ) -> np.ndarray:
        """Convert geodetic to ENU meters. Returns array [east, north, up]."""
        e, n, u = geodetic_to_enu(
            lat_deg, lon_deg, alt_m,
            self.lat0_deg, self.lon0_deg, self.alt0_m,
        )
        return np.array([e, n, u])

    def enu_to_geodetic(
        self, e_m: float, n_m: float, u_m: float = 0.0,
    ) -> tuple[float, float, float]:
        """Convert ENU meters to geodetic (lat_deg, lon_deg, alt_m)."""
        return enu_to_geodetic(
            e_m, n_m, u_m,
            self.lat0_deg, self.lon0_deg, self.alt0_m,
        )

    def target_geodetic_to_xy(
        self, lat_deg: float, lon_deg: float, alt_m: float = 0.0,
    ) -> np.ndarray:
        """Convert geodetic target position to 2D ENU [x, y] meters.

        Drops the Up component. x=East, y=North (matching existing convention
        where azimuth=0 is along +x).
        """
        enu = self.geodetic_to_enu(lat_deg, lon_deg, alt_m)
        return enu[:2]

    def xy_to_geodetic(
        self, x_m: float, y_m: float, alt_m: float = 0.0,
    ) -> tuple[float, float, float]:
        """Convert 2D ENU [x, y] meters to geodetic. x=East, y=North."""
        return self.enu_to_geodetic(x_m, y_m, alt_m)

    def distance_m(
        self, lat1: float, lon1: float, lat2: float, lon2: float,
    ) -> float:
        """Great-circle distance between two geodetic points in meters."""
        return haversine_distance(lat1, lon1, lat2, lon2)

    def bearing_deg(
        self, lat1: float, lon1: float, lat2: float, lon2: float,
    ) -> float:
        """Geodetic bearing from point 1 to point 2 in degrees (0=North, CW)."""
        return geodetic_bearing(lat1, lon1, lat2, lon2)

    @classmethod
    def from_config(cls, cfg) -> GeoContext | None:
        """Create from config dict/DictConfig. Returns None if not configured.

        Expects a dict with keys: enabled, lat, lon, alt, name.
        Returns None if 'enabled' is False or missing.
        """
        if cfg is None:
            return None
        geo = cfg if not hasattr(cfg, "get") else cfg
        enabled = getattr(geo, "enabled", None)
        if enabled is None:
            enabled = geo.get("enabled", False) if hasattr(geo, "get") else False
        if not enabled:
            return None
        lat = float(getattr(geo, "lat", 0.0) if not hasattr(geo, "get") else geo.get("lat", 0.0))
        lon = float(getattr(geo, "lon", 0.0) if not hasattr(geo, "get") else geo.get("lon", 0.0))
        alt = float(getattr(geo, "alt", 0.0) if not hasattr(geo, "get") else geo.get("alt", 0.0))
        name = str(getattr(geo, "name", "") if not hasattr(geo, "get") else geo.get("name", ""))
        return cls(lat0_deg=lat, lon0_deg=lon, alt0_m=alt, name=name)
