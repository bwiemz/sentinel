"""Geographic engagement zones — circle, polygon, annular, sector.

Defines spatial regions with engagement authorization levels (NO_FIRE,
RESTRICTED_FIRE, SELF_DEFENSE_ONLY, WEAPONS_FREE). ZoneManager resolves
overlapping zones by priority (highest wins, most restrictive breaks ties).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from sentinel.core.types import ZoneAuth

logger = logging.getLogger(__name__)

# Zone authorization restrictiveness order (lower index = more restrictive)
_ZONE_AUTH_RESTRICTIVENESS = [
    ZoneAuth.NO_FIRE,
    ZoneAuth.RESTRICTED_FIRE,
    ZoneAuth.SELF_DEFENSE_ONLY,
    ZoneAuth.WEAPONS_FREE,
]


def _point_in_polygon(point_xy: np.ndarray, vertices: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (2D).

    Casts a ray from the test point along +X and counts edge crossings.
    Odd count = inside, even = outside.
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i, 0]), float(vertices[i, 1])
        xj, yj = float(vertices[j, 0]), float(vertices[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _check_altitude(position: np.ndarray, alt_min: float, alt_max: float) -> bool:
    """Check altitude bounds if position has a z component."""
    if len(position) >= 3:
        alt = float(position[2])
        return alt_min <= alt <= alt_max
    return True  # 2D position — no altitude to check


def _normalize_angle(deg: float) -> float:
    """Normalize angle to [0, 360)."""
    return deg % 360.0


@dataclass(frozen=True)
class CircularZone:
    """Circular engagement zone defined by center and radius in ENU meters."""

    zone_id: str
    name: str
    center_xy: np.ndarray
    radius_m: float
    authorization: ZoneAuth
    altitude_min_m: float = 0.0
    altitude_max_m: float = 30000.0
    priority: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircularZone):
            return NotImplemented
        return self.zone_id == other.zone_id

    def __hash__(self) -> int:
        return hash(self.zone_id)

    def contains(self, position: np.ndarray) -> bool:
        """Test if position is within this zone."""
        dx = float(position[0]) - float(self.center_xy[0])
        dy = float(position[1]) - float(self.center_xy[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > self.radius_m:
            return False
        return _check_altitude(position, self.altitude_min_m, self.altitude_max_m)

    def to_dict(self) -> dict:
        return {
            "type": "circle",
            "zone_id": self.zone_id,
            "name": self.name,
            "center_xy": self.center_xy.tolist(),
            "radius_m": self.radius_m,
            "authorization": self.authorization.value,
            "altitude_min_m": self.altitude_min_m,
            "altitude_max_m": self.altitude_max_m,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class PolygonZone:
    """Polygonal engagement zone defined by vertices in ENU meters."""

    zone_id: str
    name: str
    vertices: np.ndarray  # Nx2
    authorization: ZoneAuth
    altitude_min_m: float = 0.0
    altitude_max_m: float = 30000.0
    priority: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolygonZone):
            return NotImplemented
        return self.zone_id == other.zone_id

    def __hash__(self) -> int:
        return hash(self.zone_id)

    def contains(self, position: np.ndarray) -> bool:
        """Test if position is within this polygon (ray-casting)."""
        if not _point_in_polygon(position, self.vertices):
            return False
        return _check_altitude(position, self.altitude_min_m, self.altitude_max_m)

    def to_dict(self) -> dict:
        return {
            "type": "polygon",
            "zone_id": self.zone_id,
            "name": self.name,
            "vertices": self.vertices.tolist(),
            "authorization": self.authorization.value,
            "altitude_min_m": self.altitude_min_m,
            "altitude_max_m": self.altitude_max_m,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class AnnularZone:
    """Ring/annulus engagement zone defined by inner and outer radius."""

    zone_id: str
    name: str
    center_xy: np.ndarray
    inner_radius_m: float
    outer_radius_m: float
    authorization: ZoneAuth
    altitude_min_m: float = 0.0
    altitude_max_m: float = 30000.0
    priority: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnularZone):
            return NotImplemented
        return self.zone_id == other.zone_id

    def __hash__(self) -> int:
        return hash(self.zone_id)

    def contains(self, position: np.ndarray) -> bool:
        """Test if position is within the annulus."""
        dx = float(position[0]) - float(self.center_xy[0])
        dy = float(position[1]) - float(self.center_xy[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < self.inner_radius_m or dist > self.outer_radius_m:
            return False
        return _check_altitude(position, self.altitude_min_m, self.altitude_max_m)

    def to_dict(self) -> dict:
        return {
            "type": "annular",
            "zone_id": self.zone_id,
            "name": self.name,
            "center_xy": self.center_xy.tolist(),
            "inner_radius_m": self.inner_radius_m,
            "outer_radius_m": self.outer_radius_m,
            "authorization": self.authorization.value,
            "altitude_min_m": self.altitude_min_m,
            "altitude_max_m": self.altitude_max_m,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SectorZone:
    """Pie-slice sector zone defined by center, radius, and azimuth range.

    Azimuth convention: 0° = North (+Y), increases clockwise (compass bearing).
    """

    zone_id: str
    name: str
    center_xy: np.ndarray
    radius_m: float
    azimuth_min_deg: float
    azimuth_max_deg: float
    authorization: ZoneAuth
    altitude_min_m: float = 0.0
    altitude_max_m: float = 30000.0
    priority: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SectorZone):
            return NotImplemented
        return self.zone_id == other.zone_id

    def __hash__(self) -> int:
        return hash(self.zone_id)

    def contains(self, position: np.ndarray) -> bool:
        """Test if position is within the sector."""
        dx = float(position[0]) - float(self.center_xy[0])
        dy = float(position[1]) - float(self.center_xy[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > self.radius_m:
            return False
        if not _check_altitude(position, self.altitude_min_m, self.altitude_max_m):
            return False
        # Compass bearing: 0=North(+Y), clockwise
        bearing = math.degrees(math.atan2(dx, dy)) % 360.0
        az_min = _normalize_angle(self.azimuth_min_deg)
        az_max = _normalize_angle(self.azimuth_max_deg)
        if az_min <= az_max:
            return az_min <= bearing <= az_max
        else:
            # Wraps around 360/0 (e.g., 350° to 10°)
            return bearing >= az_min or bearing <= az_max

    def to_dict(self) -> dict:
        return {
            "type": "sector",
            "zone_id": self.zone_id,
            "name": self.name,
            "center_xy": self.center_xy.tolist(),
            "radius_m": self.radius_m,
            "azimuth_min_deg": self.azimuth_min_deg,
            "azimuth_max_deg": self.azimuth_max_deg,
            "authorization": self.authorization.value,
            "altitude_min_m": self.altitude_min_m,
            "altitude_max_m": self.altitude_max_m,
            "priority": self.priority,
        }


# Union type for all zone kinds
EngagementZone = Union[CircularZone, PolygonZone, AnnularZone, SectorZone]


class ZoneManager:
    """Manages a collection of geographic engagement zones.

    Zone resolution: when a point falls in multiple zones,
    the zone with the highest priority wins. Among equal priorities,
    the most restrictive authorization wins.
    """

    def __init__(
        self,
        zones: list[EngagementZone] | None = None,
        default_authorization: ZoneAuth = ZoneAuth.WEAPONS_FREE,
    ):
        self._zones: list[EngagementZone] = list(zones) if zones else []
        self._default_auth = default_authorization

    def add_zone(self, zone: EngagementZone) -> None:
        self._zones.append(zone)

    def remove_zone(self, zone_id: str) -> None:
        self._zones = [z for z in self._zones if z.zone_id != zone_id]

    def get_all_zones(self) -> list[EngagementZone]:
        return list(self._zones)

    def get_containing_zones(self, position: np.ndarray) -> list[EngagementZone]:
        """Return all zones containing the given position."""
        return [z for z in self._zones if z.contains(position)]

    def resolve_authorization(self, position: np.ndarray) -> ZoneAuth:
        """Determine the effective authorization for a position.

        Returns the authorization from the highest-priority containing zone.
        If multiple zones share the highest priority, the most restrictive wins.
        If no zones contain the point, returns the default authorization.
        """
        containing = self.get_containing_zones(position)
        if not containing:
            return self._default_auth

        # Find max priority
        max_priority = max(z.priority for z in containing)
        top_zones = [z for z in containing if z.priority == max_priority]

        if len(top_zones) == 1:
            return top_zones[0].authorization

        # Among equal priority, most restrictive wins
        best_idx = len(_ZONE_AUTH_RESTRICTIVENESS)
        for z in top_zones:
            try:
                idx = _ZONE_AUTH_RESTRICTIVENESS.index(z.authorization)
            except ValueError:
                idx = len(_ZONE_AUTH_RESTRICTIVENESS)
            if idx < best_idx:
                best_idx = idx
        return _ZONE_AUTH_RESTRICTIVENESS[best_idx]

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        geo_context: Any = None,
        default_authorization: ZoneAuth = ZoneAuth.WEAPONS_FREE,
    ) -> ZoneManager:
        """Build ZoneManager from config list of zone definitions.

        Each zone dict must have a ``type`` key ("circle", "polygon",
        "annular", or "sector") plus the relevant parameters.
        If ``center_geo`` or ``vertices_geo`` keys are present and a
        ``geo_context`` is provided, geodetic coordinates are converted
        to ENU meters automatically.
        """
        zones: list[EngagementZone] = []
        zone_defs = cfg if isinstance(cfg, (list, tuple)) else []

        for zd in zone_defs:
            zd = dict(zd) if not isinstance(zd, dict) else zd
            ztype = zd.get("type", "circle")
            zone_id = zd.get("zone_id", f"ZONE-{len(zones)}")
            name = zd.get("name", zone_id)
            auth = _parse_zone_auth(zd.get("authorization", "weapons_free"))
            alt_min = float(zd.get("altitude_min_m", 0.0))
            alt_max = float(zd.get("altitude_max_m", 30000.0))
            priority = int(zd.get("priority", 0))

            try:
                if ztype == "circle":
                    center = _resolve_center(zd, geo_context)
                    zones.append(CircularZone(
                        zone_id=zone_id, name=name,
                        center_xy=center,
                        radius_m=float(zd.get("radius_m", 10000.0)),
                        authorization=auth,
                        altitude_min_m=alt_min, altitude_max_m=alt_max,
                        priority=priority,
                    ))
                elif ztype == "polygon":
                    verts = _resolve_vertices(zd, geo_context)
                    zones.append(PolygonZone(
                        zone_id=zone_id, name=name,
                        vertices=verts,
                        authorization=auth,
                        altitude_min_m=alt_min, altitude_max_m=alt_max,
                        priority=priority,
                    ))
                elif ztype == "annular":
                    center = _resolve_center(zd, geo_context)
                    zones.append(AnnularZone(
                        zone_id=zone_id, name=name,
                        center_xy=center,
                        inner_radius_m=float(zd.get("inner_radius_m", 0.0)),
                        outer_radius_m=float(zd.get("outer_radius_m", 10000.0)),
                        authorization=auth,
                        altitude_min_m=alt_min, altitude_max_m=alt_max,
                        priority=priority,
                    ))
                elif ztype == "sector":
                    center = _resolve_center(zd, geo_context)
                    zones.append(SectorZone(
                        zone_id=zone_id, name=name,
                        center_xy=center,
                        radius_m=float(zd.get("radius_m", 10000.0)),
                        azimuth_min_deg=float(zd.get("azimuth_min_deg", 0.0)),
                        azimuth_max_deg=float(zd.get("azimuth_max_deg", 360.0)),
                        authorization=auth,
                        altitude_min_m=alt_min, altitude_max_m=alt_max,
                        priority=priority,
                    ))
                else:
                    logger.warning("Unknown zone type '%s', skipping", ztype)
            except Exception:
                logger.exception("Failed to parse zone '%s'", zone_id)

        return cls(zones=zones, default_authorization=default_authorization)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_zone_auth(value: str) -> ZoneAuth:
    """Parse a zone authorization string to enum."""
    try:
        return ZoneAuth(value.lower())
    except ValueError:
        logger.warning("Unknown zone auth '%s', defaulting to WEAPONS_FREE", value)
        return ZoneAuth.WEAPONS_FREE


def _resolve_center(zd: dict, geo_context: Any) -> np.ndarray:
    """Resolve zone center from ENU or geodetic coordinates."""
    if "center_geo" in zd and geo_context is not None:
        lat, lon = zd["center_geo"][:2]
        alt = zd["center_geo"][2] if len(zd["center_geo"]) > 2 else 0.0
        e, n, _u = geo_context.geodetic_to_enu(lat, lon, alt)
        return np.array([e, n])
    center = zd.get("center_xy", [0.0, 0.0])
    return np.array(center, dtype=float)


def _resolve_vertices(zd: dict, geo_context: Any) -> np.ndarray:
    """Resolve polygon vertices from ENU or geodetic coordinates."""
    if "vertices_geo" in zd and geo_context is not None:
        verts = []
        for pt in zd["vertices_geo"]:
            lat, lon = pt[:2]
            alt = pt[2] if len(pt) > 2 else 0.0
            e, n, _u = geo_context.geodetic_to_enu(lat, lon, alt)
            verts.append([e, n])
        return np.array(verts, dtype=float)
    return np.array(zd.get("vertices", [[0, 0], [1, 0], [1, 1], [0, 1]]), dtype=float)
