"""Bidirectional conversion between SENTINEL track types and Link 16 J-series.

The adapter converts EnhancedFusedTrack objects to J2.2 Air Track messages
(and other J-series types), and converts received J-series messages back to
RemoteTrack-compatible dicts for composite fusion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sentinel.core.types import L16Identity, L16MessageType
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.track_mapping import TrackNumberAllocator

# ------------------------------------------------------------------
# Identity mapping tables
# ------------------------------------------------------------------

_IFF_TO_L16: dict[str, L16Identity] = {
    "friendly": L16Identity.FRIEND,
    "hostile": L16Identity.HOSTILE,
    "unknown": L16Identity.UNKNOWN,
    "pending": L16Identity.PENDING,
    "assumed_friendly": L16Identity.ASSUMED_FRIEND,
    "assumed_hostile": L16Identity.SUSPECT,
    "spoof_suspect": L16Identity.HOSTILE,
}

_L16_TO_IFF: dict[L16Identity, str] = {
    L16Identity.PENDING: "pending",
    L16Identity.UNKNOWN: "unknown",
    L16Identity.ASSUMED_FRIEND: "assumed_friendly",
    L16Identity.FRIEND: "friendly",
    L16Identity.NEUTRAL: "unknown",
    L16Identity.SUSPECT: "assumed_hostile",
    L16Identity.HOSTILE: "hostile",
    L16Identity.JOKER: "unknown",
}

_THREAT_TO_INT: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
_INT_TO_THREAT: dict[int, str] = {v: k for k, v in _THREAT_TO_INT.items()}

_ENGAGEMENT_AUTH_TO_INT: dict[str, int] = {
    "weapons_free": 0, "weapons_tight": 1, "weapons_hold": 2, "hold_fire": 3,
}
_INT_TO_ENGAGEMENT_AUTH: dict[int, str] = {v: k for k, v in _ENGAGEMENT_AUTH_TO_INT.items()}

_METERS_TO_FEET = 3.28084
_FEET_TO_METERS = 1.0 / _METERS_TO_FEET
_MPS_TO_KNOTS = 1.94384
_KNOTS_TO_MPS = 1.0 / _MPS_TO_KNOTS


class DataLinkAdapter:
    """Converts between SENTINEL types and Link 16 J-series messages."""

    def __init__(
        self,
        geo_context: Any | None = None,
        track_allocator: TrackNumberAllocator | None = None,
    ) -> None:
        self._geo = geo_context
        self._allocator = track_allocator or TrackNumberAllocator()

    # ------------------------------------------------------------------
    # Outbound: SENTINEL -> Link 16
    # ------------------------------------------------------------------

    def track_to_j2_2(self, track: Any, timestamp: float) -> J2_2AirTrack | None:
        """Convert an EnhancedFusedTrack (or similar) to J2.2."""
        # Get geodetic position
        lat, lon, alt_m = self._extract_geodetic(track)
        if lat is None:
            return None  # cannot encode without position

        # Get SENTINEL track ID
        sentinel_id = getattr(track, "fused_id", "") or getattr(track, "track_id", "")
        track_number = self._allocator.get_or_allocate(sentinel_id)

        # Velocity -> speed/course
        speed_knots, course_deg = self._velocity_to_speed_course(track)

        # Identity from IFF
        iff_str = getattr(track, "iff_identification", "unknown")
        identity = _IFF_TO_L16.get(iff_str, L16Identity.UNKNOWN)

        # Track quality
        quality = self._compute_track_quality(track)

        # Sensor count -> strength
        strength = min(getattr(track, "sensor_count", 1), 7)

        # IFF Mode 3A
        mode_3a_str = getattr(track, "iff_mode_3a_code", None)
        iff_mode_3a = -1
        if mode_3a_str:
            try:
                iff_mode_3a = int(mode_3a_str, 8) if isinstance(mode_3a_str, str) else int(mode_3a_str)
            except (ValueError, TypeError):
                pass

        # Threat level
        threat_str = getattr(track, "threat_level", "LOW")
        threat_level = _THREAT_TO_INT.get(str(threat_str).upper(), 0)

        # Environment flags
        env_flags = 0
        if getattr(track, "is_stealth_candidate", False):
            env_flags |= 0b1000
        if getattr(track, "is_hypersonic_candidate", False):
            env_flags |= 0b0100
        if getattr(track, "is_decoy_candidate", False):
            env_flags |= 0b0010
        if getattr(track, "is_chaff_candidate", False):
            env_flags |= 0b0001

        alt_ft = int(alt_m * _METERS_TO_FEET) if alt_m is not None else 0

        return J2_2AirTrack(
            track_number=track_number,
            identity=identity,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_ft=max(0, min(alt_ft, 16383)),
            speed_knots=max(0, min(speed_knots, 1023)),
            course_deg=course_deg % 360.0,
            track_quality=quality,
            strength=strength,
            iff_mode_3a=iff_mode_3a,
            threat_level=threat_level,
            environment_flags=env_flags,
            source_sentinel_id=sentinel_id,
            timestamp=timestamp,
        )

    def track_to_j3_2(
        self, track: Any, action: int, timestamp: float
    ) -> J3_2TrackManagement | None:
        """Convert track status change to J3.2."""
        sentinel_id = getattr(track, "fused_id", "") or getattr(track, "track_id", "")
        track_number = self._allocator.get_or_allocate(sentinel_id)
        lat, lon, _ = self._extract_geodetic(track)
        iff_str = getattr(track, "iff_identification", "unknown")
        identity = _IFF_TO_L16.get(iff_str, L16Identity.UNKNOWN)
        quality = self._compute_track_quality(track)

        return J3_2TrackManagement(
            track_number=track_number,
            identity=identity,
            action=action,
            track_quality=quality,
            latitude_deg=lat or 0.0,
            longitude_deg=lon or 0.0,
            source_sentinel_id=sentinel_id,
            timestamp=timestamp,
        )

    def engagement_to_j3_5(
        self,
        track_id: str,
        engagement_auth: str,
        weapon_type: int = 0,
        engagement_status: int = 0,
        timestamp: float = 0.0,
    ) -> J3_5EngagementStatus | None:
        """Convert engagement status to J3.5."""
        track_number = self._allocator.get_or_allocate(track_id)
        auth_int = _ENGAGEMENT_AUTH_TO_INT.get(engagement_auth, 2)

        return J3_5EngagementStatus(
            track_number=track_number,
            engagement_auth=auth_int,
            weapon_type=weapon_type & 0x7,
            engagement_status=engagement_status & 0x7,
            source_sentinel_id=track_id,
            timestamp=timestamp,
        )

    def iff_result_to_j7_0(
        self, track_id: str, iff_result: dict, timestamp: float = 0.0
    ) -> J7_0IFF | None:
        """Convert IFF result dict to J7.0."""
        track_number = self._allocator.get_or_allocate(track_id)
        ident_str = iff_result.get("identification", "unknown")
        identity = _IFF_TO_L16.get(ident_str, L16Identity.UNKNOWN)

        mode_3a_str = iff_result.get("mode_3a_code")
        mode_3a = -1
        if mode_3a_str:
            try:
                mode_3a = int(mode_3a_str, 8) if isinstance(mode_3a_str, str) else int(mode_3a_str)
            except (ValueError, TypeError):
                pass

        mode_s_str = iff_result.get("mode_s_address")
        mode_s = -1
        if mode_s_str:
            try:
                mode_s = int(mode_s_str, 16) if isinstance(mode_s_str, str) else int(mode_s_str)
            except (ValueError, TypeError):
                pass

        return J7_0IFF(
            track_number=track_number,
            identity=identity,
            mode_3a=mode_3a,
            mode_s_address=mode_s,
            mode_4_valid=bool(iff_result.get("mode_4_valid", False)),
            mode_5_valid=bool(iff_result.get("mode_5_valid", False)),
            source_sentinel_id=track_id,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Inbound: Link 16 -> SENTINEL
    # ------------------------------------------------------------------

    def j2_2_to_remote_track(self, msg: J2_2AirTrack) -> dict:
        """Convert J2.2 to a RemoteTrack-compatible dict."""
        # Geodetic -> ENU position (if geo context available)
        position = None
        if self._geo is not None:
            try:
                enu = self._geo.geodetic_to_enu(
                    msg.latitude_deg, msg.longitude_deg,
                    msg.altitude_ft * _FEET_TO_METERS,
                )
                position = np.array(enu[:2], dtype=np.float64)
            except Exception:
                pass

        # Speed + course -> velocity vector
        speed_mps = msg.speed_knots * _KNOTS_TO_MPS
        course_rad = math.radians(msg.course_deg)
        vx = speed_mps * math.sin(course_rad)
        vy = speed_mps * math.cos(course_rad)
        velocity = np.array([vx, vy], dtype=np.float64)

        # Identity
        iff_str = _L16_TO_IFF.get(msg.identity, "unknown")

        # Threat
        threat_str = _INT_TO_THREAT.get(msg.threat_level, "LOW")

        # Quality -> confidence (0-7 mapped to 0.0-1.0)
        confidence = msg.track_quality / 7.0

        sentinel_id = self._allocator.get_sentinel_id(msg.track_number)

        return {
            "track_id": sentinel_id or f"L16-{msg.track_number}",
            "source_node": "LINK16",
            "position": position,
            "velocity": velocity,
            "covariance": None,
            "position_geo": (msg.latitude_deg, msg.longitude_deg, msg.altitude_ft * _FEET_TO_METERS),
            "sensor_types": ["link16"],
            "threat_level": threat_str,
            "iff_identification": iff_str,
            "engagement_auth": "",
            "confidence": confidence,
            "update_time": msg.timestamp,
        }

    def j7_0_to_iff_result(self, msg: J7_0IFF) -> dict:
        """Convert J7.0 to IFF result dict."""
        return {
            "identification": _L16_TO_IFF.get(msg.identity, "unknown"),
            "mode_3a_code": oct(msg.mode_3a) if msg.mode_3a >= 0 else None,
            "mode_s_address": hex(msg.mode_s_address) if msg.mode_s_address >= 0 else None,
            "mode_4_valid": msg.mode_4_valid,
            "mode_5_valid": msg.mode_5_valid,
        }

    def j3_5_to_engagement(self, msg: J3_5EngagementStatus) -> dict:
        """Convert J3.5 to engagement status dict."""
        return {
            "track_id": self._allocator.get_sentinel_id(msg.track_number) or f"L16-{msg.track_number}",
            "engagement_auth": _INT_TO_ENGAGEMENT_AUTH.get(msg.engagement_auth, "weapons_hold"),
            "weapon_type": msg.weapon_type,
            "engagement_status": msg.engagement_status,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_geodetic(self, track: Any) -> tuple[float | None, float | None, float | None]:
        """Get (lat, lon, alt_m) from a track, converting ENU if needed."""
        # Try direct geodetic position
        pos_geo = getattr(track, "position_geo", None)
        if pos_geo is not None:
            if isinstance(pos_geo, dict):
                lat = pos_geo.get("lat") or pos_geo.get("latitude")
                lon = pos_geo.get("lon") or pos_geo.get("longitude")
                alt = pos_geo.get("alt") or pos_geo.get("altitude", 0)
                if lat is not None and lon is not None:
                    return float(lat), float(lon), float(alt or 0)
            elif isinstance(pos_geo, (tuple, list)) and len(pos_geo) >= 2:
                return float(pos_geo[0]), float(pos_geo[1]), float(pos_geo[2] if len(pos_geo) > 2 else 0)

        # Try ENU + geo context
        if self._geo is not None:
            pos = getattr(track, "position_m", None)
            if pos is None:
                pos = getattr(track, "position", None)
            if pos is not None and hasattr(pos, '__len__') and len(pos) >= 2:
                try:
                    alt_m = float(pos[2]) if len(pos) > 2 else 0.0
                    lat, lon, alt = self._geo.enu_to_geodetic(
                        float(pos[0]), float(pos[1]), alt_m
                    )
                    return lat, lon, alt
                except Exception:
                    pass

        return None, None, None

    def _velocity_to_speed_course(self, track: Any) -> tuple[int, float]:
        """Extract speed (knots) and course (degrees) from track velocity."""
        vel = getattr(track, "velocity", None)
        if vel is None:
            vel = getattr(track, "velocity_mps", None)

        if vel is not None and hasattr(vel, '__len__') and len(vel) >= 2:
            vx, vy = float(vel[0]), float(vel[1])
            speed_mps = math.sqrt(vx * vx + vy * vy)
            speed_knots = int(speed_mps * _MPS_TO_KNOTS)
            if speed_mps > 0.01:
                course = math.degrees(math.atan2(vx, vy)) % 360.0
            else:
                course = 0.0
            return speed_knots, course

        # Fallback: scalar velocity_mps
        v_scalar = getattr(track, "velocity_mps", None)
        if isinstance(v_scalar, (int, float)):
            return int(float(v_scalar) * _MPS_TO_KNOTS), 0.0

        return 0, 0.0

    @staticmethod
    def _compute_track_quality(track: Any) -> int:
        """Map track metadata to 0-7 quality scale."""
        q = 0
        q += min(getattr(track, "sensor_count", 1), 4)
        if getattr(track, "iff_identification", "unknown") not in ("unknown", "pending"):
            q += 1
        conf = getattr(track, "confidence", None) or getattr(track, "fusion_quality", None)
        if conf is not None and float(conf) > 0.7:
            q += 1
        return min(q, 7)
