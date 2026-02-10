"""Link 16 J-series message dataclasses (STANAG 5516).

Frozen dataclasses representing the four supported J-series message types:
J2.2 Air Track, J3.2 Track Management, J3.5 Engagement Status, J7.0 IFF/SIF.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.core.types import L16Identity, L16MessageType


@dataclass(frozen=True)
class J2_2AirTrack:
    """J2.2 Air Track message — primary position/velocity/identity report.

    Binary layout: 128 bits (16 bytes).
    """

    message_type: L16MessageType = L16MessageType.J2_2

    # Track identification
    track_number: int = 0               # 13-bit, 0-8191
    identity: L16Identity = L16Identity.UNKNOWN  # 3-bit

    # Position (WGS84)
    latitude_deg: float = 0.0           # 23-bit signed
    longitude_deg: float = 0.0          # 24-bit signed
    altitude_ft: int = 0                # 14-bit unsigned, feet MSL

    # Kinematics
    speed_knots: int = 0                # 10-bit unsigned, 0-1023
    course_deg: float = 0.0             # 9-bit unsigned, LSB=0.703125 deg

    # Quality / metadata
    track_quality: int = 0              # 3-bit, 0-7
    strength: int = 0                   # 3-bit, 0-7 (contributing sensors)

    # IFF
    iff_mode_3a: int = -1               # 12-bit octal (0-4095), -1=no code

    # SENTINEL extensions (encoded in spare bits)
    threat_level: int = 0               # 2-bit: 0=LOW, 1=MED, 2=HIGH, 3=CRITICAL
    environment_flags: int = 0           # 4-bit: stealth|hypersonic|decoy|jammed

    # Source metadata (not in wire format)
    source_sentinel_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class J3_2TrackManagement:
    """J3.2 Track Management — track status changes.

    Binary layout: 72 bits (9 bytes).
    """

    message_type: L16MessageType = L16MessageType.J3_2

    track_number: int = 0
    identity: L16Identity = L16Identity.UNKNOWN

    action: int = 0                     # 3-bit: 0=update, 1=drop, 2=transfer, 3=quality_change
    track_quality: int = 0              # 3-bit

    latitude_deg: float = 0.0           # 23-bit signed
    longitude_deg: float = 0.0          # 24-bit signed

    source_sentinel_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class J3_5EngagementStatus:
    """J3.5 Engagement Status — engagement coordination.

    Binary layout: 32 bits (4 bytes).
    """

    message_type: L16MessageType = L16MessageType.J3_5

    track_number: int = 0

    engagement_auth: int = 0            # 3-bit
    weapon_type: int = 0                # 3-bit
    engagement_status: int = 0          # 3-bit

    source_sentinel_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class J7_0IFF:
    """J7.0 IFF/SIF Management — IFF interrogation results.

    Binary layout: 96 bits (12 bytes).
    """

    message_type: L16MessageType = L16MessageType.J7_0

    track_number: int = 0
    identity: L16Identity = L16Identity.UNKNOWN

    # IFF codes (-1 = no code available)
    mode_1: int = -1                    # 6-bit (0-63)
    mode_2: int = -1                    # 12-bit (0-4095)
    mode_3a: int = -1                   # 12-bit octal (0-4095)
    mode_c_alt_ft: int = -1             # 13-bit in 100ft increments
    mode_s_address: int = -1            # 24-bit ICAO address

    mode_4_valid: bool = False          # 1-bit
    mode_5_valid: bool = False          # 1-bit

    source_sentinel_id: str = ""
    timestamp: float = 0.0
