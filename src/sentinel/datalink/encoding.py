"""Binary encoding/decoding for Link 16 J-series messages.

Each codec packs a J-series dataclass into a fixed-size byte buffer
using BitWriter/BitReader for sub-byte field widths.
"""

from __future__ import annotations

from sentinel.core.types import L16Identity, L16MessageType
from sentinel.datalink.codec import BitReader, BitWriter
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)

# ------------------------------------------------------------------
# Identity ordinal mapping (3-bit)
# ------------------------------------------------------------------

_IDENTITY_TO_INT: dict[L16Identity, int] = {
    L16Identity.PENDING: 0,
    L16Identity.UNKNOWN: 1,
    L16Identity.ASSUMED_FRIEND: 2,
    L16Identity.FRIEND: 3,
    L16Identity.NEUTRAL: 4,
    L16Identity.SUSPECT: 5,
    L16Identity.HOSTILE: 6,
    L16Identity.JOKER: 7,
}
_INT_TO_IDENTITY: dict[int, L16Identity] = {v: k for k, v in _IDENTITY_TO_INT.items()}

# ------------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------------

# Latitude: 23-bit signed, LSB = 180 / 2^22 ≈ 0.0000429 deg
_LAT_SCALE = 180.0 / (1 << 22)
# Longitude: 24-bit signed, LSB = 360 / 2^23 ≈ 0.0000429 deg
_LON_SCALE = 360.0 / (1 << 23)
# Course: 9-bit unsigned, LSB = 360/512 = 0.703125 deg
_COURSE_SCALE = 360.0 / 512

_NO_CODE_12 = 0xFFF      # 12-bit sentinel for "no code"
_NO_CODE_6 = 0x3F         # 6-bit sentinel
_NO_CODE_13 = 0x1FFF      # 13-bit sentinel
_NO_CODE_24 = 0xFFFFFF    # 24-bit sentinel


def _encode_lat(deg: float) -> int:
    return round(max(-90.0, min(90.0, deg)) / _LAT_SCALE)


def _decode_lat(enc: int) -> float:
    return enc * _LAT_SCALE


def _encode_lon(deg: float) -> int:
    return round(max(-180.0, min(180.0, deg)) / _LON_SCALE)


def _decode_lon(enc: int) -> float:
    return enc * _LON_SCALE


def _encode_course(deg: float) -> int:
    return round((deg % 360.0) / _COURSE_SCALE) & 0x1FF


def _decode_course(enc: int) -> float:
    return enc * _COURSE_SCALE


# ------------------------------------------------------------------
# Message sub-type tags (4-bit)
# ------------------------------------------------------------------

_SUBTAG_J2_2 = 0x2
_SUBTAG_J3_2 = 0x3
_SUBTAG_J3_5 = 0x5
_SUBTAG_J7_0 = 0x7


# ------------------------------------------------------------------
# J2.2 Air Track — 128 bits (16 bytes)
# ------------------------------------------------------------------


class J2_2Codec:
    """Encode/decode J2.2 Air Track messages."""

    BYTE_SIZE = 16

    @staticmethod
    def encode(msg: J2_2AirTrack) -> bytes:
        w = BitWriter()
        w.write_unsigned(_SUBTAG_J2_2, 4)
        w.write_unsigned(msg.track_number & 0x1FFF, 13)
        w.write_unsigned(_IDENTITY_TO_INT.get(msg.identity, 1), 3)
        w.write_signed(_encode_lat(msg.latitude_deg), 23)
        w.write_signed(_encode_lon(msg.longitude_deg), 24)
        w.write_unsigned(min(max(msg.altitude_ft, 0), 16383), 14)
        w.write_unsigned(min(max(msg.speed_knots, 0), 1023), 10)
        w.write_unsigned(_encode_course(msg.course_deg), 9)
        w.write_unsigned(msg.track_quality & 0x7, 3)
        w.write_unsigned(msg.strength & 0x7, 3)
        w.write_unsigned(msg.iff_mode_3a if msg.iff_mode_3a >= 0 else _NO_CODE_12, 12)
        w.write_unsigned(msg.threat_level & 0x3, 2)
        w.write_unsigned(msg.environment_flags & 0xF, 4)
        w.write_unsigned(0, 4)  # reserved
        return w.to_bytes()

    @staticmethod
    def decode(data: bytes) -> J2_2AirTrack:
        r = BitReader(data)
        tag = r.read_unsigned(4)
        if tag != _SUBTAG_J2_2:
            raise ValueError(f"Expected J2.2 tag 0x{_SUBTAG_J2_2:X}, got 0x{tag:X}")
        track_number = r.read_unsigned(13)
        identity = _INT_TO_IDENTITY.get(r.read_unsigned(3), L16Identity.UNKNOWN)
        lat = _decode_lat(r.read_signed(23))
        lon = _decode_lon(r.read_signed(24))
        alt = r.read_unsigned(14)
        speed = r.read_unsigned(10)
        course = _decode_course(r.read_unsigned(9))
        quality = r.read_unsigned(3)
        strength = r.read_unsigned(3)
        mode_3a_raw = r.read_unsigned(12)
        mode_3a = mode_3a_raw if mode_3a_raw != _NO_CODE_12 else -1
        threat = r.read_unsigned(2)
        env_flags = r.read_unsigned(4)
        r.read_unsigned(4)  # reserved
        return J2_2AirTrack(
            track_number=track_number,
            identity=identity,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_ft=alt,
            speed_knots=speed,
            course_deg=course,
            track_quality=quality,
            strength=strength,
            iff_mode_3a=mode_3a,
            threat_level=threat,
            environment_flags=env_flags,
        )


# ------------------------------------------------------------------
# J3.2 Track Management — 72 bits (9 bytes)
# ------------------------------------------------------------------


class J3_2Codec:
    """Encode/decode J3.2 Track Management messages."""

    BYTE_SIZE = 9

    @staticmethod
    def encode(msg: J3_2TrackManagement) -> bytes:
        w = BitWriter()
        w.write_unsigned(_SUBTAG_J3_2, 4)
        w.write_unsigned(msg.track_number & 0x1FFF, 13)
        w.write_unsigned(_IDENTITY_TO_INT.get(msg.identity, 1), 3)
        w.write_unsigned(msg.action & 0x7, 3)
        w.write_unsigned(msg.track_quality & 0x7, 3)
        w.write_signed(_encode_lat(msg.latitude_deg), 23)
        w.write_signed(_encode_lon(msg.longitude_deg), 24)
        # 4+13+3+3+3+23+24 = 73 -> pad to 9 bytes = 72 bits
        # Actually 73 bits, so we let pad_to_byte handle it
        return w.to_bytes()

    @staticmethod
    def decode(data: bytes) -> J3_2TrackManagement:
        r = BitReader(data)
        tag = r.read_unsigned(4)
        if tag != _SUBTAG_J3_2:
            raise ValueError(f"Expected J3.2 tag 0x{_SUBTAG_J3_2:X}, got 0x{tag:X}")
        track_number = r.read_unsigned(13)
        identity = _INT_TO_IDENTITY.get(r.read_unsigned(3), L16Identity.UNKNOWN)
        action = r.read_unsigned(3)
        quality = r.read_unsigned(3)
        lat = _decode_lat(r.read_signed(23))
        lon = _decode_lon(r.read_signed(24))
        return J3_2TrackManagement(
            track_number=track_number,
            identity=identity,
            action=action,
            track_quality=quality,
            latitude_deg=lat,
            longitude_deg=lon,
        )


# ------------------------------------------------------------------
# J3.5 Engagement Status — 32 bits (4 bytes)
# ------------------------------------------------------------------


class J3_5Codec:
    """Encode/decode J3.5 Engagement Status messages."""

    BYTE_SIZE = 4

    @staticmethod
    def encode(msg: J3_5EngagementStatus) -> bytes:
        w = BitWriter()
        w.write_unsigned(_SUBTAG_J3_5, 4)
        w.write_unsigned(msg.track_number & 0x1FFF, 13)
        w.write_unsigned(msg.engagement_auth & 0x7, 3)
        w.write_unsigned(msg.weapon_type & 0x7, 3)
        w.write_unsigned(msg.engagement_status & 0x7, 3)
        w.write_unsigned(0, 6)  # reserved
        return w.to_bytes()

    @staticmethod
    def decode(data: bytes) -> J3_5EngagementStatus:
        r = BitReader(data)
        tag = r.read_unsigned(4)
        if tag != _SUBTAG_J3_5:
            raise ValueError(f"Expected J3.5 tag 0x{_SUBTAG_J3_5:X}, got 0x{tag:X}")
        track_number = r.read_unsigned(13)
        eng_auth = r.read_unsigned(3)
        weapon = r.read_unsigned(3)
        eng_status = r.read_unsigned(3)
        r.read_unsigned(6)  # reserved
        return J3_5EngagementStatus(
            track_number=track_number,
            engagement_auth=eng_auth,
            weapon_type=weapon,
            engagement_status=eng_status,
        )


# ------------------------------------------------------------------
# J7.0 IFF/SIF — 96 bits (12 bytes)
# ------------------------------------------------------------------


class J7_0Codec:
    """Encode/decode J7.0 IFF/SIF Management messages."""

    BYTE_SIZE = 12

    @staticmethod
    def encode(msg: J7_0IFF) -> bytes:
        w = BitWriter()
        w.write_unsigned(_SUBTAG_J7_0, 4)
        w.write_unsigned(msg.track_number & 0x1FFF, 13)
        w.write_unsigned(_IDENTITY_TO_INT.get(msg.identity, 1), 3)
        w.write_unsigned(msg.mode_1 if msg.mode_1 >= 0 else _NO_CODE_6, 6)
        w.write_unsigned(msg.mode_2 if msg.mode_2 >= 0 else _NO_CODE_12, 12)
        w.write_unsigned(msg.mode_3a if msg.mode_3a >= 0 else _NO_CODE_12, 12)
        w.write_unsigned(msg.mode_c_alt_ft if msg.mode_c_alt_ft >= 0 else _NO_CODE_13, 13)
        w.write_unsigned(msg.mode_s_address if msg.mode_s_address >= 0 else _NO_CODE_24, 24)
        w.write_bool(msg.mode_4_valid)
        w.write_bool(msg.mode_5_valid)
        w.write_unsigned(0, 7)  # reserved, total = 4+13+3+6+12+12+13+24+1+1+7 = 96
        return w.to_bytes()

    @staticmethod
    def decode(data: bytes) -> J7_0IFF:
        r = BitReader(data)
        tag = r.read_unsigned(4)
        if tag != _SUBTAG_J7_0:
            raise ValueError(f"Expected J7.0 tag 0x{_SUBTAG_J7_0:X}, got 0x{tag:X}")
        track_number = r.read_unsigned(13)
        identity = _INT_TO_IDENTITY.get(r.read_unsigned(3), L16Identity.UNKNOWN)
        m1 = r.read_unsigned(6)
        m2 = r.read_unsigned(12)
        m3a = r.read_unsigned(12)
        mc = r.read_unsigned(13)
        ms = r.read_unsigned(24)
        m4 = r.read_bool()
        m5 = r.read_bool()
        r.read_unsigned(7)  # reserved
        return J7_0IFF(
            track_number=track_number,
            identity=identity,
            mode_1=m1 if m1 != _NO_CODE_6 else -1,
            mode_2=m2 if m2 != _NO_CODE_12 else -1,
            mode_3a=m3a if m3a != _NO_CODE_12 else -1,
            mode_c_alt_ft=mc if mc != _NO_CODE_13 else -1,
            mode_s_address=ms if ms != _NO_CODE_24 else -1,
            mode_4_valid=m4,
            mode_5_valid=m5,
        )


# ------------------------------------------------------------------
# Dispatch helpers
# ------------------------------------------------------------------


def peek_message_type(data: bytes) -> L16MessageType | None:
    """Peek at the 4-bit sub-type tag without consuming the buffer."""
    if len(data) < 1:
        return None
    tag = (data[0] >> 4) & 0xF
    return {
        _SUBTAG_J2_2: L16MessageType.J2_2,
        _SUBTAG_J3_2: L16MessageType.J3_2,
        _SUBTAG_J3_5: L16MessageType.J3_5,
        _SUBTAG_J7_0: L16MessageType.J7_0,
    }.get(tag)


def decode_message(data: bytes):
    """Decode a message by peeking at its sub-type tag."""
    msg_type = peek_message_type(data)
    if msg_type is None:
        raise ValueError("Cannot determine message type from data")
    if msg_type == L16MessageType.J2_2:
        return J2_2Codec.decode(data)
    if msg_type == L16MessageType.J3_2:
        return J3_2Codec.decode(data)
    if msg_type == L16MessageType.J3_5:
        return J3_5Codec.decode(data)
    if msg_type == L16MessageType.J7_0:
        return J7_0Codec.decode(data)
    raise ValueError(f"Unsupported message type: {msg_type}")


def encode_message(msg) -> bytes:
    """Encode a J-series message by dispatching on its type."""
    if isinstance(msg, J2_2AirTrack):
        return J2_2Codec.encode(msg)
    if isinstance(msg, J3_2TrackManagement):
        return J3_2Codec.encode(msg)
    if isinstance(msg, J3_5EngagementStatus):
        return J3_5Codec.encode(msg)
    if isinstance(msg, J7_0IFF):
        return J7_0Codec.encode(msg)
    raise TypeError(f"Unknown message type: {type(msg)}")
