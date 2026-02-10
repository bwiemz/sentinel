"""Unit tests for J-series binary encoding/decoding."""

from __future__ import annotations

import pytest

from sentinel.core.types import L16Identity, L16MessageType
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.encoding import (
    J2_2Codec,
    J3_2Codec,
    J3_5Codec,
    J7_0Codec,
    decode_message,
    encode_message,
    peek_message_type,
)


# ---------------------------------------------------------------------------
# J2.2 Codec
# ---------------------------------------------------------------------------


class TestJ2_2Codec:
    def test_encode_size(self):
        data = J2_2Codec.encode(J2_2AirTrack())
        assert len(data) == 16

    def test_roundtrip_defaults(self):
        original = J2_2AirTrack()
        data = J2_2Codec.encode(original)
        decoded = J2_2Codec.decode(data)
        assert decoded.track_number == 0
        assert decoded.identity == L16Identity.UNKNOWN

    def test_roundtrip_full(self):
        original = J2_2AirTrack(
            track_number=4567,
            identity=L16Identity.HOSTILE,
            latitude_deg=37.7749,
            longitude_deg=-122.4194,
            altitude_ft=10000,
            speed_knots=480,
            course_deg=270.0,
            track_quality=5,
            strength=3,
            iff_mode_3a=1234,
            threat_level=2,
            environment_flags=0b0101,
        )
        decoded = J2_2Codec.decode(J2_2Codec.encode(original))
        assert decoded.track_number == 4567
        assert decoded.identity == L16Identity.HOSTILE
        assert decoded.latitude_deg == pytest.approx(37.7749, abs=0.001)
        assert decoded.longitude_deg == pytest.approx(-122.4194, abs=0.001)
        assert decoded.altitude_ft == 10000
        assert decoded.speed_knots == 480
        assert decoded.course_deg == pytest.approx(270.0, abs=1.0)
        assert decoded.track_quality == 5
        assert decoded.strength == 3
        assert decoded.iff_mode_3a == 1234
        assert decoded.threat_level == 2
        assert decoded.environment_flags == 0b0101

    def test_latitude_precision(self):
        """Position resolution should be ~5m."""
        for lat in [-90.0, -45.0, 0.0, 45.0, 89.999]:
            msg = J2_2AirTrack(latitude_deg=lat)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.latitude_deg == pytest.approx(lat, abs=0.0001)

    def test_longitude_precision(self):
        for lon in [-180.0, -90.0, 0.0, 90.0, 179.999]:
            msg = J2_2AirTrack(longitude_deg=lon)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.longitude_deg == pytest.approx(lon, abs=0.0001)

    def test_altitude_zero_and_max(self):
        for alt in [0, 16383]:
            msg = J2_2AirTrack(altitude_ft=alt)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.altitude_ft == alt

    def test_speed_range(self):
        for speed in [0, 100, 500, 1023]:
            msg = J2_2AirTrack(speed_knots=speed)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.speed_knots == speed

    def test_course_values(self):
        for course in [0.0, 90.0, 180.0, 270.0, 359.0]:
            msg = J2_2AirTrack(course_deg=course)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.course_deg == pytest.approx(course, abs=1.0)

    def test_all_identities(self):
        for ident in L16Identity:
            msg = J2_2AirTrack(identity=ident)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.identity == ident

    def test_iff_no_code(self):
        msg = J2_2AirTrack(iff_mode_3a=-1)
        decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
        assert decoded.iff_mode_3a == -1

    def test_iff_valid_code(self):
        msg = J2_2AirTrack(iff_mode_3a=7700)
        decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
        # 7700 > 4095 (12-bit max), so clamped
        assert 0 <= decoded.iff_mode_3a <= 4095

    def test_threat_levels(self):
        for tl in [0, 1, 2, 3]:
            msg = J2_2AirTrack(threat_level=tl)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.threat_level == tl

    def test_environment_flags(self):
        for flags in [0b0000, 0b1111, 0b1010, 0b0101]:
            msg = J2_2AirTrack(environment_flags=flags)
            decoded = J2_2Codec.decode(J2_2Codec.encode(msg))
            assert decoded.environment_flags == flags

    def test_wrong_tag_raises(self):
        data = J3_5Codec.encode(J3_5EngagementStatus())
        with pytest.raises(ValueError, match="Expected J2.2"):
            J2_2Codec.decode(data)


# ---------------------------------------------------------------------------
# J3.2 Codec
# ---------------------------------------------------------------------------


class TestJ3_2Codec:
    def test_encode_size(self):
        data = J3_2Codec.encode(J3_2TrackManagement())
        assert len(data) == 10  # 73 bits padded to 10 bytes

    def test_roundtrip_drop(self):
        msg = J3_2TrackManagement(
            track_number=100,
            identity=L16Identity.HOSTILE,
            action=1,
            track_quality=4,
            latitude_deg=51.5074,
            longitude_deg=-0.1278,
        )
        decoded = J3_2Codec.decode(J3_2Codec.encode(msg))
        assert decoded.track_number == 100
        assert decoded.action == 1
        assert decoded.latitude_deg == pytest.approx(51.5074, abs=0.001)

    def test_all_actions(self):
        for action in [0, 1, 2, 3]:
            msg = J3_2TrackManagement(action=action)
            decoded = J3_2Codec.decode(J3_2Codec.encode(msg))
            assert decoded.action == action


# ---------------------------------------------------------------------------
# J3.5 Codec
# ---------------------------------------------------------------------------


class TestJ3_5Codec:
    def test_encode_size(self):
        data = J3_5Codec.encode(J3_5EngagementStatus())
        assert len(data) == 4

    def test_roundtrip(self):
        msg = J3_5EngagementStatus(
            track_number=8000,
            engagement_auth=3,
            weapon_type=2,
            engagement_status=1,
        )
        decoded = J3_5Codec.decode(J3_5Codec.encode(msg))
        assert decoded.track_number == 8000
        assert decoded.engagement_auth == 3
        assert decoded.weapon_type == 2
        assert decoded.engagement_status == 1


# ---------------------------------------------------------------------------
# J7.0 Codec
# ---------------------------------------------------------------------------


class TestJ7_0Codec:
    def test_encode_size(self):
        data = J7_0Codec.encode(J7_0IFF())
        assert len(data) == 12

    def test_roundtrip_all_modes(self):
        msg = J7_0IFF(
            track_number=200,
            identity=L16Identity.FRIEND,
            mode_1=42,
            mode_2=3000,
            mode_3a=1200,
            mode_c_alt_ft=350,
            mode_s_address=0xABCDEF,
            mode_4_valid=True,
            mode_5_valid=True,
        )
        decoded = J7_0Codec.decode(J7_0Codec.encode(msg))
        assert decoded.track_number == 200
        assert decoded.identity == L16Identity.FRIEND
        assert decoded.mode_1 == 42
        assert decoded.mode_2 == 3000
        assert decoded.mode_3a == 1200
        assert decoded.mode_c_alt_ft == 350
        assert decoded.mode_s_address == 0xABCDEF
        assert decoded.mode_4_valid is True
        assert decoded.mode_5_valid is True

    def test_roundtrip_no_codes(self):
        msg = J7_0IFF()
        decoded = J7_0Codec.decode(J7_0Codec.encode(msg))
        assert decoded.mode_1 == -1
        assert decoded.mode_2 == -1
        assert decoded.mode_3a == -1
        assert decoded.mode_c_alt_ft == -1
        assert decoded.mode_s_address == -1

    def test_mode_s_boundary(self):
        msg = J7_0IFF(mode_s_address=0xFFFFFF)
        decoded = J7_0Codec.decode(J7_0Codec.encode(msg))
        # 0xFFFFFF is sentinel for "no code" -> should be -1
        assert decoded.mode_s_address == -1

    def test_mode_s_max_valid(self):
        msg = J7_0IFF(mode_s_address=0xFFFFFE)
        decoded = J7_0Codec.decode(J7_0Codec.encode(msg))
        assert decoded.mode_s_address == 0xFFFFFE


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_peek_j2_2(self):
        data = J2_2Codec.encode(J2_2AirTrack())
        assert peek_message_type(data) == L16MessageType.J2_2

    def test_peek_j3_5(self):
        data = J3_5Codec.encode(J3_5EngagementStatus())
        assert peek_message_type(data) == L16MessageType.J3_5

    def test_peek_empty(self):
        assert peek_message_type(b"") is None

    def test_decode_message_dispatch(self):
        msg = J2_2AirTrack(track_number=42)
        data = encode_message(msg)
        decoded = decode_message(data)
        assert isinstance(decoded, J2_2AirTrack)
        assert decoded.track_number == 42

    def test_encode_message_dispatch(self):
        msg = J7_0IFF(track_number=99)
        data = encode_message(msg)
        decoded = decode_message(data)
        assert isinstance(decoded, J7_0IFF)
        assert decoded.track_number == 99

    def test_multiple_messages(self):
        messages = [
            J2_2AirTrack(track_number=i, latitude_deg=float(i))
            for i in range(10)
        ]
        encoded = [encode_message(m) for m in messages]
        decoded = [decode_message(d) for d in encoded]
        for i, d in enumerate(decoded):
            assert d.track_number == i
