"""Unit tests for J-series message dataclasses."""

from __future__ import annotations

import pytest

from sentinel.core.types import L16Identity, L16MessageType
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)


class TestJ2_2AirTrack:
    def test_defaults(self):
        msg = J2_2AirTrack()
        assert msg.message_type == L16MessageType.J2_2
        assert msg.track_number == 0
        assert msg.identity == L16Identity.UNKNOWN
        assert msg.latitude_deg == 0.0
        assert msg.longitude_deg == 0.0
        assert msg.altitude_ft == 0
        assert msg.speed_knots == 0
        assert msg.course_deg == 0.0
        assert msg.track_quality == 0
        assert msg.strength == 0
        assert msg.iff_mode_3a == -1
        assert msg.threat_level == 0
        assert msg.environment_flags == 0
        assert msg.source_sentinel_id == ""
        assert msg.timestamp == 0.0

    def test_with_all_fields(self):
        msg = J2_2AirTrack(
            track_number=4567,
            identity=L16Identity.HOSTILE,
            latitude_deg=37.7749,
            longitude_deg=-122.4194,
            altitude_ft=35000,
            speed_knots=480,
            course_deg=270.0,
            track_quality=7,
            strength=3,
            iff_mode_3a=1234,
            threat_level=3,
            environment_flags=0b1010,
            source_sentinel_id="a1b2c3d4",
            timestamp=1000.0,
        )
        assert msg.track_number == 4567
        assert msg.identity == L16Identity.HOSTILE
        assert msg.altitude_ft == 35000

    def test_frozen(self):
        msg = J2_2AirTrack()
        with pytest.raises(AttributeError):
            msg.track_number = 1  # type: ignore[misc]


class TestJ3_2TrackManagement:
    def test_defaults(self):
        msg = J3_2TrackManagement()
        assert msg.message_type == L16MessageType.J3_2
        assert msg.action == 0
        assert msg.track_quality == 0

    def test_drop_action(self):
        msg = J3_2TrackManagement(track_number=100, action=1)
        assert msg.action == 1

    def test_frozen(self):
        msg = J3_2TrackManagement()
        with pytest.raises(AttributeError):
            msg.action = 2  # type: ignore[misc]


class TestJ3_5EngagementStatus:
    def test_defaults(self):
        msg = J3_5EngagementStatus()
        assert msg.message_type == L16MessageType.J3_5
        assert msg.engagement_auth == 0
        assert msg.weapon_type == 0
        assert msg.engagement_status == 0

    def test_with_fields(self):
        msg = J3_5EngagementStatus(
            track_number=500,
            engagement_auth=2,
            weapon_type=3,
            engagement_status=1,
        )
        assert msg.track_number == 500
        assert msg.engagement_auth == 2

    def test_frozen(self):
        msg = J3_5EngagementStatus()
        with pytest.raises(AttributeError):
            msg.engagement_auth = 1  # type: ignore[misc]


class TestJ7_0IFF:
    def test_defaults(self):
        msg = J7_0IFF()
        assert msg.message_type == L16MessageType.J7_0
        assert msg.mode_1 == -1
        assert msg.mode_2 == -1
        assert msg.mode_3a == -1
        assert msg.mode_c_alt_ft == -1
        assert msg.mode_s_address == -1
        assert msg.mode_4_valid is False
        assert msg.mode_5_valid is False

    def test_all_modes(self):
        msg = J7_0IFF(
            track_number=200,
            identity=L16Identity.FRIEND,
            mode_1=42,
            mode_2=3000,
            mode_3a=1200,
            mode_c_alt_ft=35000,
            mode_s_address=0xABCDEF,
            mode_4_valid=True,
            mode_5_valid=True,
        )
        assert msg.mode_1 == 42
        assert msg.mode_s_address == 0xABCDEF
        assert msg.mode_4_valid is True

    def test_frozen(self):
        msg = J7_0IFF()
        with pytest.raises(AttributeError):
            msg.mode_1 = 0  # type: ignore[misc]


class TestL16Enums:
    def test_identity_values(self):
        assert len(L16Identity) == 8
        assert L16Identity.PENDING.value == "pending"
        assert L16Identity.HOSTILE.value == "hostile"

    def test_message_type_values(self):
        assert len(L16MessageType) == 4
        assert L16MessageType.J2_2.value == "J2.2"
        assert L16MessageType.J7_0.value == "J7.0"
