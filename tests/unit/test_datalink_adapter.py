"""Tests for the DataLinkAdapter — bidirectional SENTINEL <-> Link 16 conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from sentinel.core.types import L16Identity
from sentinel.datalink.adapter import (
    DataLinkAdapter,
    _IFF_TO_L16,
    _L16_TO_IFF,
    _METERS_TO_FEET,
    _MPS_TO_KNOTS,
)
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.track_mapping import TrackNumberAllocator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTrack:
    """Minimal stand-in for EnhancedFusedTrack."""

    fused_id: str = "abcd1234"
    track_id: str = ""
    position_geo: Any = None
    position_m: Any = None
    position: Any = None
    velocity: Any = None
    velocity_mps: Any = None
    iff_identification: str = "unknown"
    iff_mode_3a_code: str | None = None
    sensor_count: int = 2
    confidence: float | None = None
    fusion_quality: float | None = None
    threat_level: str = "LOW"
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False
    is_decoy_candidate: bool = False
    is_chaff_candidate: bool = False


class FakeGeoContext:
    """Minimal geo context for ENU <-> geodetic conversion."""

    def __init__(self, lat0: float = 40.0, lon0: float = -74.0, alt0: float = 0.0):
        self._lat0 = lat0
        self._lon0 = lon0
        self._alt0 = alt0

    def enu_to_geodetic(self, e: float, n: float, u: float):
        # Simplified: small offset in degrees
        lat = self._lat0 + n / 111320.0
        lon = self._lon0 + e / (111320.0 * math.cos(math.radians(self._lat0)))
        alt = self._alt0 + u
        return lat, lon, alt

    def geodetic_to_enu(self, lat: float, lon: float, alt: float):
        e = (lon - self._lon0) * 111320.0 * math.cos(math.radians(self._lat0))
        n = (lat - self._lat0) * 111320.0
        u = alt - self._alt0
        return e, n, u


# ===========================================================================
# Outbound: track_to_j2_2
# ===========================================================================


class TestTrackToJ2_2:
    def test_basic_conversion_with_geodetic_position(self):
        track = FakeTrack(position_geo=(38.0, -76.5, 3000.0))
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 1000.0)

        assert msg is not None
        assert isinstance(msg, J2_2AirTrack)
        assert msg.latitude_deg == pytest.approx(38.0)
        assert msg.longitude_deg == pytest.approx(-76.5)
        assert msg.altitude_ft == int(3000.0 * _METERS_TO_FEET)
        assert msg.timestamp == 1000.0

    def test_geodetic_from_dict(self):
        track = FakeTrack(position_geo={"lat": 51.5, "lon": -0.1, "alt": 300})
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg is not None
        assert msg.latitude_deg == pytest.approx(51.5)
        assert msg.longitude_deg == pytest.approx(-0.1)

    def test_geodetic_from_dict_long_keys(self):
        track = FakeTrack(position_geo={"latitude": 35.0, "longitude": 139.0, "altitude": 100})
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg is not None
        assert msg.latitude_deg == pytest.approx(35.0)

    def test_enu_conversion_with_geo_context(self):
        geo = FakeGeoContext(lat0=40.0, lon0=-74.0)
        track = FakeTrack(position_m=np.array([100.0, 200.0, 500.0]))
        adapter = DataLinkAdapter(geo_context=geo)
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg is not None
        assert msg.latitude_deg == pytest.approx(40.0 + 200.0 / 111320.0, abs=0.001)

    def test_returns_none_without_position(self):
        track = FakeTrack()  # no position
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg is None

    def test_velocity_to_speed_course(self):
        # 100 m/s east (vx=100, vy=0) → ~194 knots, course=90°
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            velocity=np.array([100.0, 0.0]),
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg is not None
        assert msg.speed_knots == int(100.0 * _MPS_TO_KNOTS)
        assert msg.course_deg == pytest.approx(90.0, abs=0.1)

    def test_velocity_north(self):
        # 50 m/s north (vx=0, vy=50) → course=0°
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            velocity=np.array([0.0, 50.0]),
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.course_deg == pytest.approx(0.0, abs=0.1)

    def test_identity_mapping_all(self):
        adapter = DataLinkAdapter()
        for iff_str, expected_l16 in _IFF_TO_L16.items():
            track = FakeTrack(
                position_geo=(0.0, 0.0, 0.0),
                iff_identification=iff_str,
            )
            msg = adapter.track_to_j2_2(track, 0.0)
            assert msg.identity == expected_l16, f"Failed for {iff_str}"

    def test_track_quality_computation(self):
        # sensor_count=3 → +3, iff="friendly" → +1, confidence=0.8 → +1 = 5
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            sensor_count=3,
            iff_identification="friendly",
            confidence=0.8,
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.track_quality == 5

    def test_track_quality_caps_at_7(self):
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            sensor_count=10,
            iff_identification="hostile",
            confidence=0.99,
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.track_quality <= 7

    def test_strength_from_sensor_count(self):
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            sensor_count=5,
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.strength == 5

    def test_strength_capped_at_7(self):
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            sensor_count=15,
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.strength == 7

    def test_environment_flags(self):
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            is_stealth_candidate=True,
            is_hypersonic_candidate=True,
            is_decoy_candidate=False,
            is_chaff_candidate=True,
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.environment_flags == 0b1101

    def test_threat_level_mapping(self):
        for threat, expected_int in [("LOW", 0), ("MEDIUM", 1), ("HIGH", 2), ("CRITICAL", 3)]:
            track = FakeTrack(position_geo=(0.0, 0.0, 0.0), threat_level=threat)
            adapter = DataLinkAdapter()
            msg = adapter.track_to_j2_2(track, 0.0)
            assert msg.threat_level == expected_int

    def test_iff_mode_3a_octal_string(self):
        track = FakeTrack(
            position_geo=(0.0, 0.0, 0.0),
            iff_mode_3a_code="1234",  # octal string
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.iff_mode_3a == 0o1234

    def test_iff_mode_3a_absent(self):
        track = FakeTrack(position_geo=(0.0, 0.0, 0.0))
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.iff_mode_3a == -1

    def test_track_number_allocated(self):
        alloc = TrackNumberAllocator()
        adapter = DataLinkAdapter(track_allocator=alloc)
        track = FakeTrack(position_geo=(0.0, 0.0, 0.0), fused_id="abcd1234")
        msg = adapter.track_to_j2_2(track, 0.0)
        assert 0 <= msg.track_number < 8192
        assert alloc.get_sentinel_id(msg.track_number) == "abcd1234"

    def test_altitude_clamped(self):
        # Very high altitude → clamped to 16383 ft
        track = FakeTrack(position_geo=(0.0, 0.0, 100000.0))
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j2_2(track, 0.0)
        assert msg.altitude_ft <= 16383


# ===========================================================================
# Outbound: track_to_j3_2
# ===========================================================================


class TestTrackToJ3_2:
    def test_basic_j3_2(self):
        track = FakeTrack(
            fused_id="dead0001",
            position_geo=(45.0, -90.0, 1000.0),
            iff_identification="hostile",
        )
        adapter = DataLinkAdapter()
        msg = adapter.track_to_j3_2(track, action=1, timestamp=500.0)
        assert msg is not None
        assert msg.action == 1
        assert msg.identity == L16Identity.HOSTILE
        assert msg.timestamp == 500.0


# ===========================================================================
# Outbound: engagement_to_j3_5
# ===========================================================================


class TestEngagementToJ3_5:
    def test_basic_j3_5(self):
        adapter = DataLinkAdapter()
        msg = adapter.engagement_to_j3_5("trk-001", "weapons_free", 2, 1, 100.0)
        assert msg is not None
        assert msg.engagement_auth == 0
        assert msg.weapon_type == 2
        assert msg.engagement_status == 1

    def test_unknown_auth_defaults(self):
        adapter = DataLinkAdapter()
        msg = adapter.engagement_to_j3_5("trk-002", "unknown_auth", 0, 0, 0.0)
        # Should default to weapons_hold (2)
        assert msg.engagement_auth == 2


# ===========================================================================
# Outbound: iff_result_to_j7_0
# ===========================================================================


class TestIffResultToJ7_0:
    def test_basic_j7_0(self):
        adapter = DataLinkAdapter()
        iff = {
            "identification": "friendly",
            "mode_3a_code": "1200",
            "mode_s_address": "A1B2C3",
            "mode_4_valid": True,
            "mode_5_valid": False,
        }
        msg = adapter.iff_result_to_j7_0("trk-003", iff, 200.0)
        assert msg is not None
        assert msg.identity == L16Identity.FRIEND
        assert msg.mode_3a == 0o1200
        assert msg.mode_s_address == 0xA1B2C3
        assert msg.mode_4_valid is True
        assert msg.mode_5_valid is False

    def test_missing_codes(self):
        adapter = DataLinkAdapter()
        iff = {"identification": "unknown"}
        msg = adapter.iff_result_to_j7_0("trk-004", iff, 0.0)
        assert msg.mode_3a == -1
        assert msg.mode_s_address == -1


# ===========================================================================
# Inbound: j2_2_to_remote_track
# ===========================================================================


class TestJ2_2ToRemoteTrack:
    def test_basic_inbound_conversion(self):
        adapter = DataLinkAdapter()
        msg = J2_2AirTrack(
            track_number=100,
            identity=L16Identity.HOSTILE,
            latitude_deg=38.9,
            longitude_deg=-77.0,
            altitude_ft=25000,
            speed_knots=450,
            course_deg=90.0,
            track_quality=5,
            threat_level=2,
        )
        result = adapter.j2_2_to_remote_track(msg)
        assert result["track_id"] == "L16-100"
        assert result["iff_identification"] == "hostile"
        assert result["threat_level"] == "HIGH"
        assert result["confidence"] == pytest.approx(5.0 / 7.0)
        assert result["position_geo"] is not None

    def test_velocity_vector_from_speed_course(self):
        adapter = DataLinkAdapter()
        msg = J2_2AirTrack(
            track_number=1,
            identity=L16Identity.UNKNOWN,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_ft=0,
            speed_knots=100,
            course_deg=90.0,  # Due east
        )
        result = adapter.j2_2_to_remote_track(msg)
        vel = result["velocity"]
        # vx should be ~positive (east), vy should be ~0
        assert vel[0] > 0
        assert abs(vel[1]) < 1.0

    def test_enu_position_with_geo_context(self):
        geo = FakeGeoContext(lat0=38.9, lon0=-77.0)
        adapter = DataLinkAdapter(geo_context=geo)
        msg = J2_2AirTrack(
            track_number=50,
            identity=L16Identity.FRIEND,
            latitude_deg=38.9,
            longitude_deg=-77.0,
            altitude_ft=1000,
        )
        result = adapter.j2_2_to_remote_track(msg)
        assert result["position"] is not None
        # Near origin
        assert abs(result["position"][0]) < 50
        assert abs(result["position"][1]) < 50

    def test_known_sentinel_id_returned(self):
        alloc = TrackNumberAllocator()
        tn = alloc.get_or_allocate("mytrack01")
        adapter = DataLinkAdapter(track_allocator=alloc)
        msg = J2_2AirTrack(
            track_number=tn,
            identity=L16Identity.UNKNOWN,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_ft=0,
        )
        result = adapter.j2_2_to_remote_track(msg)
        assert result["track_id"] == "mytrack01"


# ===========================================================================
# Inbound: j7_0_to_iff_result, j3_5_to_engagement
# ===========================================================================


class TestInboundJ7_0:
    def test_basic(self):
        adapter = DataLinkAdapter()
        msg = J7_0IFF(
            track_number=10,
            identity=L16Identity.FRIEND,
            mode_3a=0o1200,
            mode_s_address=0xABCDEF,
            mode_4_valid=True,
            mode_5_valid=True,
        )
        result = adapter.j7_0_to_iff_result(msg)
        assert result["identification"] == "friendly"
        assert result["mode_3a_code"] == oct(0o1200)
        assert result["mode_s_address"] == hex(0xABCDEF)
        assert result["mode_4_valid"] is True

    def test_no_code_fields(self):
        adapter = DataLinkAdapter()
        msg = J7_0IFF(track_number=10, identity=L16Identity.UNKNOWN)
        result = adapter.j7_0_to_iff_result(msg)
        assert result["mode_3a_code"] is None
        assert result["mode_s_address"] is None


class TestInboundJ3_5:
    def test_basic(self):
        alloc = TrackNumberAllocator()
        tn = alloc.get_or_allocate("eng-track")
        adapter = DataLinkAdapter(track_allocator=alloc)
        msg = J3_5EngagementStatus(
            track_number=tn,
            engagement_auth=0,
            weapon_type=3,
            engagement_status=1,
        )
        result = adapter.j3_5_to_engagement(msg)
        assert result["track_id"] == "eng-track"
        assert result["engagement_auth"] == "weapons_free"
        assert result["weapon_type"] == 3


# ===========================================================================
# Identity mapping completeness
# ===========================================================================


class TestIdentityMappingCompleteness:
    def test_all_l16_identities_mapped_inbound(self):
        """Every L16Identity has an inbound mapping."""
        for ident in L16Identity:
            assert ident in _L16_TO_IFF

    def test_roundtrip_common_identities(self):
        """Common identities survive outbound+inbound roundtrip."""
        roundtrip_cases = [
            ("friendly", "friendly"),
            ("hostile", "hostile"),
            ("unknown", "unknown"),
            ("pending", "pending"),
            ("assumed_friendly", "assumed_friendly"),
            ("assumed_hostile", "assumed_hostile"),
        ]
        for iff_in, expected_iff_out in roundtrip_cases:
            l16 = _IFF_TO_L16[iff_in]
            iff_out = _L16_TO_IFF[l16]
            assert iff_out == expected_iff_out, f"Roundtrip failed: {iff_in} -> {l16} -> {iff_out}"
