"""Unit tests for STANAG field validation."""

from __future__ import annotations

import pytest

from sentinel.core.types import L16Identity
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.validator import L16Validator, ValidationResult


class TestValidateJ2_2:
    def test_all_valid(self):
        msg = J2_2AirTrack(
            track_number=100,
            latitude_deg=37.0,
            longitude_deg=-122.0,
            altitude_ft=10000,
            speed_knots=480,
            course_deg=270.0,
            track_quality=5,
            strength=3,
        )
        results = L16Validator.validate_j2_2(msg)
        assert all(r.valid for r in results) or len(results) == 0

    def test_track_number_out_of_range(self):
        msg = J2_2AirTrack(track_number=9000)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "track_number" for r in invalid)

    def test_latitude_out_of_range(self):
        msg = J2_2AirTrack(latitude_deg=91.0)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "latitude_deg" for r in invalid)

    def test_longitude_out_of_range(self):
        msg = J2_2AirTrack(longitude_deg=-181.0)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "longitude_deg" for r in invalid)

    def test_altitude_negative(self):
        msg = J2_2AirTrack(altitude_ft=-100)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "altitude_ft" for r in invalid)

    def test_speed_out_of_range(self):
        msg = J2_2AirTrack(speed_knots=2000)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "speed_knots" for r in invalid)

    def test_course_out_of_range(self):
        msg = J2_2AirTrack(course_deg=361.0)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "course_deg" for r in invalid)

    def test_track_quality_out_of_range(self):
        msg = J2_2AirTrack(track_quality=10)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "track_quality" for r in invalid)

    def test_iff_no_code_not_validated(self):
        """iff_mode_3a=-1 (no code) should not be flagged."""
        msg = J2_2AirTrack(iff_mode_3a=-1)
        results = L16Validator.validate_j2_2(msg)
        invalid = [r for r in results if not r.valid]
        assert not any(r.field_name == "iff_mode_3a" for r in invalid)


class TestValidateJ3_2:
    def test_valid(self):
        msg = J3_2TrackManagement(track_number=100, action=1)
        results = L16Validator.validate_j3_2(msg)
        assert all(r.valid for r in results) or len(results) == 0

    def test_invalid_action(self):
        msg = J3_2TrackManagement(action=10)
        results = L16Validator.validate_j3_2(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "action" for r in invalid)


class TestValidateJ3_5:
    def test_valid(self):
        msg = J3_5EngagementStatus(track_number=100, engagement_auth=2)
        results = L16Validator.validate_j3_5(msg)
        assert all(r.valid for r in results) or len(results) == 0


class TestValidateJ7_0:
    def test_valid_all_modes(self):
        msg = J7_0IFF(track_number=100, mode_1=42, mode_3a=1200, mode_s_address=0xABCDEF)
        results = L16Validator.validate_j7_0(msg)
        assert all(r.valid for r in results) or len(results) == 0

    def test_mode_3a_out_of_range(self):
        msg = J7_0IFF(mode_3a=5000)
        results = L16Validator.validate_j7_0(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "mode_3a" for r in invalid)

    def test_mode_s_out_of_range(self):
        msg = J7_0IFF(mode_s_address=0x1FFFFFF)
        results = L16Validator.validate_j7_0(msg)
        invalid = [r for r in results if not r.valid]
        assert any(r.field_name == "mode_s_address" for r in invalid)


class TestAutoDispatch:
    def test_dispatch_j2_2(self):
        msg = J2_2AirTrack(track_number=100)
        results = L16Validator.validate(msg)
        assert isinstance(results, list)

    def test_dispatch_j3_2(self):
        msg = J3_2TrackManagement()
        results = L16Validator.validate(msg)
        assert isinstance(results, list)

    def test_dispatch_j3_5(self):
        msg = J3_5EngagementStatus()
        results = L16Validator.validate(msg)
        assert isinstance(results, list)

    def test_dispatch_j7_0(self):
        msg = J7_0IFF()
        results = L16Validator.validate(msg)
        assert isinstance(results, list)

    def test_dispatch_unknown_type(self):
        results = L16Validator.validate("not a message")
        assert len(results) == 1
        assert not results[0].valid
