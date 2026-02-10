"""STANAG 5516 field range validation for Link 16 J-series messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)


@dataclass(frozen=True)
class ValidationResult:
    """Single field validation outcome."""

    valid: bool
    field_name: str
    message: str = ""


class L16Validator:
    """Validates J-series message fields against STANAG ranges."""

    TRACK_NUMBER_MAX = 8191
    LATITUDE_MIN, LATITUDE_MAX = -90.0, 90.0
    LONGITUDE_MIN, LONGITUDE_MAX = -180.0, 180.0
    ALTITUDE_FT_MIN, ALTITUDE_FT_MAX = 0, 16383
    SPEED_KNOTS_MAX = 1023
    COURSE_DEG_MIN, COURSE_DEG_MAX = 0.0, 360.0
    TRACK_QUALITY_MAX = 7
    STRENGTH_MAX = 7
    MODE_3A_MAX = 4095
    MODE_C_MAX = 126000
    MODE_S_MAX = 0xFFFFFF
    ACTION_MAX = 7
    ENGAGEMENT_AUTH_MAX = 7
    WEAPON_TYPE_MAX = 7
    ENGAGEMENT_STATUS_MAX = 7
    MODE_1_MAX = 63
    MODE_2_MAX = 4095
    THREAT_LEVEL_MAX = 3
    ENV_FLAGS_MAX = 15

    # ------------------------------------------------------------------

    @classmethod
    def validate_j2_2(cls, msg: J2_2AirTrack) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        cls._check_range(results, "track_number", msg.track_number, 0, cls.TRACK_NUMBER_MAX)
        cls._check_range(results, "latitude_deg", msg.latitude_deg, cls.LATITUDE_MIN, cls.LATITUDE_MAX)
        cls._check_range(results, "longitude_deg", msg.longitude_deg, cls.LONGITUDE_MIN, cls.LONGITUDE_MAX)
        cls._check_range(results, "altitude_ft", msg.altitude_ft, cls.ALTITUDE_FT_MIN, cls.ALTITUDE_FT_MAX)
        cls._check_range(results, "speed_knots", msg.speed_knots, 0, cls.SPEED_KNOTS_MAX)
        cls._check_range(results, "course_deg", msg.course_deg, cls.COURSE_DEG_MIN, cls.COURSE_DEG_MAX)
        cls._check_range(results, "track_quality", msg.track_quality, 0, cls.TRACK_QUALITY_MAX)
        cls._check_range(results, "strength", msg.strength, 0, cls.STRENGTH_MAX)
        if msg.iff_mode_3a >= 0:
            cls._check_range(results, "iff_mode_3a", msg.iff_mode_3a, 0, cls.MODE_3A_MAX)
        cls._check_range(results, "threat_level", msg.threat_level, 0, cls.THREAT_LEVEL_MAX)
        cls._check_range(results, "environment_flags", msg.environment_flags, 0, cls.ENV_FLAGS_MAX)
        return results

    @classmethod
    def validate_j3_2(cls, msg: J3_2TrackManagement) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        cls._check_range(results, "track_number", msg.track_number, 0, cls.TRACK_NUMBER_MAX)
        cls._check_range(results, "action", msg.action, 0, cls.ACTION_MAX)
        cls._check_range(results, "track_quality", msg.track_quality, 0, cls.TRACK_QUALITY_MAX)
        cls._check_range(results, "latitude_deg", msg.latitude_deg, cls.LATITUDE_MIN, cls.LATITUDE_MAX)
        cls._check_range(results, "longitude_deg", msg.longitude_deg, cls.LONGITUDE_MIN, cls.LONGITUDE_MAX)
        return results

    @classmethod
    def validate_j3_5(cls, msg: J3_5EngagementStatus) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        cls._check_range(results, "track_number", msg.track_number, 0, cls.TRACK_NUMBER_MAX)
        cls._check_range(results, "engagement_auth", msg.engagement_auth, 0, cls.ENGAGEMENT_AUTH_MAX)
        cls._check_range(results, "weapon_type", msg.weapon_type, 0, cls.WEAPON_TYPE_MAX)
        cls._check_range(results, "engagement_status", msg.engagement_status, 0, cls.ENGAGEMENT_STATUS_MAX)
        return results

    @classmethod
    def validate_j7_0(cls, msg: J7_0IFF) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        cls._check_range(results, "track_number", msg.track_number, 0, cls.TRACK_NUMBER_MAX)
        if msg.mode_1 >= 0:
            cls._check_range(results, "mode_1", msg.mode_1, 0, cls.MODE_1_MAX)
        if msg.mode_2 >= 0:
            cls._check_range(results, "mode_2", msg.mode_2, 0, cls.MODE_2_MAX)
        if msg.mode_3a >= 0:
            cls._check_range(results, "mode_3a", msg.mode_3a, 0, cls.MODE_3A_MAX)
        if msg.mode_c_alt_ft >= 0:
            cls._check_range(results, "mode_c_alt_ft", msg.mode_c_alt_ft, 0, cls.MODE_C_MAX)
        if msg.mode_s_address >= 0:
            cls._check_range(results, "mode_s_address", msg.mode_s_address, 0, cls.MODE_S_MAX)
        return results

    @classmethod
    def validate(cls, msg: Any) -> list[ValidationResult]:
        """Auto-dispatch validation based on message type."""
        if isinstance(msg, J2_2AirTrack):
            return cls.validate_j2_2(msg)
        if isinstance(msg, J3_2TrackManagement):
            return cls.validate_j3_2(msg)
        if isinstance(msg, J3_5EngagementStatus):
            return cls.validate_j3_5(msg)
        if isinstance(msg, J7_0IFF):
            return cls.validate_j7_0(msg)
        return [ValidationResult(valid=False, field_name="type", message=f"Unknown message type: {type(msg)}")]

    # ------------------------------------------------------------------

    @staticmethod
    def _check_range(
        results: list[ValidationResult],
        field: str,
        value: int | float,
        min_val: int | float,
        max_val: int | float,
    ) -> None:
        if value < min_val or value > max_val:
            results.append(ValidationResult(
                valid=False,
                field_name=field,
                message=f"{field}={value} out of range [{min_val}, {max_val}]",
            ))
