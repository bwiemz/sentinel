"""Unit tests for weapon profiles and WEZ calculator."""

import math

import numpy as np
import pytest

from sentinel.core.types import WeaponType
from sentinel.engagement.weapons import WEZCalculator, WEZResult, WeaponProfile


# ===================================================================
# WeaponProfile
# ===================================================================


class TestWeaponProfile:
    def test_from_config_minimal(self):
        cfg = {"weapon_id": "W1", "name": "Test SAM"}
        wp = WeaponProfile.from_config(cfg)
        assert wp.weapon_id == "W1"
        assert wp.name == "Test SAM"
        assert wp.weapon_type == WeaponType.SAM_MEDIUM  # default

    def test_from_config_all_fields(self):
        cfg = {
            "weapon_id": "W2",
            "name": "Long Range SAM",
            "weapon_type": "sam_long",
            "position_xy": [1000.0, 2000.0],
            "altitude_m": 50.0,
            "min_range_m": 1000.0,
            "max_range_m": 80000.0,
            "optimal_range_m": 40000.0,
            "min_altitude_m": 200.0,
            "max_altitude_m": 30000.0,
            "max_target_speed_mps": 1500.0,
            "weapon_speed_mps": 2000.0,
            "min_aspect_angle_deg": 0.0,
            "max_aspect_angle_deg": 160.0,
            "max_simultaneous_engagements": 4,
            "rounds_remaining": 24,
            "salvo_size": 2,
            "reload_time_s": 8.0,
        }
        wp = WeaponProfile.from_config(cfg)
        assert wp.weapon_type == WeaponType.SAM_LONG
        assert wp.max_range_m == 80000.0
        assert wp.max_simultaneous_engagements == 4
        assert wp.salvo_size == 2
        np.testing.assert_allclose(wp.position_xy, [1000.0, 2000.0])

    def test_from_config_invalid_weapon_type_defaults(self):
        cfg = {"weapon_id": "W3", "weapon_type": "invalid_type"}
        wp = WeaponProfile.from_config(cfg)
        assert wp.weapon_type == WeaponType.SAM_MEDIUM

    def test_from_config_weapon_id_fallback(self):
        cfg = {"name": "No ID"}
        wp = WeaponProfile.from_config(cfg, weapon_id="FALLBACK")
        assert wp.weapon_id == "FALLBACK"

    def test_to_dict(self):
        wp = WeaponProfile(
            weapon_id="W1",
            name="Test",
            weapon_type=WeaponType.CIWS,
            position_xy=np.array([100.0, 200.0]),
        )
        d = wp.to_dict()
        assert d["weapon_id"] == "W1"
        assert d["weapon_type"] == "ciws"
        assert d["position_xy"] == [100.0, 200.0]
        assert "max_range_m" in d
        assert "rounds_remaining" in d

    def test_from_config_to_dict_roundtrip(self):
        cfg = {
            "weapon_id": "RT1",
            "name": "Roundtrip",
            "weapon_type": "aam_short",
            "position_xy": [500.0, 600.0],
            "altitude_m": 10.0,
            "min_range_m": 200.0,
            "max_range_m": 5000.0,
            "min_altitude_m": 50.0,
            "max_altitude_m": 10000.0,
            "max_simultaneous_engagements": 1,
            "rounds_remaining": 4,
        }
        wp = WeaponProfile.from_config(cfg)
        d = wp.to_dict()
        assert d["weapon_id"] == "RT1"
        assert d["weapon_type"] == "aam_short"
        assert d["min_range_m"] == 200.0
        assert d["max_range_m"] == 5000.0

    def test_defaults(self):
        wp = WeaponProfile(
            weapon_id="DEF",
            name="Default",
            weapon_type=WeaponType.GUN,
            position_xy=np.array([0.0, 0.0]),
        )
        assert wp.min_range_m == 500.0
        assert wp.max_range_m == 20000.0
        assert wp.max_target_speed_mps == 800.0
        assert wp.pk_base == 0.85

    def test_equality_by_weapon_id(self):
        w1 = WeaponProfile(
            weapon_id="SAME", name="A",
            weapon_type=WeaponType.GUN,
            position_xy=np.array([0.0, 0.0]),
        )
        w2 = WeaponProfile(
            weapon_id="SAME", name="B",
            weapon_type=WeaponType.CIWS,
            position_xy=np.array([999.0, 999.0]),
        )
        assert w1 == w2


# ===================================================================
# WEZCalculator
# ===================================================================


class TestWEZCalculator:
    def _weapon(self, **kwargs):
        defaults = dict(
            weapon_id="WPN1",
            name="Test SAM",
            weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
            altitude_m=0.0,
            min_range_m=500.0,
            max_range_m=20000.0,
            min_altitude_m=100.0,
            max_altitude_m=20000.0,
            max_target_speed_mps=800.0,
            min_aspect_angle_deg=0.0,
            max_aspect_angle_deg=180.0,
        )
        defaults.update(kwargs)
        return WeaponProfile(**defaults)

    def test_target_in_range(self):
        calc = WEZCalculator()
        w = self._weapon()
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 5000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T1",
        )
        assert result.in_range is True

    def test_target_too_close(self):
        calc = WEZCalculator()
        w = self._weapon(min_range_m=1000.0)
        # Target at (100, 0) with altitude 500 => slant range ~ 510 < 1000
        result = calc.evaluate(
            w,
            track_position=np.array([100.0, 0.0, 500.0]),
            track_velocity=np.array([-50.0, 0.0]),
            track_id="T2",
        )
        assert result.in_range is False
        assert result.feasible is False

    def test_target_too_far(self):
        calc = WEZCalculator()
        w = self._weapon(max_range_m=10000.0)
        result = calc.evaluate(
            w,
            track_position=np.array([50000.0, 0.0, 5000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T3",
        )
        assert result.in_range is False
        assert result.feasible is False

    def test_altitude_within_envelope(self):
        calc = WEZCalculator()
        w = self._weapon(min_altitude_m=100.0, max_altitude_m=15000.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 10000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T4",
        )
        assert result.in_altitude is True

    def test_altitude_too_low(self):
        calc = WEZCalculator()
        w = self._weapon(min_altitude_m=500.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 100.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T5",
        )
        assert result.in_altitude is False

    def test_altitude_too_high(self):
        calc = WEZCalculator()
        w = self._weapon(max_altitude_m=10000.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 15000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T6",
        )
        assert result.in_altitude is False

    def test_speed_within_limit(self):
        calc = WEZCalculator()
        w = self._weapon(max_target_speed_mps=1000.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 5000.0]),
            track_velocity=np.array([-300.0, 400.0]),  # speed = 500
            track_id="T7",
        )
        assert result.in_speed is True
        assert result.target_speed_mps == pytest.approx(500.0)

    def test_speed_exceeds_limit(self):
        calc = WEZCalculator()
        w = self._weapon(max_target_speed_mps=200.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 5000.0]),
            track_velocity=np.array([-300.0, 400.0]),  # speed = 500
            track_id="T8",
        )
        assert result.in_speed is False

    def test_aspect_angle_head_on(self):
        """Target flying directly toward weapon: aspect = 0 degrees."""
        calc = WEZCalculator()
        w = self._weapon()
        # Target at (10000, 0), velocity (-200, 0) -> flying toward weapon
        result = calc.evaluate(
            w,
            track_position=np.array([10000.0, 0.0, 5000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="T9",
        )
        assert result.aspect_angle_deg == pytest.approx(0.0, abs=1.0)

    def test_aspect_angle_tail_on(self):
        """Target flying directly away from weapon: aspect = 180 degrees."""
        calc = WEZCalculator()
        w = self._weapon()
        # Target at (10000, 0), velocity (200, 0) -> flying away
        result = calc.evaluate(
            w,
            track_position=np.array([10000.0, 0.0, 5000.0]),
            track_velocity=np.array([200.0, 0.0]),
            track_id="T10",
        )
        assert result.aspect_angle_deg == pytest.approx(180.0, abs=1.0)

    def test_aspect_angle_broadside(self):
        """Target flying perpendicular to LOS: aspect = 90 degrees."""
        calc = WEZCalculator()
        w = self._weapon()
        # Target at (10000, 0), velocity (0, 200) -> perpendicular
        result = calc.evaluate(
            w,
            track_position=np.array([10000.0, 0.0, 5000.0]),
            track_velocity=np.array([0.0, 200.0]),
            track_id="T11",
        )
        assert result.aspect_angle_deg == pytest.approx(90.0, abs=1.0)

    def test_slant_range_3d(self):
        """Slant range includes altitude difference."""
        calc = WEZCalculator()
        sr = calc.compute_slant_range(
            weapon_pos=np.array([0.0, 0.0]),
            weapon_alt=0.0,
            target_pos=np.array([3000.0, 4000.0]),
            target_alt=0.0,
        )
        # 2D distance = 5000
        assert sr == pytest.approx(5000.0)

    def test_slant_range_with_altitude(self):
        calc = WEZCalculator()
        sr = calc.compute_slant_range(
            weapon_pos=np.array([0.0, 0.0]),
            weapon_alt=0.0,
            target_pos=np.array([3000.0, 4000.0]),
            target_alt=5000.0,
        )
        # sqrt(3000^2 + 4000^2 + 5000^2) = sqrt(50000000) ~ 7071
        expected = math.sqrt(3000**2 + 4000**2 + 5000**2)
        assert sr == pytest.approx(expected)

    def test_closing_speed_approaching(self):
        """Target flying toward weapon yields positive closing speed."""
        calc = WEZCalculator()
        cs = calc.compute_closing_speed(
            weapon_pos=np.array([0.0, 0.0]),
            target_pos=np.array([10000.0, 0.0]),
            target_velocity=np.array([-300.0, 0.0]),
        )
        assert cs == pytest.approx(300.0)

    def test_closing_speed_receding(self):
        """Target flying away from weapon yields negative closing speed."""
        calc = WEZCalculator()
        cs = calc.compute_closing_speed(
            weapon_pos=np.array([0.0, 0.0]),
            target_pos=np.array([10000.0, 0.0]),
            target_velocity=np.array([300.0, 0.0]),
        )
        assert cs == pytest.approx(-300.0)

    def test_closing_speed_perpendicular(self):
        """Target flying perpendicular to LOS yields zero closing speed."""
        calc = WEZCalculator()
        cs = calc.compute_closing_speed(
            weapon_pos=np.array([0.0, 0.0]),
            target_pos=np.array([10000.0, 0.0]),
            target_velocity=np.array([0.0, 300.0]),
        )
        assert cs == pytest.approx(0.0, abs=1e-6)

    def test_stationary_target_aspect(self):
        """Stationary target yields 90-degree aspect (broadside default)."""
        calc = WEZCalculator()
        aspect = calc.compute_aspect_angle(
            weapon_pos=np.array([0.0, 0.0]),
            target_pos=np.array([5000.0, 0.0]),
            target_velocity=np.array([0.0, 0.0]),
        )
        assert aspect == pytest.approx(90.0)

    def test_all_checks_pass(self):
        calc = WEZCalculator()
        w = self._weapon(
            min_range_m=500.0,
            max_range_m=20000.0,
            min_altitude_m=100.0,
            max_altitude_m=20000.0,
            max_target_speed_mps=800.0,
        )
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 5000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="PASS",
        )
        assert result.in_range is True
        assert result.in_altitude is True
        assert result.in_speed is True
        assert result.in_aspect is True
        assert result.feasible is True

    def test_all_checks_fail(self):
        calc = WEZCalculator()
        w = self._weapon(
            min_range_m=50000.0,   # target too close
            max_range_m=100000.0,
            min_altitude_m=20000.0,  # target too low
            max_altitude_m=30000.0,
            max_target_speed_mps=10.0,  # target too fast
            min_aspect_angle_deg=170.0,  # aspect will be ~0 (head-on)
            max_aspect_angle_deg=180.0,
        )
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0, 5000.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="FAIL",
        )
        assert result.in_range is False
        assert result.in_altitude is False
        assert result.in_speed is False
        assert result.in_aspect is False
        assert result.feasible is False

    def test_wez_result_to_dict(self):
        r = WEZResult(
            weapon_id="W1",
            track_id="T1",
            feasible=True,
            in_range=True,
            in_altitude=True,
            in_speed=True,
            in_aspect=True,
            slant_range_m=12345.6789,
            target_altitude_m=5000.0,
            target_speed_mps=300.123,
            aspect_angle_deg=45.678,
            closing_speed_mps=250.999,
        )
        d = r.to_dict()
        assert d["weapon_id"] == "W1"
        assert d["track_id"] == "T1"
        assert d["feasible"] is True
        assert d["slant_range_m"] == 12345.7  # rounded to 1 decimal
        assert d["aspect_angle_deg"] == 45.7

    def test_2d_position_defaults_altitude_zero(self):
        """2D track position should default altitude to 0."""
        calc = WEZCalculator()
        w = self._weapon(min_altitude_m=0.0, max_altitude_m=100.0)
        result = calc.evaluate(
            w,
            track_position=np.array([5000.0, 0.0]),
            track_velocity=np.array([-200.0, 0.0]),
            track_id="2D",
        )
        assert result.target_altitude_m == 0.0
        assert result.in_altitude is True
