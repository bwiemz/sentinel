"""Tests for coordinate transform utilities."""

import numpy as np
import pytest

from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    normalize_angle,
    polar_to_cartesian,
)


class TestPolarToCartesian:
    def test_along_x_axis(self):
        result = polar_to_cartesian(100.0, 0.0)
        np.testing.assert_allclose(result, [100.0, 0.0], atol=1e-10)

    def test_along_y_axis(self):
        result = polar_to_cartesian(100.0, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 100.0], atol=1e-10)

    def test_negative_x(self):
        result = polar_to_cartesian(50.0, np.pi)
        np.testing.assert_allclose(result, [-50.0, 0.0], atol=1e-10)

    def test_45_degrees(self):
        result = polar_to_cartesian(np.sqrt(2), np.pi / 4)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-10)

    def test_zero_range(self):
        result = polar_to_cartesian(0.0, 1.5)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)


class TestCartesianToPolar:
    def test_along_x_axis(self):
        r, az = cartesian_to_polar(100.0, 0.0)
        assert r == pytest.approx(100.0)
        assert az == pytest.approx(0.0)

    def test_along_y_axis(self):
        r, az = cartesian_to_polar(0.0, 100.0)
        assert r == pytest.approx(100.0)
        assert az == pytest.approx(np.pi / 2)

    def test_negative_x(self):
        r, az = cartesian_to_polar(-50.0, 0.0)
        assert r == pytest.approx(50.0)
        assert abs(az) == pytest.approx(np.pi)

    def test_origin(self):
        r, az = cartesian_to_polar(0.0, 0.0)
        assert r == pytest.approx(0.0)

    def test_known_345(self):
        r, az = cartesian_to_polar(3.0, 4.0)
        assert r == pytest.approx(5.0)
        assert az == pytest.approx(np.arctan2(4.0, 3.0))


class TestRoundTrip:
    @pytest.mark.parametrize("r,az", [
        (100.0, 0.0),
        (50.0, np.pi / 4),
        (200.0, -np.pi / 3),
        (1.0, np.pi / 2),
        (5000.0, -0.1),
    ])
    def test_polar_cartesian_roundtrip(self, r, az):
        xy = polar_to_cartesian(r, az)
        r2, az2 = cartesian_to_polar(xy[0], xy[1])
        assert r2 == pytest.approx(r, abs=1e-10)
        assert az2 == pytest.approx(az, abs=1e-10)


class TestAzimuthConversions:
    def test_deg_to_rad(self):
        assert azimuth_deg_to_rad(180.0) == pytest.approx(np.pi)

    def test_rad_to_deg(self):
        assert azimuth_rad_to_deg(np.pi) == pytest.approx(180.0)

    def test_roundtrip(self):
        assert azimuth_rad_to_deg(azimuth_deg_to_rad(45.0)) == pytest.approx(45.0)


class TestNormalizeAngle:
    def test_already_normalized(self):
        assert normalize_angle(0.5) == pytest.approx(0.5)

    def test_positive_wrap(self):
        # 3*pi normalizes to -pi (equivalent to pi at the boundary)
        assert abs(normalize_angle(3 * np.pi)) == pytest.approx(np.pi, abs=1e-10)

    def test_negative_wrap(self):
        assert normalize_angle(-3 * np.pi) == pytest.approx(-np.pi, abs=1e-10)

    def test_pi(self):
        # pi should map to pi (or -pi, both valid at boundary)
        result = normalize_angle(np.pi)
        assert abs(result) == pytest.approx(np.pi, abs=1e-10)

    def test_two_pi(self):
        assert normalize_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)
