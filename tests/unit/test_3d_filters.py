"""Tests for 3D Kalman filters and coordinate transforms."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.tracking.filters import ExtendedKalmanFilter3D, KalmanFilter3D
from sentinel.tracking.radar_track import RadarTrack
from sentinel.utils.coords import (
    cartesian_to_polar_3d,
    polar_to_cartesian_3d,
)


class TestCoords3D:
    """Test 3D coordinate transforms."""

    def test_polar_to_cartesian_3d_horizon(self):
        """At elevation 0, z should be 0."""
        pos = polar_to_cartesian_3d(1000.0, 0.0, 0.0)
        np.testing.assert_allclose(pos, [1000.0, 0.0, 0.0], atol=1e-10)

    def test_polar_to_cartesian_3d_up(self):
        """At elevation pi/2, should be straight up."""
        pos = polar_to_cartesian_3d(1000.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(pos, [0.0, 0.0, 1000.0], atol=1e-10)

    def test_polar_to_cartesian_3d_45deg(self):
        """At 45 deg elevation, xy and z should be equal in magnitude."""
        r = 1000.0
        pos = polar_to_cartesian_3d(r, 0.0, np.pi / 4)
        expected_z = r * np.sin(np.pi / 4)
        expected_x = r * np.cos(np.pi / 4)
        assert pos[2] == pytest.approx(expected_z)
        assert pos[0] == pytest.approx(expected_x)

    def test_cartesian_to_polar_3d_roundtrip(self):
        """roundtrip: polar -> cartesian -> polar."""
        r_orig, az_orig, el_orig = 5000.0, np.radians(30.0), np.radians(15.0)
        pos = polar_to_cartesian_3d(r_orig, az_orig, el_orig)
        r, az, el = cartesian_to_polar_3d(pos[0], pos[1], pos[2])
        assert r == pytest.approx(r_orig, rel=1e-10)
        assert az == pytest.approx(az_orig, rel=1e-10)
        assert el == pytest.approx(el_orig, rel=1e-10)

    def test_cartesian_to_polar_3d_basic(self):
        r, az, el = cartesian_to_polar_3d(1000.0, 0.0, 0.0)
        assert r == pytest.approx(1000.0)
        assert az == pytest.approx(0.0)
        assert el == pytest.approx(0.0)


class TestKalmanFilter3D:
    """Test 3D linear Kalman filter."""

    def test_init(self):
        kf = KalmanFilter3D(dt=0.1)
        assert kf.dim_state == 6
        assert kf.dim_meas == 3
        assert kf.x.shape == (6,)
        assert kf.F.shape == (6, 6)
        assert kf.H.shape == (3, 6)

    def test_position_3d(self):
        kf = KalmanFilter3D()
        kf.x = np.array([100.0, 0.0, 200.0, 0.0, 300.0, 0.0])
        np.testing.assert_allclose(kf.position, [100.0, 200.0, 300.0])

    def test_velocity_3d(self):
        kf = KalmanFilter3D()
        kf.x = np.array([0.0, 10.0, 0.0, 20.0, 0.0, 30.0])
        np.testing.assert_allclose(kf.velocity, [10.0, 20.0, 30.0])

    def test_predict_constant_velocity(self):
        kf = KalmanFilter3D(dt=1.0)
        kf.x = np.array([0.0, 10.0, 0.0, 20.0, 0.0, -5.0])
        kf.predict()
        assert kf.x[0] == pytest.approx(10.0, abs=0.1)  # x + vx*dt
        assert kf.x[2] == pytest.approx(20.0, abs=0.1)  # y + vy*dt
        assert kf.x[4] == pytest.approx(-5.0, abs=0.1)  # z + vz*dt

    def test_update_converges(self):
        kf = KalmanFilter3D(dt=0.1)
        true_pos = np.array([1000.0, 2000.0, 500.0])
        for _ in range(30):
            kf.predict()
            z = true_pos + np.random.randn(3) * 5.0
            kf.update(z)
        np.testing.assert_allclose(kf.position, true_pos, atol=50.0)

    def test_gating_distance_at_prediction(self):
        kf = KalmanFilter3D()
        kf.x = np.array([100.0, 0.0, 200.0, 0.0, 300.0, 0.0])
        dist = kf.gating_distance(np.array([100.0, 200.0, 300.0]))
        assert dist < 1.0


class TestExtendedKalmanFilter3D:
    """Test 3D EKF with polar measurements."""

    def test_init(self):
        ekf = ExtendedKalmanFilter3D(dt=0.1)
        assert ekf.dim_state == 6
        assert ekf.dim_meas == 3
        assert ekf.R.shape == (3, 3)

    def test_measurement_function(self):
        ekf = ExtendedKalmanFilter3D()
        # Target at (3000, 0, 0, 4000, 0, 0, 500, 0)
        ekf.x = np.array([3000.0, 0.0, 4000.0, 0.0, 500.0, 0.0])
        h = ekf.h(ekf.x)
        expected_r = np.sqrt(3000**2 + 4000**2 + 500**2)
        expected_az = np.arctan2(4000.0, 3000.0)
        r_xy = np.sqrt(3000**2 + 4000**2)
        expected_el = np.arctan2(500.0, r_xy)
        assert h[0] == pytest.approx(expected_r, rel=1e-6)
        assert h[1] == pytest.approx(expected_az, rel=1e-6)
        assert h[2] == pytest.approx(expected_el, rel=1e-4)

    def test_jacobian_shape(self):
        ekf = ExtendedKalmanFilter3D()
        ekf.x = np.array([3000.0, 10.0, 4000.0, -5.0, 500.0, 0.0])
        H = ekf.H_jacobian(ekf.x)
        assert H.shape == (3, 6)

    def test_position_velocity(self):
        ekf = ExtendedKalmanFilter3D()
        ekf.x = np.array([100.0, 10.0, 200.0, 20.0, 300.0, 30.0])
        np.testing.assert_allclose(ekf.position, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(ekf.velocity, [10.0, 20.0, 30.0])

    def test_update_converges(self):
        ekf = ExtendedKalmanFilter3D(dt=0.1)
        true_x, true_y, true_z = 5000.0, 3000.0, 1000.0
        ekf.x = np.array([4000.0, 0.0, 2000.0, 0.0, 500.0, 0.0])

        true_r = np.sqrt(true_x**2 + true_y**2 + true_z**2)
        true_az = np.arctan2(true_y, true_x)
        true_el = np.arctan2(true_z, np.sqrt(true_x**2 + true_y**2))

        for _ in range(50):
            ekf.predict()
            z = np.array([
                true_r + np.random.randn() * 5.0,
                true_az + np.random.randn() * np.radians(1.0),
                true_el + np.random.randn() * np.radians(1.0),
            ])
            ekf.update(z)

        assert abs(ekf.position[0] - true_x) < 300.0
        assert abs(ekf.position[1] - true_y) < 300.0
        assert abs(ekf.position[2] - true_z) < 300.0

    def test_set_measurement_noise(self):
        ekf = ExtendedKalmanFilter3D()
        ekf.set_measurement_noise(10.0, np.radians(2.0), np.radians(2.0))
        assert ekf.R[0, 0] == pytest.approx(100.0)
        assert ekf.R.shape == (3, 3)


class TestRadarTrack3D:
    """Test RadarTrack with 3D mode enabled."""

    def _make_radar_det(self, range_m=5000.0, az_deg=30.0, el_deg=10.0):
        return Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=range_m,
            azimuth_deg=az_deg,
            elevation_deg=el_deg,
            confidence=0.9,
        )

    def test_3d_init(self):
        det = self._make_radar_det()
        track = RadarTrack(det, use_3d=True)
        # Position should be 3D
        assert track.position.shape == (3,)
        assert track.velocity.shape == (3,)

    def test_3d_range(self):
        det = self._make_radar_det(range_m=5000.0, az_deg=0.0, el_deg=0.0)
        track = RadarTrack(det, use_3d=True)
        assert track.range_m == pytest.approx(5000.0, rel=0.01)

    def test_3d_elevation(self):
        det = self._make_radar_det(range_m=5000.0, az_deg=0.0, el_deg=30.0)
        track = RadarTrack(det, use_3d=True)
        assert track.elevation_deg is not None
        assert track.elevation_deg == pytest.approx(30.0, abs=1.0)

    def test_2d_has_no_elevation(self):
        det = self._make_radar_det()
        track = RadarTrack(det, use_3d=False)
        assert track.elevation_deg is None

    def test_3d_predict_update(self):
        det = self._make_radar_det(range_m=5000.0, az_deg=30.0, el_deg=10.0)
        track = RadarTrack(det, use_3d=True)
        track.predict()
        # Update with a slightly different measurement
        det2 = self._make_radar_det(range_m=4990.0, az_deg=30.1, el_deg=10.1)
        track.update(det2)
        assert track.hits == 2

    def test_3d_update_converges(self):
        """3D track should converge to true target position."""
        true_r, true_az, true_el = 8000.0, np.radians(45.0), np.radians(20.0)
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=0.0,
            range_m=true_r,
            azimuth_deg=np.degrees(true_az),
            elevation_deg=np.degrees(true_el),
            confidence=0.9,
        )
        track = RadarTrack(det, use_3d=True, dt=0.1)

        for _ in range(30):
            track.predict()
            noisy_det = Detection(
                sensor_type=SensorType.RADAR,
                timestamp=0.0,
                range_m=true_r + np.random.randn() * 10.0,
                azimuth_deg=np.degrees(true_az) + np.random.randn() * 1.0,
                elevation_deg=np.degrees(true_el) + np.random.randn() * 1.0,
                confidence=0.9,
            )
            track.update(noisy_det)

        assert track.range_m == pytest.approx(true_r, rel=0.05)
