"""Tests for radar simulator sensor."""

import numpy as np
import pytest

from sentinel.core.types import SensorType
from sentinel.sensors.radar_sim import (
    RadarSimConfig,
    RadarSimulator,
    RadarTarget,
    radar_frame_to_detections,
)


def _simple_config(targets=None, **kwargs):
    """Create a simple radar config for testing."""
    defaults = dict(
        scan_rate_hz=10.0,
        max_range_m=10000.0,
        fov_deg=120.0,
        noise_range_m=5.0,
        noise_azimuth_deg=1.0,
        noise_velocity_mps=0.5,
        noise_rcs_dbsm=2.0,
        false_alarm_rate=0.0,  # no false alarms by default in tests
        detection_probability=1.0,  # always detect in tests
        targets=targets or [],
    )
    defaults.update(kwargs)
    return RadarSimConfig(**defaults)


def _target_on_x_axis(range_m=3000.0):
    """Create a stationary target on the +x axis."""
    return RadarTarget(
        target_id="TGT-01",
        position=np.array([range_m, 0.0]),
        velocity=np.array([0.0, 0.0]),
        rcs_dbsm=15.0,
        class_name="vehicle",
    )


class TestRadarTarget:
    def test_position_at_zero(self):
        tgt = RadarTarget("T1", np.array([100.0, 200.0]), np.array([10.0, -5.0]))
        np.testing.assert_array_equal(tgt.position_at(0.0), [100.0, 200.0])

    def test_position_at_time(self):
        tgt = RadarTarget("T1", np.array([100.0, 200.0]), np.array([10.0, -5.0]))
        pos = tgt.position_at(10.0)
        np.testing.assert_array_equal(pos, [200.0, 150.0])


class TestRadarSimulator:
    def test_connect_disconnect(self):
        sim = RadarSimulator(_simple_config(), seed=42)
        assert not sim.is_connected
        assert sim.connect()
        assert sim.is_connected
        sim.disconnect()
        assert not sim.is_connected

    def test_read_when_disconnected(self):
        sim = RadarSimulator(_simple_config(), seed=42)
        assert sim.read_frame() is None

    def test_produces_detections(self):
        tgt = _target_on_x_axis()
        sim = RadarSimulator(_simple_config(targets=[tgt]), seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame is not None
        assert frame.sensor_type == SensorType.RADAR
        assert len(frame.data) >= 1
        sim.disconnect()

    def test_detection_has_required_fields(self):
        tgt = _target_on_x_axis()
        sim = RadarSimulator(_simple_config(targets=[tgt]), seed=42)
        sim.connect()
        frame = sim.read_frame()
        d = frame.data[0]
        assert "range_m" in d
        assert "azimuth_deg" in d
        assert "velocity_mps" in d
        assert "rcs_dbsm" in d
        sim.disconnect()

    def test_target_out_of_range(self):
        tgt = RadarTarget("T1", np.array([20000.0, 0.0]), np.array([0.0, 0.0]))
        sim = RadarSimulator(_simple_config(targets=[tgt], max_range_m=10000.0), seed=42)
        sim.connect()
        frame = sim.read_frame()
        # Should have no detections (target beyond max range, no false alarms)
        assert len(frame.data) == 0
        sim.disconnect()

    def test_target_out_of_fov(self):
        # Target at 90 degrees, FOV is 60 degrees (±30 from boresight)
        tgt = RadarTarget("T1", np.array([0.0, 5000.0]), np.array([0.0, 0.0]))
        sim = RadarSimulator(_simple_config(targets=[tgt], fov_deg=60.0), seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) == 0
        sim.disconnect()

    def test_target_inside_narrow_fov(self):
        # Target at ~5.7 degrees, FOV is 60 degrees
        tgt = RadarTarget("T1", np.array([5000.0, 500.0]), np.array([0.0, 0.0]))
        sim = RadarSimulator(_simple_config(targets=[tgt], fov_deg=60.0), seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) == 1
        sim.disconnect()

    def test_detection_noise_spread(self):
        """Repeated measurements of same target should have noise spread."""
        tgt = _target_on_x_axis(3000.0)
        cfg = _simple_config(targets=[tgt], noise_range_m=5.0)
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        ranges = []
        for _ in range(100):
            frame = sim.read_frame()
            if frame.data:
                ranges.append(frame.data[0]["range_m"])

        # Check std of range measurements is roughly noise_range_m
        assert len(ranges) > 50
        std = np.std(ranges)
        assert 2.0 < std < 10.0  # roughly 5.0 ± some tolerance
        sim.disconnect()

    def test_detection_probability(self):
        """With Pd=0.5, roughly half of scans should detect the target."""
        tgt = _target_on_x_axis()
        cfg = _simple_config(targets=[tgt], detection_probability=0.5)
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        detections = 0
        N = 200
        for _ in range(N):
            frame = sim.read_frame()
            if frame.data:
                detections += 1

        ratio = detections / N
        assert 0.3 < ratio < 0.7
        sim.disconnect()

    def test_false_alarms(self):
        """With false_alarm_rate > 0 and no targets, should see some false alarms."""
        cfg = _simple_config(false_alarm_rate=2.0)  # high rate for testing
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()

        total_fa = 0
        for _ in range(100):
            frame = sim.read_frame()
            total_fa += len(frame.data)

        assert total_fa > 50  # Poisson(2.0) * 100 ≈ 200
        sim.disconnect()

    def test_radial_velocity_approaching(self):
        """Target moving directly toward radar should have negative radial velocity."""
        tgt = RadarTarget("T1", np.array([5000.0, 0.0]), np.array([-100.0, 0.0]))
        cfg = _simple_config(targets=[tgt], noise_velocity_mps=0.0)
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame.data[0]["velocity_mps"] == pytest.approx(-100.0, abs=1.0)
        sim.disconnect()

    def test_radial_velocity_tangential(self):
        """Target moving perpendicular to LOS should have ~0 radial velocity."""
        tgt = RadarTarget("T1", np.array([5000.0, 0.0]), np.array([0.0, 50.0]))
        cfg = _simple_config(targets=[tgt], noise_velocity_mps=0.0)
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame.data[0]["velocity_mps"] == pytest.approx(0.0, abs=1.0)
        sim.disconnect()

    def test_add_remove_target(self):
        sim = RadarSimulator(_simple_config(), seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) == 0

        sim.add_target(_target_on_x_axis())
        frame = sim.read_frame()
        assert len(frame.data) == 1

        sim.remove_target("TGT-01")
        frame = sim.read_frame()
        assert len(frame.data) == 0
        sim.disconnect()

    def test_frame_metadata(self):
        sim = RadarSimulator(_simple_config(targets=[_target_on_x_axis()]), seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame.frame_number == 1
        assert frame.metadata["scan_count"] == 1
        assert frame.metadata["target_count"] == 1
        sim.disconnect()


class TestRadarFrameToDetections:
    def test_converts_to_detection_objects(self):
        tgt = _target_on_x_axis(3000.0)
        sim = RadarSimulator(_simple_config(targets=[tgt]), seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = radar_frame_to_detections(frame)
        assert len(dets) == 1
        d = dets[0]
        assert d.sensor_type == SensorType.RADAR
        assert d.range_m is not None
        assert d.azimuth_deg is not None
        assert d.velocity_mps is not None
        assert d.rcs_dbsm is not None
        assert d.position_3d is not None
        assert d.position_3d.shape == (3,)
        sim.disconnect()

    def test_empty_frame(self):
        sim = RadarSimulator(_simple_config(), seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = radar_frame_to_detections(frame)
        assert dets == []
        sim.disconnect()
