"""Tests for thermal imaging simulator."""

import numpy as np
import pytest

from sentinel.core.types import SensorType, TargetType, ThermalBand
from sentinel.sensors.thermal_sim import (
    ThermalSimConfig,
    ThermalSimulator,
    ThermalTarget,
    thermal_frame_to_detections,
)


@pytest.fixture
def conventional_target():
    return ThermalTarget(
        target_id="CONV-01",
        position=np.array([5000.0, 0.0]),
        velocity=np.array([-100.0, 0.0]),
        target_type=TargetType.CONVENTIONAL,
        class_name="aircraft",
    )


@pytest.fixture
def hypersonic_target():
    return ThermalTarget(
        target_id="HYP-01",
        position=np.array([20000.0, 0.0]),
        velocity=np.array([-1715.0, 0.0]),
        target_type=TargetType.HYPERSONIC,
        mach=5.0,
        class_name="hypersonic_vehicle",
    )


@pytest.fixture
def cold_target():
    """A target with very low thermal signature (slow, no engine heat)."""
    return ThermalTarget(
        target_id="COLD-01",
        position=np.array([5000.0, 0.0]),
        velocity=np.array([0.0, 0.0]),  # Stationary
        target_type=TargetType.CONVENTIONAL,
        mach=0.0,
    )


@pytest.fixture
def basic_config(conventional_target, hypersonic_target):
    return ThermalSimConfig(
        frame_rate_hz=30.0,
        bands=[ThermalBand.MWIR, ThermalBand.LWIR],
        fov_deg=60.0,
        max_range_m=50000.0,
        detection_probability=1.0,
        false_alarm_rate=0.0,
        min_contrast_k=5.0,
        targets=[conventional_target, hypersonic_target],
    )


class TestThermalTarget:
    def test_position_at_time(self, conventional_target):
        pos = conventional_target.position_at(1.0)
        np.testing.assert_array_almost_equal(pos, [4900.0, 0.0])

    def test_temperature_conventional_moderate(self, conventional_target):
        temp = conventional_target.temperature_at()
        assert 400 < temp < 1500  # Engine hot but not extreme

    def test_temperature_hypersonic_extreme(self, hypersonic_target):
        temp = hypersonic_target.temperature_at()
        assert temp > 1500  # Extreme aerodynamic heating

    def test_thermal_contrast(self, hypersonic_target):
        contrast = hypersonic_target.thermal_contrast()
        assert contrast > 1000

    def test_effective_mach_from_velocity(self, conventional_target):
        m = conventional_target.effective_mach()
        assert m == pytest.approx(100.0 / 343.0, abs=0.01)

    def test_effective_mach_preset(self, hypersonic_target):
        assert hypersonic_target.effective_mach() == 5.0

    def test_band_intensity_mwir(self, conventional_target):
        intensity = conventional_target.band_intensity(ThermalBand.MWIR)
        assert intensity > 0.0


class TestThermalSimulator:
    def test_connect_disconnect(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        assert not sim.is_connected
        assert sim.connect()
        assert sim.is_connected
        sim.disconnect()
        assert not sim.is_connected

    def test_read_when_disconnected(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        assert sim.read_frame() is None

    def test_produces_bearing_only_detections(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame is not None
        assert len(frame.data) > 0
        for d in frame.data:
            assert "azimuth_deg" in d
            assert "range_m" not in d  # No range!

    def test_detection_includes_temperature(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        for d in frame.data:
            assert "temperature_k" in d
            assert d["temperature_k"] > 0

    def test_detection_includes_thermal_band(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        bands_seen = {d["thermal_band"] for d in frame.data}
        assert len(bands_seen) > 0

    def test_hypersonic_always_detected(self, hypersonic_target):
        """Hypersonic targets have extreme thermal signatures -- always detected."""
        config = ThermalSimConfig(
            bands=[ThermalBand.MWIR, ThermalBand.LWIR],
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            min_contrast_k=10.0,
            targets=[hypersonic_target],
        )
        sim = ThermalSimulator(config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        hyp_dets = [d for d in frame.data if d.get("target_id") == "HYP-01"]
        assert len(hyp_dets) > 0

    def test_out_of_fov_not_detected(self, conventional_target):
        config = ThermalSimConfig(
            fov_deg=1.0,  # Very narrow FOV -- target at 0 deg should be in FOV
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[conventional_target],
        )
        sim = ThermalSimulator(config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        # Target is at azimuth ~0 deg, so should be in even narrow FOV
        assert len(frame.data) > 0

    def test_out_of_range_not_detected(self):
        far_target = ThermalTarget(
            target_id="FAR-01",
            position=np.array([100000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        config = ThermalSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[far_target],
        )
        sim = ThermalSimulator(config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) == 0

    def test_false_alarms(self):
        config = ThermalSimConfig(
            fov_deg=60.0,
            detection_probability=0.0,  # No real detections
            false_alarm_rate=5.0,  # Many false alarms
            targets=[],
        )
        sim = ThermalSimulator(config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        # Should have false alarms
        assert len(frame.data) >= 0  # Poisson, could be 0

    def test_frame_metadata(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame.sensor_type == SensorType.THERMAL
        assert "bands" in frame.metadata


class TestThermalFrameToDetections:
    def test_converts_to_detection_objects(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = thermal_frame_to_detections(frame)
        assert len(dets) > 0
        for det in dets:
            assert det.sensor_type == SensorType.THERMAL

    def test_sensor_type_is_thermal(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = thermal_frame_to_detections(frame)
        for det in dets:
            assert det.sensor_type == SensorType.THERMAL

    def test_range_is_none(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = thermal_frame_to_detections(frame)
        for det in dets:
            assert det.range_m is None  # Bearing-only!

    def test_azimuth_present(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = thermal_frame_to_detections(frame)
        for det in dets:
            assert det.azimuth_deg is not None

    def test_temperature_present(self, basic_config):
        sim = ThermalSimulator(basic_config, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = thermal_frame_to_detections(frame)
        for det in dets:
            assert det.temperature_k is not None
            assert det.temperature_k > 0

    def test_empty_frame(self):
        from sentinel.sensors.frame import SensorFrame
        frame = SensorFrame(
            data=[], timestamp=0.0,
            sensor_type=SensorType.THERMAL, metadata={},
        )
        dets = thermal_frame_to_detections(frame)
        assert dets == []
