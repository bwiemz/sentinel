"""Verify all simulators accept and use injected clocks."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.clock import Clock, SimClock, SystemClock
from sentinel.core.types import TargetType, ThermalBand, RadarBand
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
)
from sentinel.sensors.radar_sim import MultiFreqRadarTarget
from sentinel.sensors.thermal_sim import ThermalSimConfig, ThermalSimulator, ThermalTarget
from sentinel.sensors.quantum_radar_sim import QuantumRadarConfig, QuantumRadarSimulator


# ---------------------------------------------------------------
# RadarSimulator
# ---------------------------------------------------------------

class TestRadarSimClockInjection:
    def _make_sim(self, clock=None):
        target = RadarTarget("T1", np.array([5000.0, 0.0]), np.array([0.0, 0.0]))
        cfg = RadarSimConfig(
            targets=[target],
            false_alarm_rate=0.0,
            detection_probability=1.0,
        )
        return RadarSimulator(cfg, seed=42, clock=clock)

    def test_default_clock_is_system_clock(self):
        sim = self._make_sim()
        assert isinstance(sim._clock, SystemClock)

    def test_injected_clock_used(self):
        clock = SimClock(start_epoch=2000.0)
        sim = self._make_sim(clock=clock)
        assert sim._clock is clock

    def test_frame_timestamp_from_sim_clock(self):
        clock = SimClock(start_epoch=2000.0)
        sim = self._make_sim(clock=clock)
        sim.connect()
        frame = sim.read_frame()
        assert frame.timestamp == pytest.approx(2000.0, abs=0.01)

    def test_stepped_clock_changes_timestamp(self):
        clock = SimClock(start_epoch=2000.0)
        sim = self._make_sim(clock=clock)
        sim.connect()
        _ = sim.read_frame()
        clock.step(1.0)
        frame2 = sim.read_frame()
        assert frame2.timestamp == pytest.approx(2001.0, abs=0.01)

    def test_deterministic_detections(self):
        """Same clock steps + same seed = identical detection data."""
        results = []
        for _ in range(2):
            clock = SimClock(start_epoch=1000.0)
            sim = self._make_sim(clock=clock)
            sim.connect()
            clock.step(0.5)
            frame = sim.read_frame()
            results.append(frame.data)
            sim.disconnect()
        assert len(results[0]) == len(results[1])
        for d1, d2 in zip(results[0], results[1]):
            assert d1["range_m"] == pytest.approx(d2["range_m"])
            assert d1["azimuth_deg"] == pytest.approx(d2["azimuth_deg"])


# ---------------------------------------------------------------
# MultiFreqRadarSimulator
# ---------------------------------------------------------------

class TestMultiFreqRadarSimClockInjection:
    def _make_sim(self, clock=None):
        target = MultiFreqRadarTarget(
            target_id="T1",
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = MultiFreqRadarConfig(
            targets=[target],
            bands=[RadarBand.X_BAND],
            false_alarm_rate=0.0,
            base_detection_probability=1.0,
        )
        return MultiFreqRadarSimulator(cfg, seed=42, clock=clock)

    def test_default_clock_is_system_clock(self):
        sim = self._make_sim()
        assert isinstance(sim._clock, SystemClock)

    def test_injected_clock_used(self):
        clock = SimClock(start_epoch=3000.0)
        sim = self._make_sim(clock=clock)
        assert sim._clock is clock

    def test_frame_timestamp_from_sim_clock(self):
        clock = SimClock(start_epoch=3000.0)
        sim = self._make_sim(clock=clock)
        sim.connect()
        frame = sim.read_frame()
        assert frame.timestamp == pytest.approx(3000.0, abs=0.01)


# ---------------------------------------------------------------
# ThermalSimulator
# ---------------------------------------------------------------

class TestThermalSimClockInjection:
    def _make_sim(self, clock=None):
        target = ThermalTarget(
            target_id="T1",
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = ThermalSimConfig(
            targets=[target],
            bands=[ThermalBand.MWIR],
            false_alarm_rate=0.0,
            detection_probability=1.0,
        )
        return ThermalSimulator(cfg, seed=42, clock=clock)

    def test_default_clock_is_system_clock(self):
        sim = self._make_sim()
        assert isinstance(sim._clock, SystemClock)

    def test_injected_clock_used(self):
        clock = SimClock(start_epoch=4000.0)
        sim = self._make_sim(clock=clock)
        assert sim._clock is clock

    def test_frame_timestamp_from_sim_clock(self):
        clock = SimClock(start_epoch=4000.0)
        sim = self._make_sim(clock=clock)
        sim.connect()
        frame = sim.read_frame()
        assert frame.timestamp == pytest.approx(4000.0, abs=0.01)


# ---------------------------------------------------------------
# QuantumRadarSimulator
# ---------------------------------------------------------------

class TestQuantumRadarSimClockInjection:
    def _make_sim(self, clock=None):
        target = MultiFreqRadarTarget(
            target_id="T1",
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = QuantumRadarConfig(
            targets=[target],
            false_alarm_rate=0.0,
        )
        return QuantumRadarSimulator(cfg, seed=42, clock=clock)

    def test_default_clock_is_system_clock(self):
        sim = self._make_sim()
        assert isinstance(sim._clock, SystemClock)

    def test_injected_clock_used(self):
        clock = SimClock(start_epoch=5000.0)
        sim = self._make_sim(clock=clock)
        assert sim._clock is clock

    def test_frame_timestamp_from_sim_clock(self):
        clock = SimClock(start_epoch=5000.0)
        sim = self._make_sim(clock=clock)
        sim.connect()
        frame = sim.read_frame()
        assert frame.timestamp == pytest.approx(5000.0, abs=0.01)
