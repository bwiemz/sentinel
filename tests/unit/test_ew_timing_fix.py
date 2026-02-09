"""Verify EW functions receive simulation time, not wall-clock time.

The EW timing bug (fixed in Phase 14) was that simulators passed
``time.time()`` to EW environment functions instead of their own
``self._clock.now()``.  With SimClock, this ensures chaff/decoy
lifetimes are evaluated against deterministic simulation time.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.clock import SimClock
from sentinel.sensors.environment import EnvironmentModel
from sentinel.sensors.ew import ChaffCloud, DecoySource, EWModel, ECCMConfig
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
)
from sentinel.sensors.radar_sim import MultiFreqRadarTarget
from sentinel.sensors.thermal_sim import ThermalSimConfig, ThermalSimulator, ThermalTarget
from sentinel.core.types import TargetType, RadarBand, ThermalBand


class TestChaffUsesSimTime:
    """Chaff activity is evaluated against simulation time."""

    def test_chaff_active_at_sim_start(self):
        """Chaff deployed at sim epoch should be active immediately."""
        clock = SimClock(start_epoch=1_000_000.0)

        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=1_000_000.0,
            initial_rcs_dbsm=30.0,
            lifetime_s=10.0,
            cloud_id="chaff_test",
        )
        ew = EWModel(chaff_clouds=[chaff])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        cfg = RadarSimConfig(
            targets=[],
            max_range_m=20000.0,
            false_alarm_rate=0.0,
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42, clock=clock)
        sim.connect()

        frame = sim.read_frame()
        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) > 0, "Chaff should produce EW detections at sim start"

    def test_chaff_expired_after_lifetime(self):
        """After chaff lifetime, it should no longer produce returns."""
        clock = SimClock(start_epoch=1_000_000.0)

        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=1_000_000.0,
            initial_rcs_dbsm=30.0,
            lifetime_s=10.0,
            cloud_id="chaff_test",
        )
        ew = EWModel(chaff_clouds=[chaff])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        cfg = RadarSimConfig(
            targets=[],
            max_range_m=20000.0,
            false_alarm_rate=0.0,
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42, clock=clock)
        sim.connect()

        # Advance past chaff lifetime
        clock.step(15.0)
        frame = sim.read_frame()
        chaff_dets = [d for d in frame.data if d.get("ew_source_id") == "chaff_test"]
        assert len(chaff_dets) == 0, "Expired chaff should not produce returns"


class TestDecoyUsesSimTime:
    """Decoy thermal returns use simulation time."""

    def test_decoy_thermal_at_sim_start(self):
        """Decoy with IR signature deployed at sim epoch should appear in thermal."""
        clock = SimClock(start_epoch=1_000_000.0)

        decoy = DecoySource(
            position=np.array([3000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=1_000_000.0,
            rcs_dbsm=10.0,
            has_thermal_signature=True,
            thermal_temperature_k=400.0,
            lifetime_s=30.0,
            decoy_id="decoy_test",
        )
        ew = EWModel(decoys=[decoy])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        target = ThermalTarget(
            target_id="T1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = ThermalSimConfig(
            targets=[target],
            bands=[ThermalBand.MWIR],
            false_alarm_rate=0.0,
            environment=env,
        )
        sim = ThermalSimulator(cfg, seed=42, clock=clock)
        sim.connect()

        frame = sim.read_frame()
        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) > 0, "Decoy with IR should appear in thermal"


class TestMultiFreqEWUsesSimTime:
    """Multi-freq radar EW uses simulation time for chaff/decoy."""

    def test_chaff_returns_on_multifreq(self):
        clock = SimClock(start_epoch=1_000_000.0)

        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=1_000_000.0,
            initial_rcs_dbsm=30.0,
            lifetime_s=10.0,
            cloud_id="chaff_mf",
        )
        ew = EWModel(chaff_clouds=[chaff])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        target = MultiFreqRadarTarget(
            target_id="T1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = MultiFreqRadarConfig(
            targets=[target],
            bands=[RadarBand.X_BAND],
            false_alarm_rate=0.0,
            base_detection_probability=1.0,
            environment=env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42, clock=clock)
        sim.connect()

        frame = sim.read_frame()
        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) > 0, "Chaff should produce EW detections on multifreq"
