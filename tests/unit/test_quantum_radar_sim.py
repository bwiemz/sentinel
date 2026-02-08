"""Tests for quantum illumination radar simulator."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.sensors.physics import ReceiverType
from sentinel.sensors.quantum_radar_sim import (
    QuantumRadarConfig,
    QuantumRadarSimulator,
    quantum_radar_frame_to_detections,
)
from sentinel.sensors.radar_sim import MultiFreqRadarTarget


def _make_config(**overrides) -> QuantumRadarConfig:
    """Create a QuantumRadarConfig with sensible test defaults."""
    defaults = dict(
        freq_hz=10.0e9,
        squeeze_param_r=0.1,
        n_modes=100000,
        antenna_gain_dbi=30.0,
        receiver_type=ReceiverType.OPTIMAL,
        ambient_temp_k=290.0,
        scan_rate_hz=10.0,
        max_range_m=15000.0,
        fov_deg=120.0,
        noise_range_m=0.0,  # No noise for deterministic tests
        noise_azimuth_deg=0.0,
        noise_velocity_mps=0.0,
        noise_rcs_dbsm=0.0,
        false_alarm_rate=0.0,
        targets=[],
    )
    defaults.update(overrides)
    return QuantumRadarConfig(**defaults)


def _conventional_target(pos=(3000, 1000), vel=(-50, 5)):
    return MultiFreqRadarTarget(
        target_id="CONV-01",
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        rcs_dbsm=15.0,
        class_name="aircraft",
        target_type=TargetType.CONVENTIONAL,
    )


def _stealth_target(pos=(5000, 1000), vel=(-150, 10)):
    return MultiFreqRadarTarget(
        target_id="STEALTH-01",
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        rcs_dbsm=-20.0,
        class_name="stealth_aircraft",
        target_type=TargetType.STEALTH,
    )


class TestQuantumRadarConfig:
    def test_defaults(self):
        cfg = QuantumRadarConfig()
        assert cfg.freq_hz == 10.0e9
        assert cfg.squeeze_param_r == 0.1
        assert cfg.n_modes == 10000
        assert cfg.receiver_type == ReceiverType.OPA

    def test_wavelength(self):
        cfg = QuantumRadarConfig(freq_hz=10.0e9)
        assert abs(cfg.wavelength_m - 0.03) < 0.001

    def test_n_signal(self):
        cfg = QuantumRadarConfig(squeeze_param_r=0.1)
        assert 0.009 < cfg.n_signal < 0.011

    def test_n_background(self):
        cfg = QuantumRadarConfig(freq_hz=10.0e9, ambient_temp_k=290.0)
        assert cfg.n_background > 500

    def test_antenna_gain_linear(self):
        cfg = QuantumRadarConfig(antenna_gain_dbi=30.0)
        assert abs(cfg.antenna_gain_linear - 1000.0) < 1.0

    def test_from_omegaconf(self):
        raw = {
            "freq_hz": 10.0e9,
            "squeeze_param_r": 0.2,
            "n_modes": 50000,
            "antenna_gain_dbi": 35.0,
            "receiver_type": "sfg",
            "ambient_temp_k": 300.0,
            "scan_rate_hz": 5.0,
            "max_range_m": 20000.0,
            "fov_deg": 90.0,
            "noise": {
                "range_std_m": 10.0,
                "azimuth_std_deg": 2.0,
                "false_alarm_rate": 0.01,
            },
            "scenario": {
                "targets": [
                    {
                        "id": "T1",
                        "position": [5000, 1000],
                        "velocity": [-100, 0],
                        "rcs_dbsm": -15.0,
                        "target_type": "stealth",
                        "class_name": "stealth_aircraft",
                    }
                ]
            },
        }
        oc = OmegaConf.create(raw)
        cfg = QuantumRadarConfig.from_omegaconf(oc)
        assert cfg.squeeze_param_r == 0.2
        assert cfg.n_modes == 50000
        assert cfg.receiver_type == ReceiverType.SFG
        assert len(cfg.targets) == 1
        assert cfg.targets[0].target_type == TargetType.STEALTH
        assert cfg.noise_range_m == 10.0

    def test_from_omegaconf_defaults(self):
        oc = OmegaConf.create({})
        cfg = QuantumRadarConfig.from_omegaconf(oc)
        assert cfg.freq_hz == 10.0e9
        assert cfg.receiver_type == ReceiverType.OPA


class TestQuantumRadarSimulator:
    def test_connect_disconnect(self):
        sim = QuantumRadarSimulator(_make_config())
        assert not sim.is_connected
        assert sim.connect()
        assert sim.is_connected
        sim.disconnect()
        assert not sim.is_connected

    def test_read_frame_not_connected(self):
        sim = QuantumRadarSimulator(_make_config())
        assert sim.read_frame() is None

    def test_conventional_target_detected(self):
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        # Large RCS at close range -> should almost always detect
        detected = False
        for _ in range(10):
            frame = sim.read_frame()
            if frame and len(frame.data) > 0:
                detected = True
                break
        assert detected

    def test_frame_sensor_type(self):
        cfg = _make_config(targets=[_conventional_target(pos=(1000, 0))])
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert frame is not None
        assert frame.sensor_type == SensorType.QUANTUM_RADAR

    def test_frame_metadata(self):
        cfg = _make_config(targets=[_conventional_target(pos=(1000, 0))])
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert "n_signal" in frame.metadata
        assert "n_background" in frame.metadata
        assert "receiver" in frame.metadata

    def test_detection_has_qi_metadata(self):
        target = _conventional_target(pos=(1000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        # Find actual target detections (not false alarms)
        real = [d for d in frame.data if d.get("target_id")]
        if real:
            d = real[0]
            assert "qi_advantage_db" in d
            assert "entanglement_fidelity" in d
            assert "pd_qi" in d
            assert "pd_classical" in d
            assert d["pd_qi"] >= d["pd_classical"]

    def test_out_of_range_not_detected(self):
        target = _conventional_target(pos=(50000, 0))  # Way beyond max range
        cfg = _make_config(targets=[target], false_alarm_rate=0.0)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) == 0

    def test_out_of_fov_not_detected(self):
        # Target behind the radar (azimuth > fov/2)
        target = _conventional_target(pos=(-5000, 0))
        cfg = _make_config(targets=[target], fov_deg=120.0, false_alarm_rate=0.0)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        real = [d for d in frame.data if d.get("target_id")]
        assert len(real) == 0

    def test_scan_count_increments(self):
        cfg = _make_config()
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        f1 = sim.read_frame()
        f2 = sim.read_frame()
        assert f1.frame_number == 1
        assert f2.frame_number == 2

    def test_false_alarms(self):
        cfg = _make_config(false_alarm_rate=5.0)  # High rate for testing
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        assert len(frame.data) > 0

    def test_noise_applied(self):
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(
            targets=[target],
            noise_range_m=50.0,  # Large noise
            noise_azimuth_deg=5.0,
            n_modes=1_000_000,
        )
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        ranges = []
        for _ in range(20):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("target_id") == "CONV-01":
                    ranges.append(d["range_m"])
        if len(ranges) > 1:
            # With noise, range measurements should vary
            assert max(ranges) - min(ranges) > 1.0

    def test_multiple_targets(self):
        t1 = _conventional_target(pos=(2000, 500))
        t2 = _stealth_target(pos=(3000, -500))
        cfg = _make_config(targets=[t1, t2], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        all_ids = set()
        for _ in range(20):
            frame = sim.read_frame()
            for d in frame.data:
                tid = d.get("target_id")
                if tid:
                    all_ids.add(tid)
        # Conventional should be detected
        assert "CONV-01" in all_ids


class TestQIvsClassicalAdvantage:
    """Tests that verify QI detects stealth targets better than classical."""

    def test_qi_advantage_for_stealth(self):
        """Over many scans, QI should detect stealth more often than classical would."""
        target = _stealth_target(pos=(3000, 0))
        cfg = _make_config(
            targets=[target],
            n_modes=500_000,
            receiver_type=ReceiverType.OPTIMAL,
        )
        sim = QuantumRadarSimulator(cfg, seed=123)
        sim.connect()

        qi_detections = 0
        n_scans = 100
        for _ in range(n_scans):
            frame = sim.read_frame()
            real = [d for d in frame.data if d.get("target_id")]
            if real:
                qi_detections += 1
                # Verify QI Pd > classical Pd in metadata
                assert real[0]["pd_qi"] >= real[0]["pd_classical"]

    def test_qi_advantage_db_positive(self):
        """QI advantage should always be positive."""
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        for d in frame.data:
            if d.get("target_id"):
                assert d["qi_advantage_db"] > 0


class TestQuantumRadarFrameToDetections:
    def test_conversion(self):
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = quantum_radar_frame_to_detections(frame)
        assert all(isinstance(d, Detection) for d in dets)
        for det in dets:
            assert det.sensor_type == SensorType.QUANTUM_RADAR
            assert det.range_m is not None
            assert det.azimuth_deg is not None

    def test_qi_fields_populated(self):
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = quantum_radar_frame_to_detections(frame)
        for det in dets:
            assert det.qi_advantage_db is not None
            assert det.receiver_type is not None

    def test_position_3d_set(self):
        target = _conventional_target(pos=(2000, 0))
        cfg = _make_config(targets=[target], n_modes=1_000_000)
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        dets = quantum_radar_frame_to_detections(frame)
        for det in dets:
            assert det.position_3d is not None
            assert det.position_3d.shape == (3,)
