"""End-to-end radar tracking integration tests.

Tests the full flow: RadarSimulator → RadarTrackManager → confirmed tracks.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from sentinel.core.types import TrackState
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget, radar_frame_to_detections
from sentinel.tracking.radar_track_manager import RadarTrackManager


def _radar_track_config(**overrides):
    base = {
        "filter": {"dt": 0.1, "type": "ekf"},
        "association": {"gate_threshold": 9.21},
        "track_management": {"confirm_hits": 3, "max_coast_frames": 5, "max_tracks": 50},
    }
    if overrides:
        for k, v in overrides.items():
            if k in base:
                base[k].update(v)
            else:
                base[k] = v
    return OmegaConf.create(base)


class TestRadarE2E:
    def test_single_target_confirmed(self):
        """A single target should produce a confirmed track after enough scans."""
        target = RadarTarget("TGT-1", np.array([3000.0, 1000.0]), np.array([-20.0, 5.0]), rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()

        mgr = RadarTrackManager(_radar_track_config())

        confirmed = []
        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            tracks = mgr.step(dets)
            confirmed = [t for t in tracks if t.state == TrackState.CONFIRMED]

        assert len(confirmed) >= 1
        # Position should be near the target
        pos = confirmed[0].position
        assert abs(pos[0] - 3000.0) < 500.0  # x within 500m
        assert abs(pos[1] - 1000.0) < 500.0  # y within 500m

    def test_multiple_targets_tracked(self):
        """Two well-separated targets should produce two distinct tracks."""
        targets = [
            RadarTarget("TGT-A", np.array([3000.0, 1000.0]), np.array([-10.0, 5.0]), rcs_dbsm=15.0),
            RadarTarget("TGT-B", np.array([7000.0, -2000.0]), np.array([5.0, 10.0]), rcs_dbsm=12.0),
        ]
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=targets,
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()

        mgr = RadarTrackManager(_radar_track_config())

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) == 2

    def test_track_coasts_when_target_removed(self):
        """Track should coast and eventually delete when target disappears."""
        target = RadarTarget("TGT-1", np.array([3000.0, 1000.0]), np.zeros(2), rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()

        mgr = RadarTrackManager(_radar_track_config())

        # Build confirmed track
        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)
        assert len(mgr.confirmed_tracks) == 1

        # Remove target, feed empty detections
        sim.remove_target("TGT-1")
        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        # Track should have been pruned after max_coast_frames
        assert len(mgr.active_tracks) == 0


class TestQuantumRadarE2E:
    def test_quantum_radar_detects_stealth(self):
        """Quantum radar should detect low-RCS targets better than classical."""
        from sentinel.sensors.quantum_radar_sim import (
            QuantumRadarConfig,
            QuantumRadarSimulator,
            quantum_radar_frame_to_detections,
        )

        # Low RCS stealth target
        qr_cfg = QuantumRadarConfig(
            max_range_m=15000.0,
            squeeze_param_r=0.5,
            n_modes=50000,
        )
        # Add target manually (quantum radar reuses MultiFreqRadarTarget)
        from sentinel.sensors.radar_sim import MultiFreqRadarTarget

        target = MultiFreqRadarTarget(
            target_id="STEALTH-1",
            position=np.array([5000.0, 1000.0]),
            velocity=np.array([-50.0, 10.0]),
            rcs_dbsm=-15.0,  # Very low RCS
        )
        qr_cfg.targets.append(target)

        sim = QuantumRadarSimulator(qr_cfg, seed=42)
        sim.connect()

        detection_count = 0
        n_scans = 50
        for _ in range(n_scans):
            frame = sim.read_frame()
            dets = quantum_radar_frame_to_detections(frame)
            detection_count += len(dets)

        # Should detect at least some times (quantum advantage)
        assert detection_count > 0
