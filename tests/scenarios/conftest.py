"""Scenario validation framework — helpers, data structures, and ScenarioRunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from sentinel.core.types import (
    Detection,
    RadarBand,
    SensorType,
    TargetType,
    TrackState,
)
from sentinel.fusion.multi_sensor_fusion import (
    EnhancedFusedTrack,
    MultiSensorFusion,
)
from sentinel.fusion.multifreq_correlator import (
    CorrelatedDetection,
    MultiFreqCorrelator,
)
from sentinel.core.clock import Clock, SimClock
from sentinel.sensors.environment import EnvironmentModel
from sentinel.utils.geo_context import GeoContext
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
    MultiFreqRadarTarget,
    multifreq_radar_frame_to_detections,
)
from sentinel.sensors.quantum_radar_sim import (
    QuantumRadarConfig,
    QuantumRadarSimulator,
    quantum_radar_frame_to_detections,
)
from sentinel.sensors.thermal_sim import (
    ThermalSimConfig,
    ThermalSimulator,
    ThermalTarget,
    thermal_frame_to_detections,
)
from sentinel.tracking.radar_track_manager import RadarTrackManager
from sentinel.tracking.thermal_track_manager import ThermalTrackManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioTarget:
    """A target in a scripted scenario with expected outcomes."""

    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    target_type: TargetType
    rcs_dbsm: float
    mach: float
    expected_threat: str
    expected_stealth: bool = False
    expected_hypersonic: bool = False
    position_geo: tuple[float, float, float] | None = None  # (lat, lon, alt)


@dataclass
class ScenarioResult:
    """Collected results from a scenario run."""

    radar_tracks: list
    thermal_tracks: list
    quantum_tracks: list
    correlated_detections: list[CorrelatedDetection]
    fused_tracks: list[EnhancedFusedTrack]
    radar_confirmed_count: int
    thermal_confirmed_count: int
    quantum_confirmed_count: int
    multifreq_detection_log: list[list[Detection]]
    thermal_detection_log: list[list[Detection]]
    quantum_detection_log: list[list[Detection]]
    step_count: int


# ---------------------------------------------------------------------------
# Config helpers (pattern from tests/integration/test_phase9_e2e.py)
# ---------------------------------------------------------------------------

def radar_tracking_config(**overrides) -> DictConfig:
    """RadarTrackManager config with low confirm_hits for fast confirmation."""
    cfg = OmegaConf.create({
        "filter": {"dt": 0.1, "type": "ekf"},
        "association": {"gate_threshold": 9.21},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    })
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def thermal_tracking_config(**overrides) -> DictConfig:
    """ThermalTrackManager config with bearing-only EKF."""
    cfg = OmegaConf.create({
        "filter": {"type": "bearing_ekf", "dt": 0.1, "assumed_initial_range_m": 10000.0},
        "association": {"gate_threshold": 6.635},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    })
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------

class ScenarioRunner:
    """Orchestrates a full detection -> tracking -> fusion scenario.

    Drives MultiFreqRadarSimulator, ThermalSimulator, and optionally
    QuantumRadarSimulator through N steps, feeds detections into their
    respective TrackManagers, then runs MultiFreqCorrelator and
    MultiSensorFusion to produce classified EnhancedFusedTracks.
    """

    def __init__(
        self,
        targets: list[ScenarioTarget],
        seed: int = 42,
        n_steps: int = 15,
        use_multifreq: bool = True,
        use_thermal: bool = True,
        use_quantum: bool = False,
        multifreq_bands: list[RadarBand] | None = None,
        multifreq_base_pd: float = 0.9,
        thermal_fov_deg: float = 60.0,
        quantum_squeeze_r: float = 0.5,
        quantum_max_range_m: float = 15000.0,
        environment: EnvironmentModel | None = None,
        clock: Clock | None = None,
        step_dt: float = 0.1,
        geo_context: GeoContext | None = None,
    ):
        self.targets = targets
        self.geo_context = geo_context
        self._resolve_geodetic_positions()
        self.seed = seed
        self.n_steps = n_steps
        self.use_multifreq = use_multifreq
        self.use_thermal = use_thermal
        self.use_quantum = use_quantum
        self.multifreq_bands = multifreq_bands or list(RadarBand)
        self.multifreq_base_pd = multifreq_base_pd
        self.thermal_fov_deg = thermal_fov_deg
        self.quantum_squeeze_r = quantum_squeeze_r
        self.quantum_max_range_m = quantum_max_range_m
        self.environment = environment
        self.clock = clock
        self.step_dt = step_dt

    # ---------------------------------------------------------------
    # Geodetic resolution
    # ---------------------------------------------------------------

    def _resolve_geodetic_positions(self) -> None:
        """Convert position_geo to ENU position for targets that specify it."""
        if self.geo_context is None:
            return
        for t in self.targets:
            if t.position_geo is not None:
                lat, lon = t.position_geo[0], t.position_geo[1]
                alt = t.position_geo[2] if len(t.position_geo) > 2 else 0.0
                t.position = self.geo_context.target_geodetic_to_xy(lat, lon, alt)

    # ---------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------

    def run(self) -> ScenarioResult:
        """Execute the scenario and return results."""
        # Use provided clock or create a deterministic SimClock
        clock = self.clock if self.clock is not None else SimClock()

        # Build per-simulator target lists
        mf_targets = self._build_multifreq_targets()
        th_targets = self._build_thermal_targets()
        qi_targets = self._build_multifreq_targets()  # quantum reuses MultiFreqRadarTarget

        # Create simulators (all share the same clock for consistency)
        mf_sim = self._create_multifreq_sim(mf_targets, clock=clock)
        th_sim = self._create_thermal_sim(th_targets, clock=clock) if self.use_thermal else None
        qi_sim = self._create_quantum_sim(qi_targets, clock=clock) if self.use_quantum else None

        # Create track managers
        radar_mgr = RadarTrackManager(radar_tracking_config(), geo_context=self.geo_context)
        thermal_mgr = ThermalTrackManager(thermal_tracking_config(), geo_context=self.geo_context) if self.use_thermal else None
        quantum_mgr = RadarTrackManager(radar_tracking_config(), geo_context=self.geo_context) if self.use_quantum else None

        # Connect
        mf_sim.connect()
        if th_sim:
            th_sim.connect()
        if qi_sim:
            qi_sim.connect()

        # Detection logs
        mf_det_log: list[list[Detection]] = []
        th_det_log: list[list[Detection]] = []
        qi_det_log: list[list[Detection]] = []

        # Main loop
        for _ in range(self.n_steps):
            # Step the clock before reading frames (deterministic time advance)
            if isinstance(clock, SimClock):
                clock.step(self.step_dt)

            # Multi-freq radar
            frame = mf_sim.read_frame()
            dets = multifreq_radar_frame_to_detections(frame)
            mf_det_log.append(dets)
            radar_mgr.step(dets)

            # Thermal
            if th_sim and thermal_mgr:
                frame = th_sim.read_frame()
                dets = thermal_frame_to_detections(frame)
                th_det_log.append(dets)
                thermal_mgr.step(dets)

            # Quantum radar
            if qi_sim and quantum_mgr:
                frame = qi_sim.read_frame()
                dets = quantum_radar_frame_to_detections(frame)
                qi_det_log.append(dets)
                quantum_mgr.step(dets)

        # Disconnect
        mf_sim.disconnect()
        if th_sim:
            th_sim.disconnect()
        if qi_sim:
            qi_sim.disconnect()

        # Multi-freq correlation (on last step's detections)
        correlator = MultiFreqCorrelator(
            range_gate_m=200.0,
            azimuth_gate_deg=5.0,
            stealth_rcs_variation_db=15.0,
        )
        last_mf_dets = mf_det_log[-1] if mf_det_log else []
        correlated, _ = correlator.correlate(last_mf_dets)

        # Multi-sensor fusion
        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=5.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=radar_mgr.confirmed_tracks,
            thermal_tracks=thermal_mgr.confirmed_tracks if thermal_mgr else [],
            correlated_detections=correlated or None,
            quantum_radar_tracks=quantum_mgr.confirmed_tracks if quantum_mgr else [],
        )

        return ScenarioResult(
            radar_tracks=radar_mgr.active_tracks,
            thermal_tracks=thermal_mgr.active_tracks if thermal_mgr else [],
            quantum_tracks=quantum_mgr.active_tracks if quantum_mgr else [],
            correlated_detections=correlated,
            fused_tracks=fused,
            radar_confirmed_count=len(radar_mgr.confirmed_tracks),
            thermal_confirmed_count=len(thermal_mgr.confirmed_tracks) if thermal_mgr else 0,
            quantum_confirmed_count=len(quantum_mgr.confirmed_tracks) if quantum_mgr else 0,
            multifreq_detection_log=mf_det_log,
            thermal_detection_log=th_det_log,
            quantum_detection_log=qi_det_log,
            step_count=self.n_steps,
        )

    # ---------------------------------------------------------------
    # Target builders
    # ---------------------------------------------------------------

    def _build_multifreq_targets(self) -> list[MultiFreqRadarTarget]:
        return [
            MultiFreqRadarTarget(
                target_id=t.target_id,
                position=t.position.copy(),
                velocity=t.velocity.copy(),
                rcs_dbsm=t.rcs_dbsm,
                target_type=t.target_type,
                mach=t.mach,
            )
            for t in self.targets
        ]

    def _build_thermal_targets(self) -> list[ThermalTarget]:
        return [
            ThermalTarget(
                target_id=t.target_id,
                position=t.position.copy(),
                velocity=t.velocity.copy(),
                target_type=t.target_type,
                mach=t.mach,
            )
            for t in self.targets
        ]

    # ---------------------------------------------------------------
    # Simulator factories
    # ---------------------------------------------------------------

    def _create_multifreq_sim(self, targets: list[MultiFreqRadarTarget],
                              clock: Clock | None = None) -> MultiFreqRadarSimulator:
        cfg = MultiFreqRadarConfig(
            bands=self.multifreq_bands,
            max_range_m=50000.0,
            fov_deg=120.0,
            base_detection_probability=self.multifreq_base_pd,
            false_alarm_rate=0.0,
            targets=targets,
            environment=self.environment,
        )
        return MultiFreqRadarSimulator(cfg, seed=self.seed, clock=clock)

    def _create_thermal_sim(self, targets: list[ThermalTarget],
                            clock: Clock | None = None) -> ThermalSimulator:
        cfg = ThermalSimConfig(
            fov_deg=self.thermal_fov_deg,
            max_range_m=50000.0,
            detection_probability=0.95,
            false_alarm_rate=0.0,
            targets=targets,
            environment=self.environment,
        )
        return ThermalSimulator(cfg, seed=self.seed + 1, clock=clock)

    def _create_quantum_sim(self, targets: list[MultiFreqRadarTarget],
                            clock: Clock | None = None) -> QuantumRadarSimulator:
        cfg = QuantumRadarConfig(
            max_range_m=self.quantum_max_range_m,
            squeeze_param_r=self.quantum_squeeze_r,
            n_modes=50000,
            targets=targets,
            false_alarm_rate=0.0,
            environment=self.environment,
        )
        return QuantumRadarSimulator(cfg, seed=self.seed + 2, clock=clock)
