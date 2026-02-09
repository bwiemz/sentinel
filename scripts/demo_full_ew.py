"""SENTINEL full-feature demo: all sensors + environment + EW.

Runs real simulators (multi-freq radar, thermal, quantum illumination) with:
  - Mixed threat scenario: stealth, hypersonic, conventional aircraft
  - Electronic warfare: noise jammer, chaff cloud, expendable decoy
  - Environment: light rain, atmospheric propagation
  - ECCM: QI quantum jamming resistance

Feeds live data into the web dashboard + prints console summary.

Run (real-time):
    python scripts/demo_full_ew.py

Run (deterministic sim clock — fast replay, reproducible):
    python scripts/demo_full_ew.py --sim

Then open http://localhost:8080 in your browser.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

from sentinel.core.clock import Clock, SimClock, SystemClock

# ---------------------------------------------------------------------------
# Imports — sensors, tracking, fusion, EW, environment, web UI
# ---------------------------------------------------------------------------

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
    THREAT_CRITICAL,
    THREAT_HIGH,
    THREAT_LOW,
    THREAT_MEDIUM,
)
from sentinel.fusion.multifreq_correlator import MultiFreqCorrelator
from sentinel.sensors.environment import EnvironmentModel, WeatherConditions
from sentinel.sensors.ew import (
    ChaffCloud,
    DecoySource,
    ECCMConfig,
    EWModel,
    JammerSource,
)
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

from omegaconf import OmegaConf

PORT = 8080
UPDATE_HZ = 10  # Simulation steps per second
STEP_INTERVAL = 1.0 / UPDATE_HZ

# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

def _build_environment(deploy_time: float | None = None) -> EnvironmentModel:
    """Full environment: weather + EW (jammer + chaff + decoy + ECCM).

    Args:
        deploy_time: Epoch time for chaff/decoy deployment. Defaults to
            ``time.time()`` for real-time mode, but should be set to
            ``clock.now()`` for deterministic sim mode.
    """
    now = deploy_time if deploy_time is not None else time.time()

    # Weather: light rain
    weather = WeatherConditions(
        rain_rate_mm_h=5.0,
        visibility_km=8.0,
        cloud_cover_pct=30.0,
        humidity_pct=70.0,
    )

    # EW threats
    jammer = JammerSource(
        position=np.array([30000.0, 5000.0]),  # Stand-off jammer at 30km
        erp_watts=3e5,
        bandwidth_hz=1e6,
        jam_type="noise",
        active=True,
        target_bands=[RadarBand.X_BAND, RadarBand.S_BAND],  # Narrowband — doesn't affect VHF
    )
    chaff = ChaffCloud(
        position=np.array([7000.0, 2000.0]),
        velocity=np.array([-5.0, -1.0]),  # Drifting
        deploy_time=now,
        initial_rcs_dbsm=32.0,
        lifetime_s=600.0,  # 10 minutes for demo
    )
    decoy = DecoySource(
        position=np.array([5000.0, -1000.0]),
        velocity=np.array([40.0, 5.0]),
        rcs_dbsm=8.0,
        has_thermal_signature=False,  # No IR — fusion will flag as decoy
        deploy_time=now,
        lifetime_s=600.0,
    )
    eccm = ECCMConfig(
        quantum_eccm=True,
        quantum_eccm_advantage_db=6.0,
        sidelobe_blanking=True,
        frequency_agility=True,
        frequency_agility_bands=3,
    )
    ew = EWModel(
        jammers=[jammer],
        chaff_clouds=[chaff],
        decoys=[decoy],
        eccm=eccm,
        radar_peak_power_w=1e6,
        radar_gain_db=30.0,
        radar_bandwidth_hz=1e6,
    )

    return EnvironmentModel(
        weather=weather,
        ew=ew,
        use_weather_effects=True,
        use_atmospheric_propagation=True,
        use_ew_effects=True,
    )


def _build_targets() -> tuple[list[MultiFreqRadarTarget], list[ThermalTarget]]:
    """Mixed threat: stealth + hypersonic + conventional."""
    stealth = dict(
        target_id="STEALTH-1",
        position=np.array([12000.0, 3000.0]),
        velocity=np.array([-80.0, -20.0]),  # Inbound
        rcs_dbsm=-15.0,  # Very low X-band RCS
        target_type=TargetType.STEALTH,
        mach=0.9,
    )
    hypersonic = dict(
        target_id="HYPER-1",
        position=np.array([20000.0, -2000.0]),
        velocity=np.array([-500.0, 50.0]),  # Mach 5+ inbound
        rcs_dbsm=5.0,
        target_type=TargetType.HYPERSONIC,
        mach=5.5,
    )
    conventional = dict(
        target_id="CONV-1",
        position=np.array([8000.0, 1000.0]),
        velocity=np.array([-60.0, 10.0]),
        rcs_dbsm=10.0,
        target_type=TargetType.CONVENTIONAL,
        mach=0.8,
    )

    radar_targets = [
        MultiFreqRadarTarget(**stealth),
        MultiFreqRadarTarget(**hypersonic),
        MultiFreqRadarTarget(**conventional),
    ]
    thermal_targets = [
        ThermalTarget(
            target_id=t["target_id"],
            position=t["position"].copy(),
            velocity=t["velocity"].copy(),
            target_type=t["target_type"],
            mach=t["mach"],
        )
        for t in [stealth, hypersonic, conventional]
    ]
    return radar_targets, thermal_targets


# ---------------------------------------------------------------------------
# Tracking configs
# ---------------------------------------------------------------------------

def _radar_cfg():
    return OmegaConf.create({
        "filter": {"dt": 0.2, "type": "ekf"},
        "association": {"gate_threshold": 9.21},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 8,
            "max_tracks": 50,
        },
    })


def _thermal_cfg():
    return OmegaConf.create({
        "filter": {"type": "bearing_ekf", "dt": 0.2, "assumed_initial_range_m": 10000.0},
        "association": {"gate_threshold": 6.635},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 8,
            "max_tracks": 50,
        },
    })


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------

THREAT_COLORS = {
    "CRITICAL": "\033[91m",  # Red
    "HIGH": "\033[93m",      # Yellow
    "MEDIUM": "\033[96m",    # Cyan
    "LOW": "\033[90m",       # Gray
    "UNKNOWN": "\033[37m",   # White
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _print_header():
    print("\033[2J\033[H", end="")  # Clear screen
    print(f"{BOLD}{'=' * 76}")
    print(f"  SENTINEL  Full-Feature Demo — All Sensors + EW + Environment")
    print(f"{'=' * 76}{RESET}")
    print(f"  Web dashboard: {BOLD}http://localhost:{PORT}{RESET}")
    print(f"  Press Ctrl+C to stop\n")


def _print_step(step: int, elapsed: float,
                mf_dets: list, th_dets: list, qi_dets: list,
                radar_mgr, thermal_mgr, quantum_mgr,
                correlated, fused: list[EnhancedFusedTrack],
                env: EnvironmentModel,
                clock: Clock | None = None):
    """Print a compact console summary of the current step."""
    # Move cursor to line 8 (below header)
    print(f"\033[8;0H\033[J", end="")

    ew_dets = sum(1 for d in mf_dets if d.is_ew_generated)
    real_dets = len(mf_dets) - ew_dets

    print(f"  {DIM}Step {step:>4d}  |  t = {elapsed:.1f}s{RESET}")
    print()

    # Sensor detections
    print(f"  {BOLD}Detections this step:{RESET}")
    print(f"    Multi-freq radar: {real_dets} real + {ew_dets} EW-injected  |  "
          f"Thermal: {len(th_dets)}  |  Quantum: {len(qi_dets)}")

    # EW status
    now = clock.now() if clock else time.time()
    n_jammers = len(env.ew.jammers) if env.ew else 0
    n_chaff = sum(1 for c in (env.ew.chaff_clouds if env.ew else []) if c.is_active(now))
    n_decoys = sum(1 for d in (env.ew.decoys if env.ew else []) if d.is_active(now))
    print(f"    {DIM}EW active: {n_jammers} jammer(s), {n_chaff} chaff, {n_decoys} decoy(s){RESET}")
    print()

    # Track counts
    rc = len(radar_mgr.confirmed_tracks)
    ra = len(radar_mgr.active_tracks)
    tc = len(thermal_mgr.confirmed_tracks)
    ta = len(thermal_mgr.active_tracks)
    qc = len(quantum_mgr.confirmed_tracks)
    qa = len(quantum_mgr.active_tracks)
    print(f"  {BOLD}Tracks (confirmed / active):{RESET}")
    print(f"    Radar: {rc}/{ra}  |  Thermal: {tc}/{ta}  |  Quantum: {qc}/{qa}")
    print()

    # Correlation
    chaff_corr = sum(1 for c in correlated if c.is_chaff_candidate)
    stealth_corr = sum(1 for c in correlated if c.is_stealth_candidate)
    print(f"  {BOLD}Multi-freq correlation:{RESET} {len(correlated)} groups"
          f"  ({stealth_corr} stealth, {chaff_corr} chaff)")
    print()

    # Fused tracks with threat classification
    print(f"  {BOLD}Fused tracks ({len(fused)}):{RESET}")
    if not fused:
        print(f"    {DIM}(none yet — waiting for track confirmation){RESET}")
    for ft in fused:
        color = THREAT_COLORS.get(ft.threat_level, RESET)
        flags = []
        if ft.is_stealth_candidate:
            flags.append("STEALTH")
        if ft.is_hypersonic_candidate:
            flags.append("HYPERSONIC")
        if ft.is_chaff_candidate:
            flags.append("CHAFF")
        if ft.is_decoy_candidate:
            flags.append("DECOY")
        if ft.has_quantum_confirmation:
            flags.append("QI")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        sensors = []
        if ft.radar_track:
            sensors.append("R")
        if ft.thermal_track:
            sensors.append("T")
        if ft.quantum_radar_track:
            sensors.append("Q")
        if ft.camera_track:
            sensors.append("C")
        sensor_str = "+".join(sensors) if sensors else "?"
        rng = f"{ft.range_m:.0f}m" if ft.range_m else "---"
        az = f"{ft.azimuth_deg:.1f}deg" if ft.azimuth_deg is not None else "---"
        temp = f" {ft.temperature_k:.0f}K" if ft.temperature_k else ""

        print(f"    {color}{ft.threat_level:>8s}{RESET}  "
              f"[{sensor_str:>5s}]  "
              f"r={rng:>8s}  az={az:>8s}{temp}"
              f"  q={ft.fusion_quality:.2f}"
              f"{flag_str}")
    print()

    # Weather
    w = env.weather
    if w:
        print(f"  {DIM}Weather: rain {w.rain_rate_mm_h:.0f} mm/h, "
              f"vis {w.visibility_km:.0f} km, "
              f"cloud {w.cloud_cover_pct:.0f}%, "
              f"humidity {w.humidity_pct:.0f}%{RESET}")


# ---------------------------------------------------------------------------
# Web dashboard feed
# ---------------------------------------------------------------------------

def _snapshot_from_state(
    radar_mgr, thermal_mgr, quantum_mgr,
    fused: list[EnhancedFusedTrack],
    step: int, elapsed: float,
    detect_ms: float = 0.0, track_ms: float = 0.0,
    fusion_ms: float = 0.0,
) -> "StateSnapshot":
    from sentinel.ui.web.state_buffer import StateSnapshot

    # Serialize radar tracks
    radar_list = []
    for rt in radar_mgr.active_tracks:
        radar_list.append({
            "track_id": rt.track_id,
            "state": rt.state.value if hasattr(rt.state, 'value') else str(rt.state),
            "range_m": float(rt.range_m),
            "azimuth_deg": float(rt.azimuth_deg),
            "velocity_mps": float(np.linalg.norm(rt.velocity)) if rt.velocity is not None else 0.0,
            "score": float(rt.score),
        })

    # Serialize thermal tracks
    thermal_list = []
    for tt in thermal_mgr.active_tracks:
        thermal_list.append({
            "track_id": tt.track_id,
            "state": tt.state.value if hasattr(tt.state, 'value') else str(tt.state),
            "azimuth_deg": float(tt.azimuth_deg),
            "temperature_k": float(tt.temperature_k) if tt.temperature_k else None,
            "score": float(tt.score),
        })

    # Serialize fused tracks
    enhanced_list = []
    for ft in fused:
        enhanced_list.append(ft.to_dict())

    # Threat counts
    threat_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for ft in fused:
        if ft.threat_level in threat_counts:
            threat_counts[ft.threat_level] += 1

    system_status = {
        "fps": round(UPDATE_HZ, 1),
        "track_count": len(radar_mgr.active_tracks),
        "confirmed_count": len(radar_mgr.confirmed_tracks),
        "detection_count": len(radar_mgr.active_tracks) + len(thermal_mgr.active_tracks),
        "uptime": elapsed,
        "camera_connected": False,  # No camera in this demo
        "radar_track_count": len(radar_mgr.active_tracks),
        "thermal_track_count": len(thermal_mgr.active_tracks),
        "fused_track_count": len(fused),
        "sensor_health": {
            "camera": {"enabled": False, "error_count": 0},
            "radar": {"enabled": True, "error_count": 0},
            "thermal": {"enabled": True, "error_count": 0},
            "quantum_radar": {"enabled": True, "error_count": 0},
            "multifreq_radar": {"enabled": True, "error_count": 0},
            "fusion": {"enabled": True, "error_count": 0},
        },
        "detect_ms": round(detect_ms, 1),
        "track_ms": round(track_ms, 1),
        "radar_ms": round(detect_ms + track_ms, 1),
        "fusion_ms": round(fusion_ms, 1),
        "render_ms": 0.1,
        "threat_counts": threat_counts,
    }

    return StateSnapshot(
        timestamp=time.time(),
        system_status=system_status,
        camera_tracks=[],
        radar_tracks=radar_list,
        thermal_tracks=thermal_list,
        fused_tracks=[],
        enhanced_fused_tracks=enhanced_list,
        hud_frame_jpeg=None,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL full-feature demo")
    parser.add_argument("--sim", action="store_true",
                        help="Use deterministic SimClock (fast replay, reproducible)")
    parser.add_argument("--steps", type=int, default=0,
                        help="Max steps in --sim mode (0 = unlimited)")
    args = parser.parse_args()

    use_sim_clock = args.sim
    max_steps = args.steps

    # Create clock
    if use_sim_clock:
        clock: Clock = SimClock(start_epoch=1_000_000.0)
    else:
        clock = SystemClock()

    # Check for web dependencies
    try:
        import uvicorn
        from sentinel.ui.web.server import create_app
        from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot
        has_web = True
    except ImportError:
        has_web = False

    # Build environment + targets (deploy_time aligned with clock)
    env = _build_environment(deploy_time=clock.now())
    radar_targets, thermal_targets = _build_targets()

    # Create simulators (all share the same clock)
    mf_sim = MultiFreqRadarSimulator(
        MultiFreqRadarConfig(
            bands=[RadarBand.VHF, RadarBand.UHF, RadarBand.L_BAND, RadarBand.S_BAND, RadarBand.X_BAND],
            max_range_m=50000.0,
            fov_deg=120.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.001,
            targets=radar_targets,
            environment=env,
        ),
        seed=42,
        clock=clock,
    )
    th_sim = ThermalSimulator(
        ThermalSimConfig(
            fov_deg=60.0,
            max_range_m=50000.0,
            detection_probability=0.95,
            false_alarm_rate=0.001,
            targets=thermal_targets,
            environment=env,
        ),
        seed=43,
        clock=clock,
    )
    qi_sim = QuantumRadarSimulator(
        QuantumRadarConfig(
            max_range_m=50000.0,
            squeeze_param_r=0.5,
            n_modes=50000,
            targets=radar_targets,  # Reuses same targets
            false_alarm_rate=0.0,
            environment=env,
        ),
        seed=44,
        clock=clock,
    )

    # Create track managers
    radar_mgr = RadarTrackManager(_radar_cfg())
    thermal_mgr = ThermalTrackManager(_thermal_cfg())
    quantum_mgr = RadarTrackManager(_radar_cfg())

    # Correlator + Fusion
    correlator = MultiFreqCorrelator(
        range_gate_m=200.0,
        azimuth_gate_deg=5.0,
        stealth_rcs_variation_db=15.0,
    )
    fusion = MultiSensorFusion(
        azimuth_gate_deg=10.0,
        thermal_azimuth_gate_deg=8.0,
    )

    # Web dashboard (optional)
    buf = None
    if has_web:
        buf = StateBuffer()
        app = create_app(buf, update_hz=UPDATE_HZ, video_fps=0)
        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs=dict(app=app, host="0.0.0.0", port=PORT, log_level="warning"),
            daemon=True,
        )
        server_thread.start()

    # Connect simulators
    mf_sim.connect()
    th_sim.connect()
    qi_sim.connect()

    _print_header()
    if not has_web:
        print(f"  {DIM}(Web dashboard unavailable — install with: pip install -e '.[web]'){RESET}\n")
    if use_sim_clock:
        mode_str = f"SimClock (epoch={clock.now():.0f})"
        if max_steps:
            mode_str += f", max {max_steps} steps"
        print(f"  {BOLD}Mode:{RESET} {mode_str}\n")

    step = 0
    t0 = time.monotonic()

    try:
        while True:
            step += 1

            # Advance sim clock before reading frames
            if isinstance(clock, SimClock):
                clock.step(STEP_INTERVAL)
            elapsed = clock.elapsed()

            # --- Detection phase (timed) ---
            t_detect_start = time.perf_counter()

            mf_frame = mf_sim.read_frame()
            mf_dets = multifreq_radar_frame_to_detections(mf_frame)

            th_frame = th_sim.read_frame()
            th_dets = thermal_frame_to_detections(th_frame)

            qi_frame = qi_sim.read_frame()
            qi_dets = quantum_radar_frame_to_detections(qi_frame)

            detect_ms = (time.perf_counter() - t_detect_start) * 1000.0

            # --- Tracking phase (timed) ---
            t_track_start = time.perf_counter()

            radar_mgr.step(mf_dets)
            thermal_mgr.step(th_dets)
            quantum_mgr.step(qi_dets)

            track_ms = (time.perf_counter() - t_track_start) * 1000.0

            # --- Correlation + Fusion (timed) ---
            t_fusion_start = time.perf_counter()

            correlated, _ = correlator.correlate(mf_dets)
            fused = fusion.fuse(
                camera_tracks=[],
                radar_tracks=radar_mgr.confirmed_tracks,
                thermal_tracks=thermal_mgr.confirmed_tracks,
                correlated_detections=correlated or None,
                quantum_radar_tracks=quantum_mgr.confirmed_tracks,
            )

            fusion_ms = (time.perf_counter() - t_fusion_start) * 1000.0

            # --- Console output ---
            _print_step(
                step, elapsed,
                mf_dets, th_dets, qi_dets,
                radar_mgr, thermal_mgr, quantum_mgr,
                correlated, fused, env,
                clock=clock,
            )

            # --- Web dashboard ---
            if buf is not None:
                snapshot = _snapshot_from_state(
                    radar_mgr, thermal_mgr, quantum_mgr,
                    fused, step, elapsed,
                    detect_ms=detect_ms,
                    track_ms=track_ms,
                    fusion_ms=fusion_ms,
                )
                buf.update(snapshot)

            # Pace the loop (real-time only; sim mode runs at full speed)
            if not use_sim_clock:
                time.sleep(STEP_INTERVAL)

            # Max steps in sim mode
            if max_steps and step >= max_steps:
                print(f"\n  {BOLD}Reached {max_steps} steps — stopping.{RESET}")
                break

    except KeyboardInterrupt:
        print(f"\n\n  {BOLD}Shutting down...{RESET}")
    finally:
        mf_sim.disconnect()
        th_sim.disconnect()
        qi_sim.disconnect()
        wall_time = time.monotonic() - t0
        print(f"  {DIM}Ran {step} steps in {wall_time:.1f}s wall-clock "
              f"(sim elapsed: {clock.elapsed():.1f}s){RESET}")
        print(f"  Done.\n")


if __name__ == "__main__":
    main()
