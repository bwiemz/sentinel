# SENTINEL

**Sensor-Enhanced Networked Tracking, Intelligence, and Engagement Library**

Real-time multi-sensor tracking system that fuses camera, multi-frequency radar, thermal imaging, and quantum illumination radar to detect and classify targets -- including stealth aircraft and hypersonic vehicles.

## License & Usage

This repository is publicly available **for learning and reference only**.

- You may **view and study** the code for educational purposes.
- You may **NOT copy, reuse, redistribute, or modify** any part of this
  repository without explicit permission.
- **Commercial use is strictly prohibited** without contacting the author
  and receiving written authorization.

If you would like permission to use any part of this project, please
contact me directly.

## Architecture

```
Camera (30 Hz) ──> YOLOv8 ──> TrackManager (Kalman Filter) ──────────────────┐
                                                                               │
Multi-Freq Radar ──> MultiFreqCorrelator ──> RadarTrackManager (EKF) ─────────┤
  (VHF/UHF/L/S/X)     (cross-band grouping)                                   │
                                                                               ├──> MultiSensorFusion ──> HUD
Thermal FLIR ──> ThermalTrackManager (Bearing-Only EKF) ──────────────────────┤    (threat classification)
  (MWIR/LWIR)                                                                  │
                                                                               │
Quantum Radar ──> RadarTrackManager (EKF) ────────────────────────────────────┘
  (QI X-band)       (reuses existing EKF)
```

Four independent sensor paths run at their native rates. Tracks are fused by angular correspondence or statistical (Mahalanobis) distance, with optional temporal alignment to a common epoch. Data association uses either the Hungarian algorithm (hard assignment) or JPDA (soft probabilistic assignment). Targets are classified by threat level. The quantum radar reuses the existing radar EKF pipeline -- quantum advantage manifests in detection probability, not measurement format.

## Key Capabilities

- **Object detection**: YOLOv8 on camera frames (USB, RTSP, or video file)
- **Kalman filter tracking**: Predict-update cycle with automatic track lifecycle (tentative -> confirmed -> coasting -> deleted)
- **Data association**: Hungarian algorithm (globally optimal hard assignment) or JPDA (Joint Probabilistic Data Association -- soft probabilistic assignment with spread-of-innovations covariance for dense/crossing target scenarios)
- **Radar tracking**: Extended Kalman Filter in polar coordinates with simulated radar returns
- **Multi-frequency radar**: 5-band simulation (VHF/UHF/L/S/X) with frequency-dependent RCS, plasma sheath attenuation modeling
- **Thermal imaging**: Passive FLIR simulation (MWIR/LWIR) with bearing-only tracking -- no range, unaffected by plasma
- **Stealth detection**: Radar-absorbing materials absorb X-band but not VHF/UHF. Cross-band RCS variation flags stealth candidates
- **Hypersonic detection**: Mach 5+ targets create extreme thermal signatures (>1500K) that thermal sensors always detect, even when radar is degraded by plasma sheath
- **Quantum illumination radar**: Entangled microwave photon pairs (TMSV state) for enhanced stealth detection at X-band where classical radar fails. OPA/SFG/Phase-Conjugate receiver models. 6 dB SNR advantage over classical at same energy budget
- **Track quality monitoring**: Normalized Innovation Squared (NIS) metrics with rolling-window filter consistency monitor (nominal/over-confident/under-confident/diverged health states)
- **Temporal alignment**: Predict all tracks to a common reference epoch before fusion using constant-velocity propagation with CWNA process noise -- eliminates 171m error at Mach 5 per 100ms sensor time offset
- **Statistical fusion**: Mahalanobis track-to-track distance replaces angular heuristics for cross-sensor correlation, with camera-to-world coordinate projection
- **Threat classification**: CRITICAL (hypersonic, quantum-confirmed stealth), HIGH (stealth, quantum-only), MEDIUM (multi-sensor conventional), LOW (single-sensor, chaff, decoy)
- **Military HUD**: Real-time overlay with track boxes, velocity vectors, targeting reticle, radar/thermal/quantum blips, threat badges, and stealth/hypersonic alert banners
- **Web dashboard**: Real-time browser-based monitoring via FastAPI/WebSocket -- tactical PPI radar scope, sortable track table, threat cards, per-stage latency bars, MJPEG HUD video feed. Military dark theme, vanilla JS (no build step)
- **Terrain & environment**: 2D elevation grid with ray-marching line-of-sight, ITU-R P.676/P.838 atmospheric propagation (frequency-dependent), weather effects (rain, fog, cloud cover), surface/rain clutter models. All effects optional and backward-compatible
- **Electronic warfare**: Noise jamming (J/S ratio, SNR degradation, burn-through range), deceptive jamming (RGPO false targets), chaff clouds (high uniform RCS, drag deceleration, time decay), expendable decoys (radar-only, no IR). ECCM countermeasures: sidelobe blanking, frequency agility, burn-through mode, QI ECCM (entangled photons resist noise jamming -- 6 dB advantage). Fusion-layer discrimination: chaff flagged by cross-band RCS uniformity, decoys flagged by radar-only/no-thermal signature. All EW effects default OFF and backward-compatible

### Web Dashboard

![SENTINEL Web Dashboard](docs/images/web-dashboard.png)

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run

```bash
# Default: USB webcam + radar simulator
sentinel

# Custom config
sentinel --config config/default.yaml

# Video file input
sentinel --source path/to/video.mp4

# Override YOLO model and device
sentinel --model yolov8s.pt --device cuda:0

# Enable web dashboard (requires web extras)
pip install -e ".[web]"
sentinel --web
```

### Test

```bash
pytest tests/ -v
```

1070 tests covering all subsystems.

## Configuration

All settings live in `config/default.yaml` under the `sentinel:` namespace. Key sections:

| Section | Description |
|---------|-------------|
| `sensors.camera` | Camera source, resolution, FPS |
| `sensors.radar` | Single-frequency radar simulator (Phase 4) |
| `sensors.multifreq_radar` | Multi-band radar with per-band noise profiles |
| `sensors.thermal` | Thermal FLIR simulator with MWIR/LWIR bands |
| `sensors.quantum_radar` | Quantum illumination radar (QI X-band, TMSV source) |
| `environment` | Terrain masking, atmospheric propagation, weather effects, clutter, electronic warfare |
| `detection` | YOLOv8 model, confidence, device |
| `tracking` | Kalman filter params, association gating, track lifecycle |
| `tracking.association` | Association method (`hungarian`/`jpda`), JPDA parameters |
| `tracking.track_quality` | NIS monitoring (enabled, window size) |
| `tracking.radar` | EKF params for radar tracking |
| `tracking.thermal` | Bearing-only EKF params for thermal tracking |
| `tracking.quantum_radar` | EKF params for quantum radar tracking |
| `fusion` | Azimuth gates, temporal alignment, statistical distance, threat thresholds |
| `ui.hud` | HUD colors, overlay alpha, scanline effect |
| `ui.web` | Web dashboard: host, port, update rate, video FPS |

Enable multi-freq radar, thermal, and quantum radar by setting `enabled: true` in their respective sections. When disabled, the system runs in camera-only or camera+single-radar mode (backward compatible). Environment effects (terrain masking, atmospheric propagation, weather, clutter, electronic warfare) are also disabled by default -- enable each independently in the `environment` section. The web dashboard is also disabled by default -- enable with `ui.web.enabled: true` or the `--web` CLI flag.

## Project Structure

```
sentinel/
  config/
    default.yaml              # All configuration
  src/sentinel/
    core/
      types.py                # Detection, TrackState, SensorType, enums
      pipeline.py             # Main orchestrator loop
      config.py               # Config loader (OmegaConf)
      clock.py                # Frame timer, system clock
      bus.py                  # Event bus
    sensors/
      base.py                 # AbstractSensor interface
      camera.py               # OpenCV camera adapter
      radar_sim.py            # Single-freq radar simulator + MultiFreqRadarTarget
      multifreq_radar_sim.py  # 5-band radar simulator
      thermal_sim.py          # Thermal FLIR simulator (bearing-only)
      quantum_radar_sim.py    # Quantum illumination radar simulator
      environment.py          # Terrain, atmosphere, weather, clutter models
      ew.py                   # Electronic warfare: jamming, chaff, decoys, ECCM
      physics.py              # RCS profiles, plasma sheath, thermal signatures, QI physics
      frame.py                # SensorFrame container
    detection/
      yolo.py                 # YOLOv8 detector wrapper
    tracking/
      filters.py              # KalmanFilter, ExtendedKalmanFilter, BearingOnlyEKF
      base_track.py           # TrackBase (shared state machine, M-of-N, scoring, quality monitor)
      track.py                # Camera Track (KF-based)
      track_manager.py        # Camera track lifecycle manager
      association.py          # Hungarian associator (IoU + Mahalanobis)
      jpda.py                 # JPDA associators (camera, radar, thermal)
      cost_functions.py       # Cost matrix computation + track-to-track Mahalanobis
      track_quality.py        # NIS/NEES metrics, FilterConsistencyMonitor
      radar_track.py          # Radar Track (EKF-based, polar coords)
      radar_association.py    # Radar Hungarian associator
      radar_track_manager.py  # Radar track lifecycle
      thermal_track.py        # Thermal Track (bearing-only EKF)
      thermal_association.py  # Thermal bearing-based associator
      thermal_track_manager.py # Thermal track lifecycle
    fusion/
      track_fusion.py         # Camera-radar fusion (angular + statistical)
      temporal_alignment.py   # Predict tracks to common epoch (CV + CWNA)
      multifreq_correlator.py # Cross-band detection grouping
      multi_sensor_fusion.py  # 4-sensor fusion + threat classification
    ui/hud/
      renderer.py             # HUD compositor
      elements.py             # Drawing primitives (boxes, blips, alerts)
      styles.py               # Colors, fonts, visual config
    ui/web/
      state_buffer.py         # Thread-safe pipeline→web state transfer
      mjpeg.py                # JPEG frame encoder
      server.py               # FastAPI app (REST, WebSocket, MJPEG)
      bridge.py               # WebDashboard lifecycle manager
      static/                 # Frontend: HTML, CSS, JS (vanilla, no build step)
  tests/
    unit/test_*.py            # 989 unit tests
    integration/test_*.py     # 37 integration tests
    scenarios/test_*.py       # 44 scenario validation tests
```

## Physics Models

### Frequency-Dependent RCS (Stealth)

Stealth materials (RAM) are optimized for X-band absorption. At lower frequencies, the RCS increases dramatically:

| Band | Stealth RCS offset vs X-band |
|------|------------------------------|
| VHF | +25 dB |
| UHF | +22 dB |
| L-band | +15 dB |
| S-band | +7 dB |
| X-band | 0 dB (baseline) |

A target with -20 dBsm at X-band appears as +5 dBsm at VHF -- easily detectable. The `MultiFreqCorrelator` flags targets with >15 dB RCS variation across bands as stealth candidates.

### Plasma Sheath (Hypersonic)

At Mach 5+, aerodynamic heating ionizes the surrounding air. This plasma sheath attenuates radar returns, with higher frequencies affected more:

```
attenuation_db = base_atten * (1 + freq_factor) * mach_factor^2
```

Where `freq_factor` ranges from 0.3 (VHF) to 2.0 (X-band). VHF penetrates plasma better, but all radar is degraded.

### Thermal Signatures

Thermal imaging is **not affected by plasma** -- it's a passive sensor measuring emitted IR radiation. Hypersonic leading edges reach 1650-2760°C from aerodynamic heating:

```
T_surface = T_ambient * (1 + 0.85 * 0.2 * Mach^2)
```

At Mach 5, surface temperatures exceed 1500K, making thermal detection the primary sensor for hypersonic threats.

### Quantum Illumination (Stealth Detection)

Quantum illumination uses entangled microwave photon pairs (Two-Mode Squeezed Vacuum state) to achieve a detection advantage over classical radar. The signal photon is transmitted toward the target while the idler is retained at the receiver; joint measurement exploits quantum correlations.

**TMSV source**: Mean signal photons per mode: `N_S = sinh²(r)`, where r is the squeeze parameter. QI advantage is maximal when N_S << 1 (r ≈ 0.1 → N_S ≈ 0.01).

**Error exponents** (asymptotic detection performance):

```
QI:        β_QI = M · N_S / N_B
Classical: β_C  = M · N_S² / (4 · N_B)
Advantage: β_QI / β_C = 4 / N_S  (6 dB when N_S << 1)
```

Where M = signal-idler mode pairs per pulse, N_B = thermal background photons.

**Receiver models** (fraction of theoretical 6 dB advantage achieved):

| Receiver | Advantage | Status |
|----------|-----------|--------|
| OPA (Optical Parametric Amplifier) | 3 dB | Experimentally demonstrated |
| SFG (Sum-Frequency Generation) | 6 dB | Theoretical, requires ideal conversion |
| Phase Conjugate | 3 dB | Demonstrated in optical domain |
| Optimal (Helstrom bound) | 6 dB | Theoretical limit |

**Where QI excels**: Low N_S + high N_B = exactly the stealth detection scenario. A stealth target with -20 dBsm at X-band is nearly invisible to classical X-band radar, but QI's 6 dB SNR advantage can push it above the detection threshold.

**Entanglement fidelity**: `F = η·N_S / (η·N_S + (1-η)·N_B + 1)` -- measures how well quantum correlations survive the round-trip channel loss η.

### Combined Detection Probability

Multi-sensor detection probability across N independent sensors:

```
P_total = 1 - product(1 - P_i)
```

## Development Phases

| Phase | Description | Commit |
|-------|-------------|--------|
| 1 | Project scaffold, camera pipeline, YOLOv8 detection | `ee70e40` |
| 2 | Kalman filter tracking, track lifecycle, military HUD | `4a5b839` |
| 3 | Hungarian algorithm for globally optimal data association | `8436e97` |
| 4 | Radar sensor fusion with Extended Kalman Filter | `15eb377` |
| 5 | Multi-frequency radar + thermal imaging for stealth/hypersonic detection | `5ac63ad` |
| 6 | Quantum illumination radar for enhanced stealth detection | -- |
| 7 | Algorithm optimization: CA-KF, IMM, 3D/Doppler EKF, cascaded association | `e3d3947` |
| 8 | Production hardening: error handling, validation, logging, CI/CD | `ca46515` |
| 9 | Association & fusion integrity: JPDA, temporal alignment, statistical distance, NIS monitoring | `636d08b` |
| 10 | Real-time web dashboard: FastAPI, WebSocket, PPI radar scope, threat cards, MJPEG feed | `86e769c` |
| 11 | Scenario validation: stealth ingress, hypersonic raid, multi-target swarm, mixed threat | `3287f3c` |
| 12 | Terrain & environment: terrain masking, atmospheric propagation, weather effects, clutter models | `c8ba6fd` |
| 13 | Electronic warfare: noise/deceptive jamming, chaff, decoys, ECCM, QI jamming resistance | -- |

## Dependencies

- Python >= 3.10
- NumPy, SciPy (linear algebra, optimization)
- OpenCV (camera I/O, HUD rendering)
- Ultralytics (YOLOv8)
- OmegaConf + PyYAML (configuration)
- FastAPI, uvicorn, websockets (web dashboard -- optional `[web]` extras)

## License

MIT
