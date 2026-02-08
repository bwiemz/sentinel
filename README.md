# SENTINEL

**Sensor-Enhanced Networked Tracking, Intelligence, and Engagement Library**

Real-time multi-sensor tracking system that fuses camera, multi-frequency radar, and thermal imaging to detect and classify targets -- including stealth aircraft and hypersonic vehicles.

## Architecture

```
Camera (30 Hz) ──> YOLOv8 ──> TrackManager (Kalman Filter) ──────────────────┐
                                                                               │
Multi-Freq Radar ──> MultiFreqCorrelator ──> RadarTrackManager (EKF) ─────────┼──> MultiSensorFusion ──> HUD
  (VHF/UHF/L/S/X)     (cross-band grouping)                                   │    (threat classification)
                                                                               │
Thermal FLIR ──> ThermalTrackManager (Bearing-Only EKF) ──────────────────────┘
  (MWIR/LWIR)
```

Three independent sensor paths run at their native rates. Tracks are fused by angular correspondence (Hungarian algorithm) and classified by threat level.

## Key Capabilities

- **Object detection**: YOLOv8 on camera frames (USB, RTSP, or video file)
- **Kalman filter tracking**: Predict-update cycle with automatic track lifecycle (tentative -> confirmed -> coasting -> deleted)
- **Globally optimal association**: Hungarian algorithm on combined IoU + Mahalanobis cost matrices
- **Radar tracking**: Extended Kalman Filter in polar coordinates with simulated radar returns
- **Multi-frequency radar**: 5-band simulation (VHF/UHF/L/S/X) with frequency-dependent RCS, plasma sheath attenuation modeling
- **Thermal imaging**: Passive FLIR simulation (MWIR/LWIR) with bearing-only tracking -- no range, unaffected by plasma
- **Stealth detection**: Radar-absorbing materials absorb X-band but not VHF/UHF. Cross-band RCS variation flags stealth candidates
- **Hypersonic detection**: Mach 5+ targets create extreme thermal signatures (>1500K) that thermal sensors always detect, even when radar is degraded by plasma sheath
- **Threat classification**: CRITICAL (hypersonic), HIGH (stealth), MEDIUM (multi-sensor conventional), LOW (single-sensor)
- **Military HUD**: Real-time overlay with track boxes, velocity vectors, targeting reticle, radar/thermal blips, threat badges, and stealth/hypersonic alert banners

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
```

### Test

```bash
pytest tests/unit/ -v
```

333 tests covering all subsystems.

## Configuration

All settings live in `config/default.yaml` under the `sentinel:` namespace. Key sections:

| Section | Description |
|---------|-------------|
| `sensors.camera` | Camera source, resolution, FPS |
| `sensors.radar` | Single-frequency radar simulator (Phase 4) |
| `sensors.multifreq_radar` | Multi-band radar with per-band noise profiles |
| `sensors.thermal` | Thermal FLIR simulator with MWIR/LWIR bands |
| `detection` | YOLOv8 model, confidence, device |
| `tracking` | Kalman filter params, association gating, track lifecycle |
| `tracking.radar` | EKF params for radar tracking |
| `tracking.thermal` | Bearing-only EKF params for thermal tracking |
| `fusion` | Azimuth gates, multi-freq correlation, threat thresholds |
| `ui.hud` | HUD colors, overlay alpha, scanline effect |

Enable multi-freq radar and thermal by setting `enabled: true` in their respective sections. When disabled, the system runs in camera-only or camera+single-radar mode (backward compatible).

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
      physics.py              # RCS profiles, plasma sheath, thermal signatures
      frame.py                # SensorFrame container
    detection/
      yolo.py                 # YOLOv8 detector wrapper
    tracking/
      filters.py              # KalmanFilter, ExtendedKalmanFilter, BearingOnlyEKF
      track.py                # Camera Track (KF-based)
      track_manager.py        # Camera track lifecycle manager
      association.py          # Hungarian associator (IoU + Mahalanobis)
      cost_functions.py       # Cost matrix computation
      radar_track.py          # Radar Track (EKF-based, polar coords)
      radar_association.py    # Radar Hungarian associator
      radar_track_manager.py  # Radar track lifecycle
      thermal_track.py        # Thermal Track (bearing-only EKF)
      thermal_association.py  # Thermal bearing-based associator
      thermal_track_manager.py # Thermal track lifecycle
    fusion/
      track_fusion.py         # Camera-radar fusion (Phase 4)
      multifreq_correlator.py # Cross-band detection grouping
      multi_sensor_fusion.py  # 3-sensor fusion + threat classification
    ui/hud/
      renderer.py             # HUD compositor
      elements.py             # Drawing primitives (boxes, blips, alerts)
      styles.py               # Colors, fonts, visual config
  tests/unit/
    test_*.py                 # 333 unit tests
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

## Dependencies

- Python >= 3.10
- NumPy, SciPy (linear algebra, optimization)
- OpenCV (camera I/O, HUD rendering)
- Ultralytics (YOLOv8)
- OmegaConf + PyYAML (configuration)

## License

MIT
