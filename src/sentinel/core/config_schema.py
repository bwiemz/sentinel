"""Pydantic schema for SENTINEL configuration validation.

Mirrors the YAML structure in config/default.yaml. Used when
``validate=True`` is passed to ``SentinelConfig.load()``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Leaf / shared models
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    name: str = "SENTINEL"
    version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    graceful_degradation: bool = True
    max_sensor_errors: int = Field(default=10, gt=0)
    validate_config: bool = False
    log_file: str | None = None
    log_json: bool = False


class CameraConfig(BaseModel):
    enabled: bool = True
    source: int | str = 0
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    fps: int = Field(default=30, gt=0)
    buffer_size: int = Field(default=1, ge=1)
    backend: str = "auto"


class RadarNoiseConfig(BaseModel):
    range_m: float = Field(default=5.0, ge=0)
    azimuth_deg: float = Field(default=1.0, ge=0)
    velocity_mps: float = Field(default=0.5, ge=0)
    rcs_dbsm: float = Field(default=2.0, ge=0)
    false_alarm_rate: float = Field(default=0.01, ge=0, le=1)
    detection_probability: float = Field(default=0.9, ge=0, le=1)
    range_dependent: bool = False
    use_snr_pd: bool = False


class RadarScenarioTarget(BaseModel):
    id: str
    position: list[float]
    velocity: list[float]
    rcs_dbsm: float = 10.0
    class_name: str = "unknown"
    target_type: str | None = None
    mach: float | None = None


class RadarScenarioConfig(BaseModel):
    targets: list[RadarScenarioTarget] = Field(default_factory=list)


class RadarConfig(BaseModel):
    enabled: bool = True
    mode: str = "simulator"
    scan_rate_hz: float = Field(default=10.0, gt=0)
    max_range_m: float = Field(default=10000.0, gt=0)
    fov_deg: float = Field(default=120.0, gt=0, le=360)
    noise: RadarNoiseConfig = Field(default_factory=RadarNoiseConfig)
    scenario: RadarScenarioConfig = Field(default_factory=RadarScenarioConfig)


class BandNoiseEntry(BaseModel):
    range_std_m: float = Field(default=10.0, ge=0)
    azimuth_std_deg: float = Field(default=2.0, ge=0)


class MultiFreqNoiseConfig(BaseModel):
    false_alarm_rate: float = Field(default=0.01, ge=0, le=1)
    detection_probability: float = Field(default=0.9, ge=0, le=1)


class MultiFreqRadarConfig(BaseModel):
    enabled: bool = False
    mode: str = "simulator"
    scan_rate_hz: float = Field(default=10.0, gt=0)
    max_range_m: float = Field(default=20000.0, gt=0)
    fov_deg: float = Field(default=120.0, gt=0, le=360)
    bands: list[str] = Field(default_factory=lambda: ["vhf", "uhf", "l_band", "s_band", "x_band"])
    band_noise: dict[str, BandNoiseEntry] = Field(default_factory=dict)
    noise: MultiFreqNoiseConfig = Field(default_factory=MultiFreqNoiseConfig)
    scenario: RadarScenarioConfig = Field(default_factory=RadarScenarioConfig)


class ThermalNoiseConfig(BaseModel):
    detection_probability: float = Field(default=0.95, ge=0, le=1)
    false_alarm_rate: float = Field(default=0.5, ge=0, le=1)
    min_contrast_k: float = Field(default=10.0, ge=0)


class ThermalConfig(BaseModel):
    enabled: bool = False
    mode: str = "simulator"
    frame_rate_hz: float = Field(default=30.0, gt=0)
    fov_deg: float = Field(default=60.0, gt=0, le=360)
    max_range_m: float = Field(default=50000.0, gt=0)
    bands: list[str] = Field(default_factory=lambda: ["mwir", "lwir"])
    noise: ThermalNoiseConfig = Field(default_factory=ThermalNoiseConfig)
    scenario: RadarScenarioConfig = Field(default_factory=RadarScenarioConfig)


class QuantumRadarNoiseConfig(BaseModel):
    range_std_m: float = Field(default=8.0, ge=0)
    azimuth_std_deg: float = Field(default=1.5, ge=0)
    false_alarm_rate: float = Field(default=0.005, ge=0, le=1)


class QuantumRadarConfig(BaseModel):
    enabled: bool = False
    mode: str = "simulator"
    scan_rate_hz: float = Field(default=10.0, gt=0)
    max_range_m: float = Field(default=15000.0, gt=0)
    fov_deg: float = Field(default=120.0, gt=0, le=360)
    freq_hz: float = Field(default=10.0e9, gt=0)
    squeeze_param_r: float = Field(default=0.1, gt=0)
    n_modes: int = Field(default=10000, gt=0)
    antenna_gain_dbi: float = 30.0
    receiver_type: Literal["opa", "sfg", "phase_conjugate", "optimal"] = "opa"
    ambient_temp_k: float = Field(default=290.0, gt=0)
    noise: QuantumRadarNoiseConfig = Field(default_factory=QuantumRadarNoiseConfig)
    scenario: RadarScenarioConfig = Field(default_factory=RadarScenarioConfig)


class SensorsConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)
    multifreq_radar: MultiFreqRadarConfig = Field(default_factory=MultiFreqRadarConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    quantum_radar: QuantumRadarConfig = Field(default_factory=QuantumRadarConfig)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    model: str = "yolov8n.pt"
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    device: str = "auto"
    classes: list[int] | None = None
    max_detections: int = Field(default=100, gt=0)
    image_size: int = Field(default=640, gt=0)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class IMMConfig(BaseModel):
    transition_prob: float = Field(default=0.98, ge=0, le=1)


class FilterConfig(BaseModel):
    type: Literal["kf", "ekf", "ca", "imm"] = "kf"
    dt: float = Field(default=0.033, gt=0)
    process_noise_std: float = Field(default=1.0, gt=0)
    measurement_noise_std: float = Field(default=10.0, gt=0)
    imm: IMMConfig = Field(default_factory=IMMConfig)


class AssociationConfig(BaseModel):
    method: Literal["hungarian"] = "hungarian"
    gate_threshold: float = Field(default=9.21, gt=0)
    iou_weight: float = Field(default=0.5, ge=0, le=1)
    mahalanobis_weight: float = Field(default=0.5, ge=0, le=1)
    cascaded: bool = False


class TrackManagementConfig(BaseModel):
    confirm_hits: int = Field(default=3, gt=0)
    confirm_window: int = Field(default=5, gt=0)
    max_coast_frames: int = Field(default=15, gt=0)
    max_tracks: int = Field(default=100, gt=0)
    tentative_delete_misses: int = Field(default=3, gt=0)
    confirmed_coast_misses: int = Field(default=5, gt=0)
    coast_reconfirm_hits: int = Field(default=2, gt=0)


class RadarMeasNoiseConfig(BaseModel):
    range_m: float = Field(default=5.0, ge=0)
    azimuth_deg: float = Field(default=1.0, ge=0)


class RadarFilterConfig(BaseModel):
    type: Literal["ekf", "ca", "imm"] = "ekf"
    dt: float = Field(default=0.1, gt=0)
    process_noise_std: float = Field(default=1.0, gt=0)
    use_3d: bool = False
    use_doppler: bool = False
    elevation_noise_deg: float = Field(default=1.0, ge=0)
    measurement_noise: RadarMeasNoiseConfig = Field(default_factory=RadarMeasNoiseConfig)


class RadarAssociationConfig(BaseModel):
    method: Literal["hungarian"] = "hungarian"
    gate_threshold: float = Field(default=9.21, gt=0)
    velocity_gate_mps: float | None = None
    cascaded: bool = False


class RadarTrackMgmtConfig(BaseModel):
    confirm_hits: int = Field(default=3, gt=0)
    max_coast_frames: int = Field(default=5, gt=0)
    max_tracks: int = Field(default=50, gt=0)


class RadarTrackingConfig(BaseModel):
    filter: RadarFilterConfig = Field(default_factory=RadarFilterConfig)
    association: RadarAssociationConfig = Field(default_factory=RadarAssociationConfig)
    track_management: RadarTrackMgmtConfig = Field(default_factory=RadarTrackMgmtConfig)


class ThermalFilterConfig(BaseModel):
    type: str = "bearing_ekf"
    dt: float = Field(default=0.033, gt=0)
    assumed_initial_range_m: float = Field(default=10000.0, gt=0)


class ThermalAssociationConfig(BaseModel):
    method: Literal["hungarian"] = "hungarian"
    gate_threshold: float = Field(default=6.635, gt=0)


class ThermalTrackMgmtConfig(BaseModel):
    confirm_hits: int = Field(default=3, gt=0)
    max_coast_frames: int = Field(default=10, gt=0)
    max_tracks: int = Field(default=50, gt=0)


class ThermalTrackingConfig(BaseModel):
    filter: ThermalFilterConfig = Field(default_factory=ThermalFilterConfig)
    association: ThermalAssociationConfig = Field(default_factory=ThermalAssociationConfig)
    track_management: ThermalTrackMgmtConfig = Field(default_factory=ThermalTrackMgmtConfig)


class QuantumRadarTrackingConfig(BaseModel):
    filter: RadarFilterConfig = Field(default_factory=lambda: RadarFilterConfig(type="ekf", dt=0.1))
    association: RadarAssociationConfig = Field(default_factory=RadarAssociationConfig)
    track_management: RadarTrackMgmtConfig = Field(default_factory=RadarTrackMgmtConfig)


class TrackingConfig(BaseModel):
    filter: FilterConfig = Field(default_factory=FilterConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    track_management: TrackManagementConfig = Field(default_factory=TrackManagementConfig)
    radar: RadarTrackingConfig = Field(default_factory=RadarTrackingConfig)
    thermal: ThermalTrackingConfig = Field(default_factory=ThermalTrackingConfig)
    quantum_radar: QuantumRadarTrackingConfig = Field(default_factory=QuantumRadarTrackingConfig)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class MultiFreqCorrelationConfig(BaseModel):
    range_gate_m: float = Field(default=500.0, gt=0)
    azimuth_gate_deg: float = Field(default=3.0, gt=0)


class ThreatClassificationConfig(BaseModel):
    hypersonic_temp_threshold_k: float = Field(default=1500.0, gt=0)
    stealth_rcs_variation_db: float = Field(default=15.0, gt=0)


class FusionConfig(BaseModel):
    enabled: bool = True
    camera_hfov_deg: float = Field(default=60.0, gt=0)
    azimuth_gate_deg: float = Field(default=5.0, gt=0)
    thermal_azimuth_gate_deg: float = Field(default=3.0, gt=0)
    min_fusion_quality: float = Field(default=0.3, ge=0, le=1)
    multifreq_correlation: MultiFreqCorrelationConfig = Field(default_factory=MultiFreqCorrelationConfig)
    threat_classification: ThreatClassificationConfig = Field(default_factory=ThreatClassificationConfig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class ScanlineConfig(BaseModel):
    enabled: bool = True
    spacing: int = Field(default=3, gt=0)
    alpha: float = Field(default=0.03, ge=0, le=1)


class HUDConfig(BaseModel):
    enabled: bool = True
    display: bool = True
    overlay_alpha: float = Field(default=0.85, ge=0, le=1)
    colors: dict[str, list[int]] = Field(default_factory=dict)
    scanline: ScanlineConfig = Field(default_factory=ScanlineConfig)


class WebConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8080, gt=0, le=65535)
    track_update_hz: int = Field(default=10, gt=0)
    video_stream_fps: int = Field(default=15, gt=0)


class UIConfig(BaseModel):
    hud: HUDConfig = Field(default_factory=HUDConfig)
    web: WebConfig = Field(default_factory=WebConfig)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class SentinelRootConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    sensors: SensorsConfig = Field(default_factory=SensorsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    model_config = {"extra": "allow"}


class SentinelConfigSchema(BaseModel):
    """Top-level wrapper matching YAML root key ``sentinel:``."""

    sentinel: SentinelRootConfig

    model_config = {"extra": "allow"}


def validate_config(cfg_dict: dict) -> SentinelConfigSchema:
    """Validate a raw config dict (e.g. from OmegaConf) against the schema.

    Raises ``pydantic.ValidationError`` on invalid config.
    """
    return SentinelConfigSchema.model_validate(cfg_dict)
