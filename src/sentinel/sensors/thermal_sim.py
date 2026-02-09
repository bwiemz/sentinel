"""Thermal imaging sensor simulator.

Generates bearing-only detections based on thermal contrast.
Thermal sensors cannot measure range -- they provide azimuth, elevation,
temperature, and intensity. Key advantage: NOT affected by plasma sheath.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.clock import SystemClock
from sentinel.core.types import Detection, SensorType, TargetType, ThermalBand
from sentinel.sensors.base import AbstractSensor
from sentinel.sensors.frame import SensorFrame
from sentinel.sensors.environment import EnvironmentModel
from sentinel.sensors.physics import ThermalSignature
from sentinel.utils.coords import azimuth_rad_to_deg, cartesian_to_polar

logger = logging.getLogger(__name__)


@dataclass
class ThermalTarget:
    """A target visible to thermal sensors."""

    target_id: str
    position: np.ndarray  # [x, y] meters at t=0
    velocity: np.ndarray  # [vx, vy] m/s
    target_type: TargetType = TargetType.CONVENTIONAL
    mach: float = 0.0  # 0 = compute from velocity
    class_name: str = "unknown"
    _signature: ThermalSignature | None = field(default=None, repr=False)

    def __post_init__(self):
        if self._signature is None:
            self._signature = ThermalSignature(target_type=self.target_type)

    def position_at(self, t: float) -> np.ndarray:
        return self.position + self.velocity * t

    def effective_mach(self, speed_of_sound: float = 343.0) -> float:
        if self.mach > 0:
            return self.mach
        return float(np.linalg.norm(self.velocity)) / speed_of_sound

    def temperature_at(self, t: float = 0.0) -> float:
        """Peak temperature in Kelvin."""
        m = self.effective_mach()
        return self._signature.peak_temperature_k(m)

    def thermal_contrast(self, ambient_k: float = 280.0) -> float:
        m = self.effective_mach()
        return self._signature.thermal_contrast(m)

    def band_intensity(self, band: ThermalBand) -> float:
        m = self.effective_mach()
        return self._signature.band_intensity(m, band)


@dataclass
class ThermalSimConfig:
    """Configuration for the thermal imaging simulator."""

    frame_rate_hz: float = 30.0
    bands: list[ThermalBand] = field(default_factory=lambda: [ThermalBand.MWIR, ThermalBand.LWIR])
    fov_deg: float = 20.0
    max_range_m: float = 50000.0
    noise_azimuth_deg: float = 0.1
    noise_elevation_deg: float = 0.1
    noise_temperature_k: float = 5.0
    ambient_temperature_k: float = 280.0
    min_contrast_k: float = 10.0
    detection_probability: float = 0.95
    false_alarm_rate: float = 0.005
    targets: list[ThermalTarget] = field(default_factory=list)
    environment: EnvironmentModel | None = None

    @classmethod
    def from_omegaconf(cls, cfg) -> ThermalSimConfig:
        """Create from OmegaConf DictConfig."""
        noise = cfg.get("noise", {})
        band_names = cfg.get("bands", ["mwir", "lwir"])
        bands = [ThermalBand(b) for b in band_names]

        targets: list[ThermalTarget] = []
        scenario = cfg.get("scenario", {})
        for t in scenario.get("targets", []):
            tt_str = t.get("target_type", "conventional")
            targets.append(
                ThermalTarget(
                    target_id=t.get("id", "TGT"),
                    position=np.array(t.get("position", [0, 0]), dtype=float),
                    velocity=np.array(t.get("velocity", [0, 0]), dtype=float),
                    target_type=TargetType(tt_str),
                    mach=t.get("mach", 0.0),
                    class_name=t.get("class_name", "unknown"),
                )
            )

        return cls(
            frame_rate_hz=cfg.get("frame_rate_hz", 30.0),
            bands=bands,
            fov_deg=cfg.get("fov_deg", 20.0),
            max_range_m=cfg.get("max_range_m", 50000.0),
            noise_azimuth_deg=noise.get("azimuth_deg", 0.1),
            noise_elevation_deg=noise.get("elevation_deg", 0.1),
            noise_temperature_k=noise.get("temperature_k", 5.0),
            ambient_temperature_k=cfg.get("ambient_temperature_k", 280.0),
            min_contrast_k=cfg.get("min_contrast_k", 10.0),
            detection_probability=cfg.get("detection_probability", 0.95),
            false_alarm_rate=cfg.get("false_alarm_rate", 0.005),
            targets=targets,
        )


class ThermalSimulator(AbstractSensor):
    """Simulated thermal imaging sensor (FLIR-style).

    Produces bearing-only detections: azimuth, elevation, temperature, intensity.
    Does NOT produce range measurements.

    SensorFrame.data is list[dict] with keys:
    azimuth_deg, elevation_deg, temperature_k, thermal_band, intensity, target_id
    """

    def __init__(self, config: ThermalSimConfig, seed: int | None = None):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._clock = SystemClock()
        self._connected = False
        self._start_time = 0.0
        self._frame_count = 0
        self._fov_half_deg = config.fov_deg / 2

    def connect(self) -> bool:
        self._connected = True
        self._start_time = self._clock.now()
        self._frame_count = 0
        bands_str = ", ".join(b.value for b in self._config.bands)
        logger.info(
            "Thermal simulator connected: %d targets, bands=[%s], %.0f Hz",
            len(self._config.targets),
            bands_str,
            self._config.frame_rate_hz,
        )
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Thermal simulator disconnected after %d frames", self._frame_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self) -> SensorFrame | None:
        """Generate one thermal scan."""
        if not self._connected:
            return None

        t = self._clock.now() - self._start_time
        all_detections: list[dict] = []

        for band in self._config.bands:
            all_detections.extend(self._generate_band_detections(t, band))

        self._frame_count += 1
        return SensorFrame(
            data=all_detections,
            timestamp=self._clock.now(),
            sensor_type=SensorType.THERMAL,
            frame_number=self._frame_count,
            metadata={
                "frame_count": self._frame_count,
                "bands": [b.value for b in self._config.bands],
                "target_count": len(self._config.targets),
            },
        )

    def _generate_band_detections(self, t: float, band: ThermalBand) -> list[dict]:
        """Generate thermal detections for one IR band."""
        detections: list[dict] = []

        for target in self._config.targets:
            pos = target.position_at(t)
            r, az = cartesian_to_polar(pos[0], pos[1])

            # Range check (with visibility-based reduction)
            env = self._config.environment
            effective_max_range = self._config.max_range_m
            if env:
                effective_max_range = env.effective_thermal_max_range(effective_max_range)
            if r > effective_max_range:
                continue

            # Terrain masking
            if env and not env.is_target_visible(pos[0], pos[1]):
                continue

            # FOV check (azimuth only, thermal has narrow FOV)
            az_deg = azimuth_rad_to_deg(az)
            if abs(az_deg) > self._fov_half_deg:
                continue

            # Thermal contrast check
            temp = target.temperature_at(t)
            contrast = temp - self._config.ambient_temperature_k
            # Cloud cover reduces effective contrast
            if env and env.use_weather_effects:
                from sentinel.sensors.environment import weather_thermal_contrast_factor
                contrast *= weather_thermal_contrast_factor(env.weather)
            if contrast < self._config.min_contrast_k:
                continue

            # Band intensity -- skip if below threshold for this band
            intensity = target.band_intensity(band)
            if intensity < 0.05:
                continue

            # Detection probability (with atmospheric transmission)
            pd = self._config.detection_probability
            if env:
                pd *= env.thermal_detection_factor(band, r)
            if self._rng.rand() > pd:
                continue

            # Add noise (bearing-only: no range noise)
            noisy_az = az_deg + self._rng.randn() * self._config.noise_azimuth_deg
            noisy_el = 0.0 + self._rng.randn() * self._config.noise_elevation_deg  # Flat scenario
            noisy_temp = temp + self._rng.randn() * self._config.noise_temperature_k

            detections.append(
                {
                    "azimuth_deg": noisy_az,
                    "elevation_deg": noisy_el,
                    "temperature_k": max(0.0, noisy_temp),
                    "thermal_band": band.value,
                    "intensity": intensity,
                    "target_id": target.target_id,
                }
            )

        # False alarms
        detections.extend(self._generate_false_alarms(band))

        # EW decoy thermal returns (most decoys have no IR → typically empty)
        env = self._config.environment
        if env and env.use_ew_effects:
            import time as _time
            ew_thermal = env.get_ew_thermal_returns(
                sensor_pos=np.array([0.0, 0.0]),
                t=_time.time(),
            )
            for ret in ew_thermal:
                detections.append({
                    "azimuth_deg": ret["azimuth_deg"],
                    "elevation_deg": 0.0,
                    "temperature_k": ret.get("temperature_k", 300.0),
                    "thermal_band": band.value,
                    "intensity": 0.5,
                    "is_ew_generated": True,
                    "ew_source_id": ret.get("source_id"),
                })

        return detections

    def _generate_false_alarms(self, band: ThermalBand) -> list[dict]:
        n_fa = self._rng.poisson(self._config.false_alarm_rate)
        alarms = []
        for _ in range(n_fa):
            alarms.append(
                {
                    "azimuth_deg": self._rng.uniform(-self._fov_half_deg, self._fov_half_deg),
                    "elevation_deg": self._rng.randn() * 2.0,
                    "temperature_k": self._config.ambient_temperature_k + self._rng.uniform(5, 50),
                    "thermal_band": band.value,
                    "intensity": self._rng.uniform(0.05, 0.3),
                }
            )
        return alarms


def thermal_frame_to_detections(frame: SensorFrame) -> list[Detection]:
    """Convert a thermal SensorFrame into Detection objects.

    Thermal detections have azimuth and temperature but NO range.
    """
    detections = []
    for d in frame.data:
        detections.append(
            Detection(
                sensor_type=SensorType.THERMAL,
                timestamp=frame.timestamp,
                azimuth_deg=d["azimuth_deg"],
                elevation_deg=d.get("elevation_deg", 0.0),
                temperature_k=d["temperature_k"],
                thermal_band=d.get("thermal_band"),
                intensity=d.get("intensity"),
            )
        )
    return detections
