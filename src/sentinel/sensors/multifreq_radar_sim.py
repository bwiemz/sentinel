"""Multi-frequency radar simulator.

Generates detections across multiple radar frequency bands for each target,
with frequency-dependent RCS, plasma sheath effects, and per-band noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.clock import SystemClock
from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.sensors.base import AbstractSensor
from sentinel.sensors.frame import SensorFrame
from sentinel.sensors.radar_sim import MultiFreqRadarTarget
from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    polar_to_cartesian,
)

logger = logging.getLogger(__name__)


@dataclass
class BandNoiseConfig:
    """Noise parameters for a specific radar frequency band.

    Lower frequencies (VHF/UHF) have worse spatial resolution, hence
    higher noise in range and azimuth.
    """

    noise_range_m: float = 5.0
    noise_azimuth_deg: float = 1.0
    noise_velocity_mps: float = 0.5
    noise_rcs_dbsm: float = 2.0


# Default noise by band (lower freq = worse resolution)
DEFAULT_BAND_NOISE: dict[RadarBand, BandNoiseConfig] = {
    RadarBand.VHF: BandNoiseConfig(50.0, 5.0, 2.0, 3.0),
    RadarBand.UHF: BandNoiseConfig(30.0, 3.0, 1.5, 2.5),
    RadarBand.L_BAND: BandNoiseConfig(15.0, 2.0, 1.0, 2.0),
    RadarBand.S_BAND: BandNoiseConfig(10.0, 1.5, 0.8, 2.0),
    RadarBand.X_BAND: BandNoiseConfig(5.0, 1.0, 0.5, 2.0),
}


@dataclass
class MultiFreqRadarConfig:
    """Configuration for the multi-frequency radar simulator."""

    bands: list[RadarBand] = field(default_factory=lambda: list(RadarBand))
    scan_rate_hz: float = 10.0
    max_range_m: float = 50000.0
    fov_deg: float = 120.0
    band_configs: dict[RadarBand, BandNoiseConfig] = field(default_factory=dict)
    false_alarm_rate: float = 0.01
    base_detection_probability: float = 0.9
    range_dependent_noise: bool = False
    targets: list[MultiFreqRadarTarget] = field(default_factory=list)

    def noise_for_band(self, band: RadarBand) -> BandNoiseConfig:
        """Get noise config for a band, falling back to defaults."""
        return self.band_configs.get(band, DEFAULT_BAND_NOISE.get(band, BandNoiseConfig()))

    @classmethod
    def from_omegaconf(cls, cfg) -> MultiFreqRadarConfig:
        """Create from OmegaConf DictConfig."""
        # Parse bands
        band_names = cfg.get("bands", ["x_band"])
        bands = [RadarBand(b) for b in band_names]

        # Parse per-band noise
        band_cfgs: dict[RadarBand, BandNoiseConfig] = {}
        raw_bc = cfg.get("band_configs", {})
        for bname, bcfg in raw_bc.items():
            band = RadarBand(bname)
            band_cfgs[band] = BandNoiseConfig(
                noise_range_m=bcfg.get("noise_range_m", 5.0),
                noise_azimuth_deg=bcfg.get("noise_azimuth_deg", 1.0),
                noise_velocity_mps=bcfg.get("noise_velocity_mps", 0.5),
                noise_rcs_dbsm=bcfg.get("noise_rcs_dbsm", 2.0),
            )

        # Parse targets
        targets: list[MultiFreqRadarTarget] = []
        scenario = cfg.get("scenario", {})
        for t in scenario.get("targets", []):
            tt_str = t.get("target_type", "conventional")
            targets.append(
                MultiFreqRadarTarget(
                    target_id=t.get("id", "TGT"),
                    position=np.array(t.get("position", [0, 0]), dtype=float),
                    velocity=np.array(t.get("velocity", [0, 0]), dtype=float),
                    rcs_dbsm=t.get("rcs_dbsm", 10.0),
                    class_name=t.get("class_name", "unknown"),
                    target_type=TargetType(tt_str),
                    mach=t.get("mach", 0.0),
                )
            )

        noise = cfg.get("noise", {})
        return cls(
            bands=bands,
            scan_rate_hz=cfg.get("scan_rate_hz", 10.0),
            max_range_m=cfg.get("max_range_m", 50000.0),
            fov_deg=cfg.get("fov_deg", 120.0),
            band_configs=band_cfgs,
            false_alarm_rate=noise.get("false_alarm_rate", cfg.get("false_alarm_rate", 0.01)),
            base_detection_probability=noise.get("detection_probability", cfg.get("base_detection_probability", 0.9)),
            range_dependent_noise=noise.get("range_dependent", False),
            targets=targets,
        )


class MultiFreqRadarSimulator(AbstractSensor):
    """Multi-frequency radar simulator.

    Generates detections at each configured frequency band for each target.
    Each detection dict includes a ``frequency_band`` field.

    SensorFrame.data is list[dict], each with keys:
    range_m, azimuth_deg, velocity_mps, rcs_dbsm, frequency_band, target_id
    """

    def __init__(self, config: MultiFreqRadarConfig, seed: int | None = None):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._clock = SystemClock()
        self._connected = False
        self._start_time = 0.0
        self._scan_count = 0
        self._fov_half_rad = azimuth_deg_to_rad(config.fov_deg / 2)

    def connect(self) -> bool:
        self._connected = True
        self._start_time = self._clock.now()
        self._scan_count = 0
        bands_str = ", ".join(b.value for b in self._config.bands)
        logger.info(
            "Multi-freq radar connected: %d targets, bands=[%s], %.0f Hz",
            len(self._config.targets),
            bands_str,
            self._config.scan_rate_hz,
        )
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Multi-freq radar disconnected after %d scans", self._scan_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self) -> SensorFrame | None:
        """Generate one multi-band radar scan."""
        if not self._connected:
            return None

        t = self._clock.now() - self._start_time
        all_detections: list[dict] = []

        for band in self._config.bands:
            all_detections.extend(self._generate_band_detections(t, band))

        self._scan_count += 1
        return SensorFrame(
            data=all_detections,
            timestamp=self._clock.now(),
            sensor_type=SensorType.RADAR,
            frame_number=self._scan_count,
            metadata={
                "scan_count": self._scan_count,
                "bands": [b.value for b in self._config.bands],
                "target_count": len(self._config.targets),
            },
        )

    def _generate_band_detections(self, t: float, band: RadarBand) -> list[dict]:
        """Generate detections for a single frequency band."""
        noise_cfg = self._config.noise_for_band(band)
        detections: list[dict] = []

        for target in self._config.targets:
            pos = target.position_at(t)
            r, az = cartesian_to_polar(pos[0], pos[1])

            # Range and FOV check
            if r > self._config.max_range_m or abs(az) > self._fov_half_rad:
                continue

            # Frequency-dependent detection probability
            pd = target.detection_probability_at_band(
                band,
                self._config.base_detection_probability,
                t,
            )
            if self._rng.rand() > pd:
                continue

            # Range-dependent noise scaling
            range_factor = 1.0 + (r / self._config.max_range_m) ** 2 if self._config.range_dependent_noise else 1.0

            # Add band-specific noise (scaled by range factor)
            noisy_range = max(0.0, r + self._rng.randn() * noise_cfg.noise_range_m * range_factor)
            noisy_az_deg = azimuth_rad_to_deg(az) + self._rng.randn() * noise_cfg.noise_azimuth_deg * range_factor
            radial_vel = self._compute_radial_velocity(pos, target.velocity)
            noisy_vel = radial_vel + self._rng.randn() * noise_cfg.noise_velocity_mps
            effective_rcs = target.rcs_at_band(band)
            noisy_rcs = effective_rcs + self._rng.randn() * noise_cfg.noise_rcs_dbsm

            detections.append(
                {
                    "range_m": noisy_range,
                    "azimuth_deg": noisy_az_deg,
                    "velocity_mps": noisy_vel,
                    "rcs_dbsm": noisy_rcs,
                    "frequency_band": band.value,
                    "target_id": target.target_id,
                }
            )

        # False alarms for this band
        detections.extend(self._generate_false_alarms(band))
        return detections

    def _compute_radial_velocity(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> float:
        r = np.linalg.norm(position)
        if r < 1e-6:
            return 0.0
        return float(np.dot(velocity, position / r))

    def _generate_false_alarms(self, band: RadarBand) -> list[dict]:
        n_fa = self._rng.poisson(self._config.false_alarm_rate)
        alarms = []
        fov_half_deg = self._config.fov_deg / 2
        for _ in range(n_fa):
            alarms.append(
                {
                    "range_m": self._rng.uniform(10.0, self._config.max_range_m),
                    "azimuth_deg": self._rng.uniform(-fov_half_deg, fov_half_deg),
                    "velocity_mps": self._rng.randn() * 5.0,
                    "rcs_dbsm": self._rng.uniform(-10, 20),
                    "frequency_band": band.value,
                }
            )
        return alarms


def multifreq_radar_frame_to_detections(frame: SensorFrame) -> list[Detection]:
    """Convert a multi-freq radar SensorFrame into Detection objects."""
    detections = []
    for d in frame.data:
        az_rad = azimuth_deg_to_rad(d["azimuth_deg"])
        pos = polar_to_cartesian(d["range_m"], az_rad)
        detections.append(
            Detection(
                sensor_type=SensorType.RADAR,
                timestamp=frame.timestamp,
                range_m=d["range_m"],
                azimuth_deg=d["azimuth_deg"],
                velocity_mps=d["velocity_mps"],
                rcs_dbsm=d["rcs_dbsm"],
                radar_band=d.get("frequency_band"),
                position_3d=np.array([pos[0], pos[1], 0.0]),
            )
        )
    return detections
