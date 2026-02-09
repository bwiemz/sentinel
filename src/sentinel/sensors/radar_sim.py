"""Radar simulator sensor adapter.

Generates synthetic radar detections from configurable scenario targets
with realistic noise, false alarms, and missed detections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.clock import Clock, SystemClock
from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.sensors.base import AbstractSensor
from sentinel.sensors.frame import SensorFrame
from sentinel.sensors.environment import EnvironmentModel, total_propagation_loss_db
from sentinel.sensors.physics import RCSProfile, _snr_to_pd, radar_snr
from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    polar_to_cartesian,
)

logger = logging.getLogger(__name__)


@dataclass
class RadarTarget:
    """A simulated radar target with constant-velocity trajectory.

    Attributes:
        target_id: Unique identifier.
        position: [x, y] in meters at t=0.
        velocity: [vx, vy] in m/s.
        rcs_dbsm: Radar cross section in dBsm.
        class_name: Optional label for the target type.
    """

    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    rcs_dbsm: float = 10.0
    class_name: str = "unknown"

    def position_at(self, t: float) -> np.ndarray:
        """Ground-truth position at time t seconds from start."""
        return self.position + self.velocity * t


@dataclass
class RadarSimConfig:
    """Configuration for the radar simulator."""

    scan_rate_hz: float = 10.0
    max_range_m: float = 10000.0
    fov_deg: float = 120.0
    noise_range_m: float = 5.0
    noise_azimuth_deg: float = 1.0
    noise_velocity_mps: float = 0.5
    noise_rcs_dbsm: float = 2.0
    false_alarm_rate: float = 0.01
    detection_probability: float = 0.9
    range_dependent_noise: bool = False
    use_snr_pd: bool = False
    targets: list[RadarTarget] = field(default_factory=list)
    environment: EnvironmentModel | None = None

    @classmethod
    def from_omegaconf(cls, cfg) -> RadarSimConfig:
        """Create from OmegaConf DictConfig (sentinel.sensors.radar section)."""
        noise = cfg.get("noise", {})
        targets = []
        scenario = cfg.get("scenario", {})
        for t in scenario.get("targets", []):
            targets.append(
                RadarTarget(
                    target_id=t.get("id", "TGT"),
                    position=np.array(t.get("position", [0, 0]), dtype=float),
                    velocity=np.array(t.get("velocity", [0, 0]), dtype=float),
                    rcs_dbsm=t.get("rcs_dbsm", 10.0),
                    class_name=t.get("class_name", "unknown"),
                )
            )
        return cls(
            scan_rate_hz=cfg.get("scan_rate_hz", 10.0),
            max_range_m=cfg.get("max_range_m", 10000.0),
            fov_deg=cfg.get("fov_deg", 120.0),
            noise_range_m=noise.get("range_m", 5.0),
            noise_azimuth_deg=noise.get("azimuth_deg", 1.0),
            noise_velocity_mps=noise.get("velocity_mps", 0.5),
            noise_rcs_dbsm=noise.get("rcs_dbsm", 2.0),
            false_alarm_rate=noise.get("false_alarm_rate", 0.01),
            detection_probability=noise.get("detection_probability", 0.9),
            range_dependent_noise=noise.get("range_dependent", False),
            use_snr_pd=noise.get("use_snr_pd", False),
            targets=targets,
        )


class RadarSimulator(AbstractSensor):
    """Synthetic radar sensor that generates detections from scenario targets.

    SensorFrame.data is a list[dict], where each dict has keys:
    range_m, azimuth_deg, velocity_mps, rcs_dbsm.

    Args:
        config: RadarSimConfig with noise parameters and target list.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, config: RadarSimConfig, seed: int | None = None,
                 clock: Clock | None = None):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._clock = clock if clock is not None else SystemClock()
        self._connected = False
        self._start_time = 0.0
        self._scan_count = 0
        self._fov_half_rad = azimuth_deg_to_rad(config.fov_deg / 2)

    def connect(self) -> bool:
        self._connected = True
        self._start_time = self._clock.now()
        self._scan_count = 0
        logger.info(
            "Radar simulator connected: %d targets, %.0f Hz, %.0f m range",
            len(self._config.targets),
            self._config.scan_rate_hz,
            self._config.max_range_m,
        )
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Radar simulator disconnected after %d scans", self._scan_count)

    def read_frame(self) -> SensorFrame | None:
        """Generate one radar scan with detections and false alarms."""
        if not self._connected:
            return None

        t = self._clock.now() - self._start_time
        detections: list[dict] = []

        # True target detections
        for target in self._config.targets:
            pos = target.position_at(t)
            r, az = cartesian_to_polar(pos[0], pos[1])

            # Check range and FOV
            if r > self._config.max_range_m:
                continue
            if abs(az) > self._fov_half_rad:
                continue

            # Terrain masking
            env = self._config.environment
            if env and not env.is_target_visible(pos[0], pos[1]):
                continue

            # Detection probability: SNR-based or flat
            if self._config.use_snr_pd:
                rcs_m2 = 10.0 ** (target.rcs_dbsm / 10.0)
                snr = radar_snr(rcs_m2, r, ref_range_m=self._config.max_range_m)
                if env:
                    snr += env.radar_snr_adjustment_db(10e9, r)
                    snr += env.ew_snr_adjustment_db(r, 10e9)
                pd = _snr_to_pd(snr)
            else:
                pd = self._config.detection_probability
                if env and env.use_atmospheric_propagation:
                    loss_db = total_propagation_loss_db(
                        10e9, r, env.weather.rain_rate_mm_h,
                        env.weather.temperature_k, env.weather.humidity_pct,
                    )
                    pd *= max(0.0, 10.0 ** (-loss_db / 20.0))
                if env and env.use_ew_effects:
                    ew_adj = env.ew_snr_adjustment_db(r, 10e9)
                    if ew_adj < 0:
                        pd *= max(0.0, 10.0 ** (ew_adj / 20.0))
            if self._rng.rand() > pd:
                continue

            # Range-dependent noise scaling: noise grows with (r / max_range)^2
            range_factor = 1.0 + (r / self._config.max_range_m) ** 2 if self._config.range_dependent_noise else 1.0

            # Add measurement noise
            noisy_range = r + self._rng.randn() * self._config.noise_range_m * range_factor
            noisy_az_deg = azimuth_rad_to_deg(az) + self._rng.randn() * self._config.noise_azimuth_deg * range_factor
            radial_vel = self._compute_radial_velocity(pos, target.velocity)
            noisy_vel = radial_vel + self._rng.randn() * self._config.noise_velocity_mps
            noisy_rcs = target.rcs_dbsm + self._rng.randn() * self._config.noise_rcs_dbsm

            detections.append(
                {
                    "range_m": max(0.0, noisy_range),
                    "azimuth_deg": noisy_az_deg,
                    "velocity_mps": noisy_vel,
                    "rcs_dbsm": noisy_rcs,
                    "target_id": target.target_id,
                }
            )

        # False alarms
        detections.extend(self._generate_false_alarms())

        # EW false targets (deceptive jammers, chaff, decoys)
        env = self._config.environment
        if env and env.use_ew_effects:
            ew_returns = env.get_ew_false_detections(
                sensor_pos=np.array([0.0, 0.0]),
                rng=np.random.default_rng(self._rng.randint(0, 2**31)),
                t=self._clock.now(),
            )
            for ew_ret in ew_returns:
                detections.append({
                    "range_m": ew_ret["range_m"],
                    "azimuth_deg": ew_ret["azimuth_deg"],
                    "velocity_mps": ew_ret.get("velocity_mps", 0.0),
                    "rcs_dbsm": ew_ret.get("rcs_dbsm", 0.0),
                    "is_ew_generated": True,
                    "ew_source_id": ew_ret.get("source_id"),
                })

        self._scan_count += 1
        return SensorFrame(
            data=detections,
            timestamp=self._clock.now(),
            sensor_type=SensorType.RADAR,
            frame_number=self._scan_count,
            metadata={"scan_count": self._scan_count, "target_count": len(self._config.targets)},
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def add_target(self, target: RadarTarget) -> None:
        self._config.targets.append(target)

    def remove_target(self, target_id: str) -> None:
        self._config.targets = [t for t in self._config.targets if t.target_id != target_id]

    def _compute_radial_velocity(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Compute radial (Doppler) velocity component toward the radar."""
        r = np.linalg.norm(position)
        if r < 1e-6:
            return 0.0
        unit = position / r
        return float(np.dot(velocity, unit))

    def _generate_false_alarms(self) -> list[dict]:
        """Generate false alarm detections uniformly in the FOV."""
        far = self._config.false_alarm_rate
        env = self._config.environment
        if env:
            far = env.effective_false_alarm_rate(far)
        n_fa = self._rng.poisson(far)
        alarms = []
        fov_half_deg = self._config.fov_deg / 2
        for _ in range(n_fa):
            r = self._rng.uniform(10.0, self._config.max_range_m)
            az = self._rng.uniform(-fov_half_deg, fov_half_deg)
            alarms.append(
                {
                    "range_m": r,
                    "azimuth_deg": az,
                    "velocity_mps": self._rng.randn() * 5.0,
                    "rcs_dbsm": self._rng.uniform(-10, 20),
                }
            )
        return alarms


@dataclass
class MultiFreqRadarTarget(RadarTarget):
    """Radar target with frequency-dependent RCS and speed-dependent plasma effects.

    Extends RadarTarget for backward compatibility. When used with the
    multi-frequency radar simulator, provides per-band RCS and plasma-adjusted
    detection probability.
    """

    target_type: TargetType = TargetType.CONVENTIONAL
    mach: float = 0.0  # 0 = compute from velocity at runtime

    def effective_mach(self, t: float = 0.0, speed_of_sound: float = 343.0) -> float:
        """Compute Mach number from velocity magnitude, or return preset."""
        if self.mach > 0:
            return self.mach
        speed = float(np.linalg.norm(self.velocity))
        return speed / speed_of_sound

    def rcs_at_band(self, band: RadarBand) -> float:
        """RCS in dBsm at the given frequency band."""
        profile = RCSProfile(x_band_dbsm=self.rcs_dbsm, target_type=self.target_type)
        return profile.rcs_at_band(band)

    def detection_probability_at_band(
        self,
        band: RadarBand,
        base_pd: float,
        t: float = 0.0,
    ) -> float:
        """Effective detection probability accounting for plasma attenuation."""
        from sentinel.sensors.physics import plasma_detection_factor

        m = self.effective_mach(t)
        factor = plasma_detection_factor(m, band)
        return base_pd * factor


def radar_frame_to_detections(frame: SensorFrame) -> list[Detection]:
    """Convert a radar SensorFrame into Detection objects."""
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
                position_3d=np.array([pos[0], pos[1], 0.0]),
            )
        )
    return detections
