"""Quantum illumination radar simulator.

Simulates a quantum radar that uses entangled microwave photon pairs
(Two-Mode Squeezed Vacuum state) for target detection. The quantum
advantage manifests as improved detection probability for low-RCS
targets compared to classical radar at the same energy budget.

The simulator produces standard radar measurements (range, azimuth,
velocity, RCS) so the output can be tracked by the existing
RadarTrackManager EKF pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.clock import Clock, SystemClock
from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.sensors.base import AbstractSensor
from sentinel.sensors.frame import SensorFrame
from sentinel.sensors.physics import (
    ReceiverType,
    channel_transmissivity,
    classical_practical_pd,
    entanglement_fidelity,
    qi_practical_pd,
    qi_snr_advantage_db,
    receiver_efficiency,
    thermal_background_photons,
    tmsv_mean_photons,
)
from sentinel.sensors.environment import EnvironmentModel
from sentinel.sensors.radar_sim import MultiFreqRadarTarget
from sentinel.utils.coords import (
    azimuth_deg_to_rad,
    azimuth_rad_to_deg,
    cartesian_to_polar,
    polar_to_cartesian,
)
from sentinel.utils.geo_context import GeoContext

logger = logging.getLogger(__name__)

# Speed of light for wavelength calculation
_C = 2.99792458e8


@dataclass
class QuantumRadarConfig:
    """Configuration for the quantum illumination radar simulator."""

    freq_hz: float = 10.0e9  # Operating frequency (10 GHz X-band)
    squeeze_param_r: float = 0.1  # TMSV squeeze parameter (N_S ~ 0.01)
    n_modes: int = 10000  # Signal-idler mode pairs per pulse
    antenna_gain_dbi: float = 30.0  # Antenna gain in dBi
    receiver_type: ReceiverType = ReceiverType.OPA
    ambient_temp_k: float = 290.0  # Background temperature

    scan_rate_hz: float = 10.0
    max_range_m: float = 15000.0
    fov_deg: float = 120.0

    noise_range_m: float = 8.0
    noise_azimuth_deg: float = 1.5
    noise_velocity_mps: float = 0.5
    noise_rcs_dbsm: float = 2.0
    false_alarm_rate: float = 0.005  # Lower than classical (quantum correlation)
    range_dependent_noise: bool = False

    targets: list[MultiFreqRadarTarget] = field(default_factory=list)
    environment: EnvironmentModel | None = None
    geo_context: GeoContext | None = None

    @property
    def wavelength_m(self) -> float:
        return _C / self.freq_hz

    @property
    def antenna_gain_linear(self) -> float:
        return 10.0 ** (self.antenna_gain_dbi / 10.0)

    @property
    def n_signal(self) -> float:
        return tmsv_mean_photons(self.squeeze_param_r)

    @property
    def n_background(self) -> float:
        return thermal_background_photons(self.freq_hz, self.ambient_temp_k)

    @property
    def receiver_eff(self) -> float:
        return receiver_efficiency(self.receiver_type)

    @classmethod
    def from_omegaconf(
        cls, cfg, geo_context: GeoContext | None = None,
    ) -> QuantumRadarConfig:
        """Create from OmegaConf DictConfig (sentinel.sensors.quantum_radar)."""
        noise = cfg.get("noise", {})

        # Parse receiver type
        rt_str = cfg.get("receiver_type", "opa").lower()
        rt_map = {
            "opa": ReceiverType.OPA,
            "sfg": ReceiverType.SFG,
            "phase_conjugate": ReceiverType.PHASE_CONJUGATE,
            "optimal": ReceiverType.OPTIMAL,
        }
        receiver = rt_map.get(rt_str, ReceiverType.OPA)

        # Parse targets (reuses MultiFreqRadarTarget for RCS profiles)
        targets = []
        scenario = cfg.get("scenario", {})
        for t in scenario.get("targets", []):
            tt_str = t.get("target_type", "conventional").lower()
            tt_map = {
                "conventional": TargetType.CONVENTIONAL,
                "stealth": TargetType.STEALTH,
                "hypersonic": TargetType.HYPERSONIC,
            }
            pos_geo = t.get("position_geo", None)
            if pos_geo is not None and geo_context is not None:
                alt = pos_geo[2] if len(pos_geo) > 2 else 0.0
                xy = geo_context.target_geodetic_to_xy(pos_geo[0], pos_geo[1], alt)
                position = xy
            else:
                position = np.array(t.get("position", [0, 0]), dtype=float)
            targets.append(
                MultiFreqRadarTarget(
                    target_id=t.get("id", "QI-TGT"),
                    position=position,
                    velocity=np.array(t.get("velocity", [0, 0]), dtype=float),
                    rcs_dbsm=t.get("rcs_dbsm", 10.0),
                    class_name=t.get("class_name", "unknown"),
                    target_type=tt_map.get(tt_str, TargetType.CONVENTIONAL),
                    mach=t.get("mach", 0.0),
                )
            )

        return cls(
            freq_hz=cfg.get("freq_hz", 10.0e9),
            squeeze_param_r=cfg.get("squeeze_param_r", 0.1),
            n_modes=int(cfg.get("n_modes", 10000)),
            antenna_gain_dbi=cfg.get("antenna_gain_dbi", 30.0),
            receiver_type=receiver,
            ambient_temp_k=cfg.get("ambient_temp_k", 290.0),
            scan_rate_hz=cfg.get("scan_rate_hz", 10.0),
            max_range_m=cfg.get("max_range_m", 15000.0),
            fov_deg=cfg.get("fov_deg", 120.0),
            noise_range_m=noise.get("range_std_m", 8.0),
            noise_azimuth_deg=noise.get("azimuth_std_deg", 1.5),
            noise_velocity_mps=noise.get("velocity_mps", 0.5),
            noise_rcs_dbsm=noise.get("rcs_dbsm", 2.0),
            false_alarm_rate=noise.get("false_alarm_rate", 0.005),
            range_dependent_noise=noise.get("range_dependent", False),
            targets=targets,
            geo_context=geo_context,
        )


class QuantumRadarSimulator(AbstractSensor):
    """Quantum illumination radar simulator.

    Generates synthetic radar detections using quantum illumination
    physics. For each target, computes both QI and classical detection
    probabilities, using the QI probability for the detection dice roll.

    The output format is identical to classical radar (range, azimuth,
    velocity, RCS) plus quantum metadata (QI advantage, entanglement
    fidelity), allowing reuse of the existing RadarTrackManager.
    """

    def __init__(self, config: QuantumRadarConfig, seed: int | None = None,
                 clock: Clock | None = None,
                 geo_context: GeoContext | None = None):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._clock = clock if clock is not None else SystemClock()
        self._connected = False
        self._start_time = 0.0
        self._scan_count = 0
        self._fov_half_rad = azimuth_deg_to_rad(config.fov_deg / 2)
        self._geo_context = geo_context if geo_context is not None else config.geo_context

    def connect(self) -> bool:
        self._connected = True
        self._start_time = self._clock.now()
        self._scan_count = 0
        ns = self._config.n_signal
        nb = self._config.n_background
        adv = qi_snr_advantage_db(ns)
        logger.info(
            "Quantum radar connected: %d targets, N_S=%.4f, N_B=%.0f, QI advantage=%.1f dB, receiver=%s",
            len(self._config.targets),
            ns,
            nb,
            adv,
            self._config.receiver_type.value,
        )
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Quantum radar disconnected after %d scans", self._scan_count)

    def read_frame(self) -> SensorFrame | None:
        """Generate one quantum radar scan."""
        if not self._connected:
            return None

        t = self._clock.now() - self._start_time
        detections: list[dict] = []

        ns = self._config.n_signal
        nb = self._config.n_background
        recv_eff = self._config.receiver_eff

        for target in self._config.targets:
            pos = target.position_at(t)
            r, az = cartesian_to_polar(pos[0], pos[1])

            # Range and FOV gating
            if r > self._config.max_range_m or r < 1.0:
                continue
            if abs(az) > self._fov_half_rad:
                continue

            # Terrain masking
            env = self._config.environment
            if env and not env.is_target_visible(pos[0], pos[1]):
                continue

            # RCS at X-band (quantum radar operates at X-band)
            rcs_dbsm = target.rcs_at_band(RadarBand.X_BAND)
            rcs_m2 = 10.0 ** (rcs_dbsm / 10.0)

            # Channel transmissivity (for metadata/fidelity)
            eta = channel_transmissivity(
                rcs_m2=rcs_m2,
                antenna_gain=self._config.antenna_gain_linear,
                wavelength_m=self._config.wavelength_m,
                range_m=r,
            )

            # Practical detection probabilities (radar-equation SNR model)
            pd_qi = qi_practical_pd(
                rcs_m2=rcs_m2,
                range_m=r,
                n_signal=ns,
                receiver_eff=recv_eff,
                ref_range_m=self._config.max_range_m,
            )
            pd_cl = classical_practical_pd(
                rcs_m2=rcs_m2,
                range_m=r,
                ref_range_m=self._config.max_range_m,
            )

            # QI advantage in dB
            adv_db = qi_snr_advantage_db(ns)

            # Entanglement fidelity
            fidelity = entanglement_fidelity(eta, ns, nb)

            # Atmospheric loss reduces QI detection probability
            if env and env.use_atmospheric_propagation:
                snr_adj = env.radar_snr_adjustment_db(self._config.freq_hz, r)
                if snr_adj < 0:
                    pd_qi *= max(0.0, 10.0 ** (snr_adj / 20.0))

            # EW noise jamming (QI has inherent resistance via entanglement)
            if env and env.use_ew_effects:
                ew_adj = env.ew_snr_adjustment_db(r, self._config.freq_hz)
                # QI ECCM advantage: entangled photons resist noise jamming
                if env.ew is not None and env.ew.eccm.quantum_eccm:
                    qi_eccm_bonus = env.ew.eccm.quantum_eccm_advantage_db
                    ew_adj = min(0.0, ew_adj + qi_eccm_bonus)
                if ew_adj < 0:
                    pd_qi *= max(0.0, 10.0 ** (ew_adj / 20.0))

            # Detection roll uses QI probability
            if self._rng.rand() > pd_qi:
                continue

            # Range-dependent noise scaling
            range_factor = 1.0 + (r / self._config.max_range_m) ** 2 if self._config.range_dependent_noise else 1.0

            # Add measurement noise
            noisy_range = r + self._rng.randn() * self._config.noise_range_m * range_factor
            noisy_az_deg = azimuth_rad_to_deg(az) + (self._rng.randn() * self._config.noise_azimuth_deg * range_factor)
            radial_vel = self._compute_radial_velocity(pos, target.velocity)
            noisy_vel = radial_vel + self._rng.randn() * self._config.noise_velocity_mps
            noisy_rcs = rcs_dbsm + self._rng.randn() * self._config.noise_rcs_dbsm

            detections.append(
                {
                    "range_m": max(0.0, noisy_range),
                    "azimuth_deg": noisy_az_deg,
                    "velocity_mps": noisy_vel,
                    "rcs_dbsm": noisy_rcs,
                    "target_id": target.target_id,
                    # Quantum metadata
                    "qi_advantage_db": adv_db,
                    "entanglement_fidelity": fidelity,
                    "n_signal_photons": ns,
                    "receiver_type": self._config.receiver_type.value,
                    "pd_qi": pd_qi,
                    "pd_classical": pd_cl,
                    "transmissivity": eta,
                }
            )

        # False alarms (fewer than classical due to quantum correlation)
        detections.extend(self._generate_false_alarms())

        self._scan_count += 1
        return SensorFrame(
            data=detections,
            timestamp=self._clock.now(),
            sensor_type=SensorType.QUANTUM_RADAR,
            frame_number=self._scan_count,
            metadata={
                "scan_count": self._scan_count,
                "target_count": len(self._config.targets),
                "n_signal": ns,
                "n_background": nb,
                "receiver": self._config.receiver_type.value,
            },
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _compute_radial_velocity(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Compute radial (Doppler) velocity toward the radar."""
        r = np.linalg.norm(position)
        if r < 1e-6:
            return 0.0
        unit = position / r
        return float(np.dot(velocity, unit))

    def _generate_false_alarms(self) -> list[dict]:
        """Generate false alarm detections."""
        n_fa = self._rng.poisson(self._config.false_alarm_rate)
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
                    "qi_advantage_db": 0.0,
                    "entanglement_fidelity": 0.0,
                    "n_signal_photons": self._config.n_signal,
                    "receiver_type": self._config.receiver_type.value,
                    "pd_qi": 0.0,
                    "pd_classical": 0.0,
                    "transmissivity": 0.0,
                }
            )
        return alarms


def quantum_radar_frame_to_detections(frame: SensorFrame) -> list[Detection]:
    """Convert a quantum radar SensorFrame into Detection objects.

    Maps the standard radar fields plus QI metadata into Detection objects
    with sensor_type=QUANTUM_RADAR.
    """
    detections = []
    for d in frame.data:
        az_rad = azimuth_deg_to_rad(d["azimuth_deg"])
        pos = polar_to_cartesian(d["range_m"], az_rad)
        detections.append(
            Detection(
                sensor_type=SensorType.QUANTUM_RADAR,
                timestamp=frame.timestamp,
                range_m=d["range_m"],
                azimuth_deg=d["azimuth_deg"],
                velocity_mps=d["velocity_mps"],
                rcs_dbsm=d["rcs_dbsm"],
                position_3d=np.array([pos[0], pos[1], 0.0]),
                qi_advantage_db=d.get("qi_advantage_db"),
                entanglement_fidelity=d.get("entanglement_fidelity"),
                n_signal_photons=d.get("n_signal_photons"),
                receiver_type=d.get("receiver_type"),
                target_id=d.get("target_id"),
            )
        )
    return detections
