"""Electronic warfare modeling: jamming, chaff, decoys, and ECCM.

All EW effects are optional and default OFF. When the EW model is ``None``
or ``use_ew_effects`` is ``False``, simulators behave identically to
pre-Phase-13 code (backward compatible).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import RadarBand
from sentinel.sensors.environment import BAND_CENTER_FREQ_HZ


# ===================================================================
# Noise Jamming
# ===================================================================


def jammer_to_signal_ratio_db(
    jammer_erp_w: float,
    jammer_range_m: float,
    jammer_bw_hz: float,
    radar_peak_power_w: float,
    radar_range_m: float,
    radar_bw_hz: float,
    radar_gain_db: float = 30.0,
    antenna_sidelobe_db: float = -20.0,
) -> float:
    """Compute Jammer-to-Signal ratio in dB.

    Positive J/S means the jammer dominates the radar receiver.

    J/S = (ERP_j · R_t^4 · B_r) / (P_t · G^2 · λ^2 · σ_ref · R_j^2 · B_j · (4π))
    Simplified to decouple from individual target RCS — models the *jamming
    environment* rather than per-target effect. For main-lobe jamming, the
    full antenna gain is used; for sidelobe, ``antenna_sidelobe_db`` is applied.
    """
    if jammer_range_m <= 0 or radar_range_m <= 0 or jammer_bw_hz <= 0 or radar_bw_hz <= 0:
        return -100.0
    if jammer_erp_w <= 0:
        return -100.0

    # J/S simplified: ERP_j * R_t^4 / (P_t * G^2 * R_j^2) * (B_r / B_j)
    gain_linear = 10.0 ** (radar_gain_db / 10.0)
    numerator = jammer_erp_w * radar_range_m ** 4 * radar_bw_hz
    denominator = radar_peak_power_w * gain_linear ** 2 * jammer_range_m ** 2 * jammer_bw_hz
    if denominator <= 0:
        return 100.0
    js_linear = numerator / denominator
    return 10.0 * math.log10(max(js_linear, 1e-30))


def noise_jamming_snr_reduction_db(js_ratio_db: float) -> float:
    """SNR reduction caused by noise jamming.

    SNR_jammed = SNR_clear / (1 + J/S_linear)
    Returns a positive dB value representing the loss.
    """
    if js_ratio_db < -50:
        return 0.0
    js_linear = 10.0 ** (js_ratio_db / 10.0)
    reduction = 10.0 * math.log10(1.0 + js_linear)
    return max(0.0, reduction)


def burn_through_range_m(
    radar_peak_power_w: float,
    radar_gain_db: float,
    rcs_m2: float,
    jammer_erp_w: float,
    jammer_range_m: float,
    jammer_bw_hz: float,
    radar_bw_hz: float,
    required_snr_db: float = 13.0,
) -> float:
    """Range inside which radar signal overcomes the jammer.

    Inside the burn-through range, the radar has enough SNR to detect
    the target despite the jammer.
    """
    if jammer_erp_w <= 0 or rcs_m2 <= 0:
        return float("inf")
    gain_linear = 10.0 ** (radar_gain_db / 10.0)
    snr_req_linear = 10.0 ** (required_snr_db / 10.0)

    # Burn-through condition: SNR = required_snr
    # P_t * G^2 * lambda^2 * sigma / ((4pi)^3 * R^4 * kTB) = snr_req * (1 + J/S)
    # Simplified: R_bt^4 = P_t * G^2 * sigma * R_j^2 * B_j / (snr_req * ERP_j * B_r * (4pi))
    numerator = radar_peak_power_w * gain_linear ** 2 * rcs_m2 * jammer_range_m ** 2 * jammer_bw_hz
    denominator = snr_req_linear * jammer_erp_w * radar_bw_hz * 4.0 * math.pi
    if denominator <= 0:
        return float("inf")
    r4 = numerator / denominator
    return r4 ** 0.25


# ===================================================================
# Deceptive Jamming (RGPO — Range Gate Pull-Off)
# ===================================================================


def deceptive_jam_false_targets(
    jammer_pos: np.ndarray,
    sensor_pos: np.ndarray,
    n_false: int = 3,
    range_offset_m: tuple[float, float] = (500.0, 5000.0),
    azimuth_spread_deg: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate false radar returns from a deceptive jammer (RGPO).

    Returns list of dicts with keys: range_m, azimuth_deg, rcs_dbsm,
    source_id, velocity_mps.
    """
    if rng is None:
        rng = np.random.default_rng()

    # True jammer bearing and range
    dx = jammer_pos[0] - sensor_pos[0]
    dy = jammer_pos[1] - sensor_pos[1]
    true_range = math.sqrt(dx ** 2 + dy ** 2)
    true_az = math.degrees(math.atan2(dx, dy)) % 360.0

    results = []
    for i in range(n_false):
        # Offset in range (RGPO pulls the range gate)
        r_offset = rng.uniform(range_offset_m[0], range_offset_m[1])
        sign = rng.choice([-1, 1])
        false_range = max(100.0, true_range + sign * r_offset)

        # Small azimuth jitter
        az_jitter = rng.uniform(-azimuth_spread_deg / 2, azimuth_spread_deg / 2)
        false_az = (true_az + az_jitter) % 360.0

        # Deceptive returns typically have moderate RCS
        false_rcs = rng.uniform(5.0, 20.0)

        results.append({
            "range_m": false_range,
            "azimuth_deg": false_az,
            "rcs_dbsm": false_rcs,
            "velocity_mps": rng.uniform(-50.0, 50.0),
            "source_id": f"deceptive_{i}",
        })
    return results


# ===================================================================
# Chaff
# ===================================================================


@dataclass
class ChaffCloud:
    """Model of a chaff cloud — metallic dipole strips dispersed in air.

    Chaff has very high, uniform RCS across all radar bands (unlike stealth
    which varies dramatically). It decelerates rapidly due to drag and
    dissipates over ~60 seconds.
    """

    position: np.ndarray  # [x, y] meters at deploy time
    velocity: np.ndarray  # [vx, vy] m/s at deploy time (aircraft speed)
    deploy_time: float  # epoch seconds
    initial_rcs_dbsm: float = 30.0  # Very high RCS
    lifetime_s: float = 60.0  # Dissipation time
    drag_coefficient: float = 0.5  # Exponential velocity decay rate (1/s)
    cloud_id: str = "chaff_0"

    def current_position(self, t: float) -> np.ndarray:
        """Position at time *t*, accounting for drag deceleration."""
        dt = max(0.0, t - self.deploy_time)
        if dt <= 0:
            return self.position.copy()
        # Exponential velocity decay: v(t) = v0 * exp(-drag * dt)
        # position = p0 + v0/drag * (1 - exp(-drag * dt))
        if self.drag_coefficient > 0:
            decay = 1.0 - math.exp(-self.drag_coefficient * dt)
            displacement = self.velocity * decay / self.drag_coefficient
        else:
            displacement = self.velocity * dt
        return self.position + displacement

    def current_velocity(self, t: float) -> np.ndarray:
        """Velocity at time *t*, decaying due to drag."""
        dt = max(0.0, t - self.deploy_time)
        decay = math.exp(-self.drag_coefficient * dt) if self.drag_coefficient > 0 else 1.0
        return self.velocity * decay

    def current_rcs_dbsm(self, t: float) -> float:
        """RCS at time *t*, decaying linearly over lifetime."""
        dt = max(0.0, t - self.deploy_time)
        if dt >= self.lifetime_s:
            return -100.0  # effectively zero
        fraction_remaining = 1.0 - dt / self.lifetime_s
        # Linear decay in dB space: rcs drops by ~20 dB over lifetime
        return self.initial_rcs_dbsm + 20.0 * math.log10(max(fraction_remaining, 0.01))

    def is_active(self, t: float) -> bool:
        """Whether the chaff cloud is still present."""
        return (t - self.deploy_time) < self.lifetime_s


def chaff_radar_return(
    chaff: ChaffCloud,
    sensor_pos: np.ndarray,
    t: float,
    band: RadarBand,
) -> dict | None:
    """Generate a radar return from a chaff cloud.

    Returns dict with range_m, azimuth_deg, rcs_dbsm, velocity_mps,
    source_id, or ``None`` if chaff has expired.

    Key physics: chaff RCS is **uniform across all bands** (unlike stealth).
    """
    if not chaff.is_active(t):
        return None
    pos = chaff.current_position(t)
    vel = chaff.current_velocity(t)
    dx = pos[0] - sensor_pos[0]
    dy = pos[1] - sensor_pos[1]
    range_m = math.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0

    # Radial velocity component
    if range_m > 0:
        radial_vel = (vel[0] * dx + vel[1] * dy) / range_m
    else:
        radial_vel = 0.0

    rcs = chaff.current_rcs_dbsm(t)

    return {
        "range_m": range_m,
        "azimuth_deg": azimuth_deg,
        "rcs_dbsm": rcs,
        "velocity_mps": radial_vel,
        "source_id": chaff.cloud_id,
    }


# ===================================================================
# Decoys
# ===================================================================


@dataclass
class DecoySource:
    """Expendable decoy that mimics a target's radar signature.

    Most decoys lack IR emitters, so thermal sensors can discriminate them.
    """

    position: np.ndarray  # [x, y] meters
    velocity: np.ndarray  # [vx, vy] m/s
    rcs_dbsm: float = 10.0  # Designed to mimic real target
    has_thermal_signature: bool = False  # Most decoys have no IR source
    thermal_temperature_k: float = 300.0  # If has_thermal, temperature
    decoy_id: str = "decoy_0"
    deploy_time: float = 0.0
    lifetime_s: float = 120.0  # Decoy operational time

    def current_position(self, t: float) -> np.ndarray:
        """Position at time *t* (ballistic trajectory, no drag for simplicity)."""
        dt = max(0.0, t - self.deploy_time)
        return self.position + self.velocity * dt

    def is_active(self, t: float) -> bool:
        return (t - self.deploy_time) < self.lifetime_s


def decoy_radar_return(
    decoy: DecoySource,
    sensor_pos: np.ndarray,
    t: float,
) -> dict | None:
    """Generate radar return from a decoy."""
    if not decoy.is_active(t):
        return None
    pos = decoy.current_position(t)
    dx = pos[0] - sensor_pos[0]
    dy = pos[1] - sensor_pos[1]
    range_m = math.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0

    # Radial velocity
    if range_m > 0:
        radial_vel = (decoy.velocity[0] * dx + decoy.velocity[1] * dy) / range_m
    else:
        radial_vel = 0.0

    return {
        "range_m": range_m,
        "azimuth_deg": azimuth_deg,
        "rcs_dbsm": decoy.rcs_dbsm,
        "velocity_mps": radial_vel,
        "source_id": decoy.decoy_id,
    }


def decoy_thermal_return(
    decoy: DecoySource,
    sensor_pos: np.ndarray,
    t: float,
) -> dict | None:
    """Generate thermal return from a decoy. None if no IR signature."""
    if not decoy.has_thermal_signature or not decoy.is_active(t):
        return None
    pos = decoy.current_position(t)
    dx = pos[0] - sensor_pos[0]
    dy = pos[1] - sensor_pos[1]
    azimuth_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    range_m = math.sqrt(dx ** 2 + dy ** 2)

    return {
        "azimuth_deg": azimuth_deg,
        "range_m": range_m,
        "temperature_k": decoy.thermal_temperature_k,
        "source_id": decoy.decoy_id,
    }


# ===================================================================
# Jammer Source
# ===================================================================


@dataclass
class JammerSource:
    """An electronic jammer emitting noise or deceptive signals."""

    position: np.ndarray  # [x, y] meters
    erp_watts: float  # Effective Radiated Power
    bandwidth_hz: float  # Jamming bandwidth
    jam_type: str = "noise"  # "noise" | "deceptive"
    active: bool = True
    target_bands: list[RadarBand] | None = None  # None = broadband (all)
    n_false_targets: int = 3  # For deceptive mode
    jammer_id: str = "jammer_0"

    def affects_band(self, band: RadarBand) -> bool:
        """Whether this jammer affects the given radar band."""
        if not self.active:
            return False
        if self.target_bands is None:
            return True  # Broadband
        return band in self.target_bands


# ===================================================================
# ECCM (Electronic Counter-Countermeasures)
# ===================================================================


@dataclass
class ECCMConfig:
    """Counter-countermeasure techniques to mitigate jamming."""

    sidelobe_blanking: bool = False
    sidelobe_blanking_threshold_db: float = -10.0
    frequency_agility: bool = False
    frequency_agility_bands: int = 3  # Number of bands to hop across
    burn_through_mode: bool = False
    burn_through_power_factor: float = 4.0  # Power increase when burn-through
    quantum_eccm: bool = False
    quantum_eccm_advantage_db: float = 6.0  # Extra dB margin from QI


def apply_eccm_to_js(
    js_ratio_db: float,
    eccm: ECCMConfig,
    is_sidelobe: bool = False,
) -> float:
    """Reduce effective J/S ratio based on ECCM techniques.

    Returns the modified J/S in dB (lower = better for radar).
    """
    effective_js = js_ratio_db

    # Sidelobe blanking: reject sidelobe jamming entirely
    if eccm.sidelobe_blanking and is_sidelobe:
        if js_ratio_db < -eccm.sidelobe_blanking_threshold_db:
            return -100.0  # Jammer fully rejected

    # Frequency agility: spread jammer energy across more bands
    if eccm.frequency_agility and eccm.frequency_agility_bands > 1:
        # Jammer must spread power across bands → effective power per band reduced
        effective_js -= 10.0 * math.log10(eccm.frequency_agility_bands)

    # Burn-through mode: increase radar power
    if eccm.burn_through_mode and eccm.burn_through_power_factor > 1.0:
        effective_js -= 10.0 * math.log10(eccm.burn_through_power_factor)

    return effective_js


def quantum_jamming_resistance_db(
    squeeze_param_r: float,
    n_background: float,
) -> float:
    """QI advantage against noise jamming.

    Entangled photon correlations are inherently resistant to added noise.
    The QI receiver can distinguish signal correlations from jammer noise,
    providing additional dB of effective SNR.

    Returns positive dB advantage.
    """
    if squeeze_param_r <= 0 or n_background <= 0:
        return 0.0
    n_signal = math.sinh(squeeze_param_r) ** 2
    if n_signal <= 0:
        return 0.0
    # QI advantage ratio: 4/N_S (capped for stability)
    ratio = min(10000.0, 4.0 / n_signal)
    return 10.0 * math.log10(ratio)


# ===================================================================
# EWModel Facade
# ===================================================================


@dataclass
class EWModel:
    """Unified EW model combining jammers, chaff, decoys, and ECCM.

    Passed via ``EnvironmentModel.ew`` to each simulator. When ``None``,
    all EW effects are skipped.
    """

    jammers: list[JammerSource] = field(default_factory=list)
    chaff_clouds: list[ChaffCloud] = field(default_factory=list)
    decoys: list[DecoySource] = field(default_factory=list)
    eccm: ECCMConfig = field(default_factory=ECCMConfig)

    # Reference radar parameters for J/S computation
    radar_peak_power_w: float = 1e6
    radar_gain_db: float = 30.0
    radar_bandwidth_hz: float = 1e6

    def noise_jamming_snr_reduction(
        self,
        target_range_m: float,
        freq_hz: float,
        band: RadarBand | None = None,
    ) -> float:
        """Total SNR reduction (positive dB) from all active noise jammers.

        Args:
            target_range_m: Range from radar to target.
            freq_hz: Radar operating frequency.
            band: Optional radar band for narrowband jammer filtering.
        """
        total_reduction_db = 0.0
        for jammer in self.jammers:
            if not jammer.active or jammer.jam_type != "noise":
                continue
            if band is not None and not jammer.affects_band(band):
                continue
            jammer_range = float(np.linalg.norm(jammer.position))
            if jammer_range <= 0:
                jammer_range = 1.0
            js = jammer_to_signal_ratio_db(
                jammer_erp_w=jammer.erp_watts,
                jammer_range_m=jammer_range,
                jammer_bw_hz=jammer.bandwidth_hz,
                radar_peak_power_w=self.radar_peak_power_w,
                radar_range_m=target_range_m,
                radar_bw_hz=self.radar_bandwidth_hz,
                radar_gain_db=self.radar_gain_db,
            )
            # Apply ECCM
            js = apply_eccm_to_js(js, self.eccm)
            total_reduction_db += noise_jamming_snr_reduction_db(js)
        return total_reduction_db

    def get_deceptive_false_targets(
        self,
        sensor_pos: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Aggregate false targets from all deceptive jammers."""
        results = []
        for jammer in self.jammers:
            if not jammer.active or jammer.jam_type != "deceptive":
                continue
            false_targets = deceptive_jam_false_targets(
                jammer_pos=jammer.position,
                sensor_pos=sensor_pos,
                n_false=jammer.n_false_targets,
                rng=rng,
            )
            results.extend(false_targets)
        return results

    def get_chaff_returns(
        self,
        sensor_pos: np.ndarray,
        t: float,
        band: RadarBand,
    ) -> list[dict]:
        """Radar returns from all active chaff clouds."""
        results = []
        for chaff in self.chaff_clouds:
            ret = chaff_radar_return(chaff, sensor_pos, t, band)
            if ret is not None:
                results.append(ret)
        return results

    def get_decoy_radar_returns(
        self,
        sensor_pos: np.ndarray,
        t: float,
    ) -> list[dict]:
        """Radar returns from all active decoys."""
        results = []
        for decoy in self.decoys:
            ret = decoy_radar_return(decoy, sensor_pos, t)
            if ret is not None:
                results.append(ret)
        return results

    def get_decoy_thermal_returns(
        self,
        sensor_pos: np.ndarray,
        t: float,
    ) -> list[dict]:
        """Thermal returns from decoys with IR signatures."""
        results = []
        for decoy in self.decoys:
            ret = decoy_thermal_return(decoy, sensor_pos, t)
            if ret is not None:
                results.append(ret)
        return results

    def effective_quantum_advantage_db(self, base_advantage_db: float) -> float:
        """Compute effective QI advantage including ECCM bonus."""
        if self.eccm.quantum_eccm:
            return base_advantage_db + self.eccm.quantum_eccm_advantage_db
        return base_advantage_db

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> EWModel:
        """Create from the ``sentinel.environment.ew`` config section."""
        jammers = []
        for jcfg in cfg.get("jammers", []):
            jammers.append(JammerSource(
                position=np.array(jcfg.get("position", [0.0, 0.0])),
                erp_watts=jcfg.get("erp_watts", 1e4),
                bandwidth_hz=jcfg.get("bandwidth_hz", 1e6),
                jam_type=jcfg.get("jam_type", "noise"),
                active=jcfg.get("active", True),
                target_bands=[RadarBand(b) for b in jcfg["target_bands"]]
                if jcfg.get("target_bands") else None,
                n_false_targets=jcfg.get("n_false_targets", 3),
                jammer_id=jcfg.get("jammer_id", "jammer_0"),
            ))

        chaff_clouds = []
        for ccfg in cfg.get("chaff_clouds", []):
            chaff_clouds.append(ChaffCloud(
                position=np.array(ccfg.get("position", [0.0, 0.0])),
                velocity=np.array(ccfg.get("velocity", [0.0, 0.0])),
                deploy_time=ccfg.get("deploy_time", 0.0),
                initial_rcs_dbsm=ccfg.get("initial_rcs_dbsm", 30.0),
                lifetime_s=ccfg.get("lifetime_s", 60.0),
                drag_coefficient=ccfg.get("drag_coefficient", 0.5),
                cloud_id=ccfg.get("cloud_id", "chaff_0"),
            ))

        decoys = []
        for dcfg in cfg.get("decoys", []):
            decoys.append(DecoySource(
                position=np.array(dcfg.get("position", [0.0, 0.0])),
                velocity=np.array(dcfg.get("velocity", [0.0, 0.0])),
                rcs_dbsm=dcfg.get("rcs_dbsm", 10.0),
                has_thermal_signature=dcfg.get("has_thermal_signature", False),
                thermal_temperature_k=dcfg.get("thermal_temperature_k", 300.0),
                decoy_id=dcfg.get("decoy_id", "decoy_0"),
                deploy_time=dcfg.get("deploy_time", 0.0),
                lifetime_s=dcfg.get("lifetime_s", 120.0),
            ))

        eccm_cfg = cfg.get("eccm", {})
        eccm = ECCMConfig(
            sidelobe_blanking=eccm_cfg.get("sidelobe_blanking", False),
            sidelobe_blanking_threshold_db=eccm_cfg.get("sidelobe_blanking_threshold_db", -10.0),
            frequency_agility=eccm_cfg.get("frequency_agility", False),
            frequency_agility_bands=eccm_cfg.get("frequency_agility_bands", 3),
            burn_through_mode=eccm_cfg.get("burn_through_mode", False),
            burn_through_power_factor=eccm_cfg.get("burn_through_power_factor", 4.0),
            quantum_eccm=eccm_cfg.get("quantum_eccm", False),
            quantum_eccm_advantage_db=eccm_cfg.get("quantum_eccm_advantage_db", 6.0),
        )

        return cls(
            jammers=jammers,
            chaff_clouds=chaff_clouds,
            decoys=decoys,
            eccm=eccm,
            radar_peak_power_w=cfg.get("radar_peak_power_w", 1e6),
            radar_gain_db=cfg.get("radar_gain_db", 30.0),
            radar_bandwidth_hz=cfg.get("radar_bandwidth_hz", 1e6),
        )
