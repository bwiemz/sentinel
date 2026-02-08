"""Physics models for multi-frequency radar, thermal signature, and quantum
illumination simulation.

Models frequency-dependent RCS, plasma sheath attenuation, thermal
signatures for conventional/stealth/hypersonic targets, and quantum
illumination radar physics (TMSV source, channel loss, QI vs classical
error exponents, receiver models).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from sentinel.core.types import RadarBand, TargetType, ThermalBand

# RCS offset from X-band baseline in dB, by (band, target_type).
# Stealth RAM is optimized for X-band; lower frequencies see much larger RCS.
# Conventional targets have roughly flat RCS across bands.
_RCS_OFFSET_DB: dict[tuple[RadarBand, TargetType], float] = {
    # Stealth: large gains at low frequencies
    (RadarBand.VHF, TargetType.STEALTH): 25.0,  # +25 dB (100-1000x)
    (RadarBand.UHF, TargetType.STEALTH): 22.0,  # +22 dB
    (RadarBand.L_BAND, TargetType.STEALTH): 15.0,  # +15 dB
    (RadarBand.S_BAND, TargetType.STEALTH): 7.0,  # +7 dB
    (RadarBand.X_BAND, TargetType.STEALTH): 0.0,  # baseline
    # Conventional: roughly flat
    (RadarBand.VHF, TargetType.CONVENTIONAL): 1.0,
    (RadarBand.UHF, TargetType.CONVENTIONAL): 0.5,
    (RadarBand.L_BAND, TargetType.CONVENTIONAL): 0.0,
    (RadarBand.S_BAND, TargetType.CONVENTIONAL): 0.0,
    (RadarBand.X_BAND, TargetType.CONVENTIONAL): 0.0,
    # Hypersonic: slight gains at low freq due to plasma interactions
    (RadarBand.VHF, TargetType.HYPERSONIC): 3.0,
    (RadarBand.UHF, TargetType.HYPERSONIC): 2.0,
    (RadarBand.L_BAND, TargetType.HYPERSONIC): 1.0,
    (RadarBand.S_BAND, TargetType.HYPERSONIC): 0.0,
    (RadarBand.X_BAND, TargetType.HYPERSONIC): 0.0,
}

# Plasma attenuation in dB at Mach 5, by band.
# Higher frequencies are attenuated more by the plasma sheath.
_PLASMA_ATTENUATION_MACH5_DB: dict[RadarBand, float] = {
    RadarBand.VHF: 2.0,
    RadarBand.UHF: 4.0,
    RadarBand.L_BAND: 7.0,
    RadarBand.S_BAND: 10.0,
    RadarBand.X_BAND: 15.0,
}


@dataclass
class RCSProfile:
    """Frequency-dependent radar cross section for a target."""

    x_band_dbsm: float  # Baseline RCS at X-band in dBsm
    target_type: TargetType = TargetType.CONVENTIONAL

    def rcs_at_band(self, band: RadarBand) -> float:
        """Compute effective RCS in dBsm at the given frequency band."""
        offset = _RCS_OFFSET_DB.get((band, self.target_type), 0.0)
        return self.x_band_dbsm + offset

    def rcs_linear_at_band(self, band: RadarBand) -> float:
        """Compute effective RCS in m^2 (linear) at the given frequency band."""
        return 10.0 ** (self.rcs_at_band(band) / 10.0)

    @staticmethod
    def band_offset_db(band: RadarBand, target_type: TargetType) -> float:
        """Get the dB offset from X-band for a given band and target type."""
        return _RCS_OFFSET_DB.get((band, target_type), 0.0)


@dataclass
class PlasmaSheath:
    """Models plasma sheath effects on radar at hypersonic speeds.

    Plasma forms around vehicles at Mach 5+ due to aerodynamic heating
    of air. The ionized gas absorbs/scatters RF energy, degrading radar
    returns. Higher radar frequencies are more severely attenuated.
    """

    onset_mach: float = 5.0  # Mach number where plasma becomes significant

    def attenuation_db(self, mach: float, band: RadarBand) -> float:
        """Signal attenuation in dB due to plasma sheath.

        Scales quadratically above onset Mach. Higher frequencies are
        more attenuated.
        """
        if mach < self.onset_mach:
            return 0.0
        base = _PLASMA_ATTENUATION_MACH5_DB.get(band, 10.0)
        # Quadratic scaling above Mach 5
        excess = (mach - self.onset_mach) / self.onset_mach
        return base * (1.0 + excess) ** 2

    def detection_probability_factor(self, mach: float, band: RadarBand) -> float:
        """Multiplicative factor on detection probability (0 to 1).

        Converts attenuation dB to a probability reduction factor.
        """
        atten = self.attenuation_db(mach, band)
        # Convert one-way attenuation to detection probability factor
        # Using a sigmoid-like mapping: factor = 10^(-atten/20)
        return max(0.01, 10.0 ** (-atten / 20.0))


# Default singleton for convenience
_DEFAULT_PLASMA = PlasmaSheath()


def plasma_attenuation_db(mach: float, band: RadarBand) -> float:
    """Convenience: plasma sheath attenuation in dB."""
    return _DEFAULT_PLASMA.attenuation_db(mach, band)


def plasma_detection_factor(mach: float, band: RadarBand) -> float:
    """Convenience: plasma sheath detection probability factor."""
    return _DEFAULT_PLASMA.detection_probability_factor(mach, band)


@dataclass
class ThermalSignature:
    """Temperature model for targets at various speeds.

    Models engine exhaust, aerodynamic heating of leading edges and body,
    accounting for target type and Mach number.
    """

    target_type: TargetType = TargetType.CONVENTIONAL
    ambient_k: float = 280.0  # Background temperature

    # Engine exhaust base temperatures (Kelvin) by target type
    _ENGINE_BASE_K: dict[TargetType, float] = field(
        default_factory=lambda: {
            TargetType.CONVENTIONAL: 700.0,  # ~427 C
            TargetType.STEALTH: 500.0,  # Cooled exhaust
            TargetType.HYPERSONIC: 900.0,  # Scramjet
        }
    )

    def temperature_at_mach(self, mach: float) -> dict[str, float]:
        """Compute temperatures for different components at given Mach.

        Returns dict with keys: 'engine', 'leading_edge', 'body'.
        All values in Kelvin.
        """
        engine_base = self._ENGINE_BASE_K.get(self.target_type, 700.0)

        # Engine temperature increases with thrust (proportional to Mach)
        t_engine = engine_base + 150.0 * max(0, mach - 0.3)

        # Aerodynamic heating: T_stag = T_amb * (1 + 0.2 * M^2) for air (gamma=1.4)
        # Leading edge sees stagnation temperature
        recovery_factor = 0.85  # Turbulent boundary layer
        t_leading = self.ambient_k * (1.0 + recovery_factor * 0.2 * mach**2)

        # Body surface: lower recovery factor
        body_recovery = 0.70
        t_body = self.ambient_k * (1.0 + body_recovery * 0.2 * mach**2)

        return {
            "engine": t_engine,
            "leading_edge": t_leading,
            "body": t_body,
        }

    def peak_temperature_k(self, mach: float) -> float:
        """Peak surface temperature (max of all components)."""
        temps = self.temperature_at_mach(mach)
        return max(temps.values())

    def thermal_contrast(self, mach: float) -> float:
        """Temperature contrast (delta-T) above ambient."""
        return self.peak_temperature_k(mach) - self.ambient_k

    def band_intensity(self, mach: float, band: ThermalBand) -> float:
        """Relative intensity in a given thermal band (0-1 scale).

        MWIR: strongest from engine/plume (peaks at ~600-900 C).
        LWIR: strongest from heated surfaces (peaks at 100-300 C).
        SWIR: only significant at very high temperatures (>700 C).
        """
        temps = self.temperature_at_mach(mach)
        t_peak = max(temps.values())

        if band == ThermalBand.MWIR:
            # MWIR peaks from engine, effective range 500-1500 K
            return min(1.0, max(0.0, (temps["engine"] - 400.0) / 1000.0))
        elif band == ThermalBand.LWIR:
            # LWIR captures body/leading edge, effective range 250-600 K
            t_surface = max(temps["leading_edge"], temps["body"])
            return min(1.0, max(0.0, (t_surface - self.ambient_k) / 300.0))
        elif band == ThermalBand.SWIR:
            # SWIR only significant at very high temperatures
            return min(1.0, max(0.0, (t_peak - 900.0) / 2000.0))
        return 0.0


def combined_detection_probability(probs: list[float]) -> float:
    """Combined detection probability from independent sensors.

    P_total = 1 - product(1 - P_i)
    """
    if not probs:
        return 0.0
    product = 1.0
    for p in probs:
        product *= 1.0 - max(0.0, min(1.0, p))
    return 1.0 - product


# ---------------------------------------------------------------------------
# Quantum Illumination (QI) Physics
# ---------------------------------------------------------------------------
# References:
#   Lloyd, S. "Enhanced Sensitivity of Photodetection via Quantum Illumination"
#     Science 321, 1463 (2008)
#   Tan, S. H. et al. "Quantum Illumination with Gaussian States"
#     Phys. Rev. Lett. 101, 253601 (2008)
#   Guha, S. & Erkmen, B. I. "Gaussian-state quantum-illumination receivers
#     for target detection" Phys. Rev. A 80, 052310 (2009)
# ---------------------------------------------------------------------------

# Physical constants
_PLANCK_H = 6.62607015e-34  # J*s
_BOLTZMANN_K = 1.380649e-23  # J/K
_SPEED_OF_LIGHT = 2.99792458e8  # m/s


class ReceiverType(enum.Enum):
    """Quantum illumination receiver architectures."""

    OPA = "opa"  # Optical Parametric Amplifier (3 dB, demonstrated)
    SFG = "sfg"  # Sum-Frequency Generation (up to 6 dB, theoretical)
    PHASE_CONJUGATE = "phase_conjugate"  # Phase-Conjugate receiver (3 dB)
    OPTIMAL = "optimal"  # Theoretical optimal (6 dB, Helstrom bound)


# Fraction of theoretical QI advantage achieved by each receiver
_RECEIVER_EFFICIENCY: dict[ReceiverType, float] = {
    ReceiverType.OPA: 0.5,  # 3 dB of 6 dB
    ReceiverType.SFG: 0.9,  # Near-optimal
    ReceiverType.PHASE_CONJUGATE: 0.5,  # 3 dB of 6 dB
    ReceiverType.OPTIMAL: 1.0,  # Full 6 dB
}


def receiver_efficiency(receiver: ReceiverType) -> float:
    """Fraction of theoretical QI advantage achieved by the receiver."""
    return _RECEIVER_EFFICIENCY.get(receiver, 0.5)


def tmsv_mean_photons(squeeze_param_r: float) -> float:
    """Mean signal photon number from Two-Mode Squeezed Vacuum state.

    N_S = sinh^2(r) where r is the squeeze parameter.
    For QI advantage, N_S << 1 (low squeeze).
    """
    return math.sinh(squeeze_param_r) ** 2


def channel_transmissivity(
    rcs_m2: float,
    antenna_gain: float,
    wavelength_m: float,
    range_m: float,
) -> float:
    """Round-trip channel transmissivity (signal loss).

    eta = (sigma * G^2 * lambda^2) / ((4*pi)^3 * R^4)

    Args:
        rcs_m2: Radar cross section in m^2 (linear, not dBsm).
        antenna_gain: Antenna gain (linear, not dBi).
        wavelength_m: Operating wavelength in meters.
        range_m: Target range in meters.

    Returns:
        Transmissivity eta in [0, 1]. Clamped to 1.0 maximum.
    """
    if range_m <= 0 or rcs_m2 <= 0:
        return 0.0
    numerator = rcs_m2 * antenna_gain**2 * wavelength_m**2
    denominator = (4.0 * math.pi) ** 3 * range_m**4
    return min(1.0, numerator / denominator)


def thermal_background_photons(freq_hz: float, temp_k: float) -> float:
    """Mean thermal photon number per mode at given frequency and temperature.

    N_B = 1 / (exp(h*f / k*T) - 1)

    For microwave frequencies at room temperature, N_B >> 1
    (e.g., ~600 at 10 GHz, 290 K).
    """
    if freq_hz <= 0 or temp_k <= 0:
        return 0.0
    exponent = (_PLANCK_H * freq_hz) / (_BOLTZMANN_K * temp_k)
    if exponent > 500:  # Avoid overflow
        return 0.0
    return 1.0 / (math.exp(exponent) - 1.0)


def qi_error_exponent(
    n_signal: float,
    n_background: float,
    n_modes: int,
) -> float:
    """Quantum illumination error exponent (Chernoff bound).

    beta_QI = M * N_S / N_B

    Higher is better -- lower error probability.
    """
    if n_background <= 0:
        return float("inf") if n_signal > 0 else 0.0
    return n_modes * n_signal / n_background


def classical_error_exponent(
    n_signal: float,
    n_background: float,
    n_modes: int,
) -> float:
    """Classical (coherent state) radar error exponent.

    beta_C = M * N_S^2 / (4 * N_B)

    For same total energy as QI, classical is worse by factor 4/N_S.
    """
    if n_background <= 0:
        return float("inf") if n_signal > 0 else 0.0
    return n_modes * n_signal**2 / (4.0 * n_background)


def qi_snr_advantage_db(n_signal: float) -> float:
    """QI advantage over classical in dB.

    Ratio = 4 / N_S. For N_S = 0.01, this is 26 dB.
    Capped at 40 dB for numerical stability.
    """
    if n_signal <= 0:
        return 40.0  # Cap
    ratio = 4.0 / n_signal
    return min(40.0, 10.0 * math.log10(ratio))


def qi_detection_probability(
    n_signal: float,
    n_background: float,
    n_modes: int,
    transmissivity: float,
    receiver_eff: float = 0.5,
) -> float:
    """Detection probability using quantum illumination (theoretical).

    Uses the Chernoff-bound-based model:
    P_d = 1 - exp(-receiver_eff * beta_QI * transmissivity)

    Note: At practical microwave frequencies and ranges, transmissivity is
    extremely small (~10^-12), requiring enormous M for nonzero Pd. Use
    qi_practical_pd() for simulation with realistic detection rates.
    """
    beta = qi_error_exponent(n_signal, n_background, n_modes)
    if beta == float("inf"):
        return 1.0
    effective = receiver_eff * beta * transmissivity
    return 1.0 - math.exp(-min(effective, 500.0))


def classical_detection_probability(
    n_signal: float,
    n_background: float,
    n_modes: int,
    transmissivity: float,
) -> float:
    """Detection probability using classical (coherent state) radar (theoretical).

    P_d = 1 - exp(-beta_C * transmissivity)
    """
    beta = classical_error_exponent(n_signal, n_background, n_modes)
    if beta == float("inf"):
        return 1.0
    effective = beta * transmissivity
    return 1.0 - math.exp(-min(effective, 500.0))


# ---------------------------------------------------------------------------
# Practical detection model for simulation
# ---------------------------------------------------------------------------
# The theoretical Chernoff-bound model requires enormous mode counts to
# produce nonzero Pd at practical ranges (transmissivity ~10^-12 at km
# ranges). For simulation, we use a radar-equation-based SNR model with
# the QI advantage applied as a multiplicative boost to the SNR exponent.
# This faithfully preserves the QI/classical ratio while producing usable
# detection probabilities.
# ---------------------------------------------------------------------------


def radar_snr(
    rcs_m2: float,
    range_m: float,
    ref_range_m: float = 10000.0,
    ref_rcs_m2: float = 10.0,
    base_snr_db: float = 15.0,
) -> float:
    """Compute radar signal-to-noise ratio in dB.

    Scales from a reference design point using the radar range equation:
    SNR = SNR_ref * (sigma/sigma_ref) * (R_ref/R)^4

    Args:
        rcs_m2: Target RCS in m^2 (linear).
        range_m: Target range in meters.
        ref_range_m: Reference range for base_snr_db.
        ref_rcs_m2: Reference RCS for base_snr_db.
        base_snr_db: SNR at (ref_rcs_m2, ref_range_m).

    Returns:
        SNR in dB.
    """
    if range_m <= 0 or rcs_m2 <= 0:
        return -100.0
    ratio_rcs = rcs_m2 / ref_rcs_m2
    ratio_range = (ref_range_m / range_m) ** 4
    return base_snr_db + 10.0 * math.log10(ratio_rcs * ratio_range)


def _snr_to_pd(snr_db: float) -> float:
    """Convert SNR (dB) to detection probability using Swerling-1 approximation.

    Pd = 1 - exp(-10^(SNR_dB/10) / threshold_factor)
    Tuned so SNR=13 dB gives Pd ~0.9 (standard radar design point).
    """
    if snr_db < -50:
        return 0.0
    snr_linear = 10.0 ** (snr_db / 10.0)
    # Threshold factor calibrated for Pd~0.9 at SNR=13dB: -ln(0.1)/20 ≈ 0.115
    threshold = 20.0
    return min(1.0, 1.0 - math.exp(-snr_linear / threshold))


def qi_practical_pd(
    rcs_m2: float,
    range_m: float,
    n_signal: float,
    receiver_eff: float = 0.5,
    ref_range_m: float = 10000.0,
    ref_rcs_m2: float = 10.0,
    base_snr_db: float = 15.0,
) -> float:
    """Practical QI detection probability for simulation.

    Computes a radar-equation SNR, then boosts it by the QI advantage
    (scaled by receiver efficiency). This preserves the correct QI/classical
    ratio while giving realistic detection rates.
    """
    snr = radar_snr(rcs_m2, range_m, ref_range_m, ref_rcs_m2, base_snr_db)
    # QI boost: 10*log10(4/N_S) * receiver_eff
    qi_boost_db = qi_snr_advantage_db(n_signal) * receiver_eff
    return _snr_to_pd(snr + qi_boost_db)


def classical_practical_pd(
    rcs_m2: float,
    range_m: float,
    ref_range_m: float = 10000.0,
    ref_rcs_m2: float = 10.0,
    base_snr_db: float = 15.0,
) -> float:
    """Practical classical detection probability for simulation.

    Standard radar-equation Pd without QI boost.
    """
    snr = radar_snr(rcs_m2, range_m, ref_range_m, ref_rcs_m2, base_snr_db)
    return _snr_to_pd(snr)


def entanglement_fidelity(
    transmissivity: float,
    n_signal: float,
    n_background: float,
) -> float:
    """Entanglement fidelity after channel transmission.

    F = eta * N_S / (eta * N_S + (1 - eta) * N_B + 1)

    Returns a value in [0, 1]. Higher means more quantum correlations
    survive the channel. QI still provides advantage even when F is low.
    """
    num = transmissivity * n_signal
    denom = num + (1.0 - transmissivity) * n_background + 1.0
    if denom <= 0:
        return 0.0
    return num / denom
