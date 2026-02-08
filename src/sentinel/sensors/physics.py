"""Physics models for multi-frequency radar and thermal signature simulation.

Models frequency-dependent RCS, plasma sheath attenuation, and thermal
signatures for conventional, stealth, and hypersonic targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sentinel.core.types import RadarBand, TargetType, ThermalBand

# RCS offset from X-band baseline in dB, by (band, target_type).
# Stealth RAM is optimized for X-band; lower frequencies see much larger RCS.
# Conventional targets have roughly flat RCS across bands.
_RCS_OFFSET_DB: dict[tuple[RadarBand, TargetType], float] = {
    # Stealth: large gains at low frequencies
    (RadarBand.VHF, TargetType.STEALTH): 25.0,      # +25 dB (100-1000x)
    (RadarBand.UHF, TargetType.STEALTH): 22.0,      # +22 dB
    (RadarBand.L_BAND, TargetType.STEALTH): 15.0,   # +15 dB
    (RadarBand.S_BAND, TargetType.STEALTH): 7.0,    # +7 dB
    (RadarBand.X_BAND, TargetType.STEALTH): 0.0,    # baseline
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
    _ENGINE_BASE_K: dict[TargetType, float] = field(default_factory=lambda: {
        TargetType.CONVENTIONAL: 700.0,   # ~427 C
        TargetType.STEALTH: 500.0,        # Cooled exhaust
        TargetType.HYPERSONIC: 900.0,     # Scramjet
    })

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
        t_leading = self.ambient_k * (1.0 + recovery_factor * 0.2 * mach ** 2)

        # Body surface: lower recovery factor
        body_recovery = 0.70
        t_body = self.ambient_k * (1.0 + body_recovery * 0.2 * mach ** 2)

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
        product *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - product
