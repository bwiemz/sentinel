"""Tests for physics models: RCS profiles, plasma sheath, thermal signatures."""

import math

import pytest

from sentinel.core.types import RadarBand, TargetType, ThermalBand
from sentinel.sensors.physics import (
    PlasmaSheath,
    RCSProfile,
    ThermalSignature,
    combined_detection_probability,
    plasma_attenuation_db,
    plasma_detection_factor,
)


# === RCS Profile ===


class TestRCSProfile:
    def test_conventional_flat_across_bands(self):
        profile = RCSProfile(x_band_dbsm=10.0, target_type=TargetType.CONVENTIONAL)
        for band in RadarBand:
            rcs = profile.rcs_at_band(band)
            assert abs(rcs - 10.0) < 2.0, f"Conventional RCS at {band} deviates too much"

    def test_stealth_low_at_xband(self):
        profile = RCSProfile(x_band_dbsm=-30.0, target_type=TargetType.STEALTH)
        assert profile.rcs_at_band(RadarBand.X_BAND) == -30.0

    def test_stealth_high_at_vhf(self):
        profile = RCSProfile(x_band_dbsm=-30.0, target_type=TargetType.STEALTH)
        vhf_rcs = profile.rcs_at_band(RadarBand.VHF)
        assert vhf_rcs > -10.0  # Much higher than X-band

    def test_stealth_gradient_vhf_to_xband(self):
        profile = RCSProfile(x_band_dbsm=-30.0, target_type=TargetType.STEALTH)
        rcs_vhf = profile.rcs_at_band(RadarBand.VHF)
        rcs_uhf = profile.rcs_at_band(RadarBand.UHF)
        rcs_l = profile.rcs_at_band(RadarBand.L_BAND)
        rcs_s = profile.rcs_at_band(RadarBand.S_BAND)
        rcs_x = profile.rcs_at_band(RadarBand.X_BAND)
        # Should monotonically decrease from VHF to X-band
        assert rcs_vhf >= rcs_uhf >= rcs_l >= rcs_s >= rcs_x

    def test_rcs_linear_conversion(self):
        profile = RCSProfile(x_band_dbsm=10.0, target_type=TargetType.CONVENTIONAL)
        linear = profile.rcs_linear_at_band(RadarBand.X_BAND)
        assert abs(linear - 10.0) < 0.01  # 10 dBsm = 10 m^2

    def test_rcs_linear_negative_dbsm(self):
        profile = RCSProfile(x_band_dbsm=-30.0, target_type=TargetType.STEALTH)
        linear = profile.rcs_linear_at_band(RadarBand.X_BAND)
        assert abs(linear - 0.001) < 0.0001  # -30 dBsm = 0.001 m^2

    def test_band_offset_static_method(self):
        offset = RCSProfile.band_offset_db(RadarBand.VHF, TargetType.STEALTH)
        assert offset == 25.0

    def test_hypersonic_slight_gains_low_freq(self):
        profile = RCSProfile(x_band_dbsm=5.0, target_type=TargetType.HYPERSONIC)
        assert profile.rcs_at_band(RadarBand.VHF) > profile.rcs_at_band(RadarBand.X_BAND)


# === Plasma Sheath ===


class TestPlasmaSheath:
    def test_no_attenuation_below_mach5(self):
        ps = PlasmaSheath()
        assert ps.attenuation_db(4.9, RadarBand.X_BAND) == 0.0
        assert ps.attenuation_db(0.0, RadarBand.VHF) == 0.0
        assert ps.attenuation_db(3.0, RadarBand.S_BAND) == 0.0

    def test_attenuation_at_mach5(self):
        ps = PlasmaSheath()
        atten = ps.attenuation_db(5.0, RadarBand.X_BAND)
        assert atten > 0.0

    def test_attenuation_increases_with_mach(self):
        ps = PlasmaSheath()
        a5 = ps.attenuation_db(5.0, RadarBand.X_BAND)
        a10 = ps.attenuation_db(10.0, RadarBand.X_BAND)
        a20 = ps.attenuation_db(20.0, RadarBand.X_BAND)
        assert a5 < a10 < a20

    def test_xband_attenuated_more_than_vhf(self):
        ps = PlasmaSheath()
        x = ps.attenuation_db(7.0, RadarBand.X_BAND)
        v = ps.attenuation_db(7.0, RadarBand.VHF)
        assert x > v

    def test_detection_factor_below_mach5_is_one(self):
        ps = PlasmaSheath()
        assert ps.detection_probability_factor(4.0, RadarBand.X_BAND) == 1.0

    def test_detection_factor_at_mach5_reduced(self):
        ps = PlasmaSheath()
        factor = ps.detection_probability_factor(5.0, RadarBand.X_BAND)
        assert 0.0 < factor < 1.0

    def test_detection_factor_vhf_higher_than_xband(self):
        ps = PlasmaSheath()
        fv = ps.detection_probability_factor(7.0, RadarBand.VHF)
        fx = ps.detection_probability_factor(7.0, RadarBand.X_BAND)
        assert fv > fx

    def test_detection_factor_never_below_floor(self):
        ps = PlasmaSheath()
        factor = ps.detection_probability_factor(20.0, RadarBand.X_BAND)
        assert factor >= 0.01

    def test_convenience_functions(self):
        assert plasma_attenuation_db(3.0, RadarBand.X_BAND) == 0.0
        assert plasma_detection_factor(3.0, RadarBand.X_BAND) == 1.0
        assert plasma_attenuation_db(7.0, RadarBand.X_BAND) > 0


# === Thermal Signature ===


class TestThermalSignature:
    def test_conventional_subsonic(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        temps = ts.temperature_at_mach(0.8)
        assert temps["engine"] > 500  # Engine is hot
        assert temps["leading_edge"] < 400  # Minimal aero heating
        assert temps["body"] < temps["leading_edge"]

    def test_hypersonic_extreme_leading_edge(self):
        ts = ThermalSignature(target_type=TargetType.HYPERSONIC)
        temps = ts.temperature_at_mach(10.0)
        # Stagnation temp at M10: ~280*(1+0.85*0.2*100) = 280*17.8 = ~5040 K
        assert temps["leading_edge"] > 3000

    def test_temperature_increases_with_mach(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        t1 = ts.peak_temperature_k(1.0)
        t3 = ts.peak_temperature_k(3.0)
        t5 = ts.peak_temperature_k(5.0)
        assert t1 < t3 < t5

    def test_stealth_cooler_engine(self):
        ts_stealth = ThermalSignature(target_type=TargetType.STEALTH)
        ts_conv = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        # Stealth has cooled exhaust
        assert ts_stealth.temperature_at_mach(0.9)["engine"] < \
               ts_conv.temperature_at_mach(0.9)["engine"]

    def test_thermal_contrast_above_ambient(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        contrast = ts.thermal_contrast(1.0)
        assert contrast > 0

    def test_hypersonic_massive_contrast(self):
        ts = ThermalSignature(target_type=TargetType.HYPERSONIC)
        contrast = ts.thermal_contrast(10.0)
        assert contrast > 2000  # Thousands of degrees above ambient

    def test_peak_temperature(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        peak = ts.peak_temperature_k(0.8)
        temps = ts.temperature_at_mach(0.8)
        assert peak == max(temps.values())

    def test_mwir_intensity_from_engine(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        intensity = ts.band_intensity(0.9, ThermalBand.MWIR)
        assert intensity > 0.0

    def test_lwir_intensity_from_surface(self):
        ts = ThermalSignature(target_type=TargetType.HYPERSONIC)
        intensity = ts.band_intensity(5.0, ThermalBand.LWIR)
        assert intensity > 0.5  # Strong surface heating

    def test_swir_low_at_subsonic(self):
        ts = ThermalSignature(target_type=TargetType.CONVENTIONAL)
        intensity = ts.band_intensity(0.8, ThermalBand.SWIR)
        assert intensity < 0.1  # SWIR needs very high temps

    def test_swir_high_at_hypersonic(self):
        ts = ThermalSignature(target_type=TargetType.HYPERSONIC)
        intensity = ts.band_intensity(10.0, ThermalBand.SWIR)
        assert intensity > 0.3  # Extreme temps visible in SWIR


# === Combined Detection Probability ===


class TestCombinedDetectionProbability:
    def test_single_sensor(self):
        assert combined_detection_probability([0.7]) == pytest.approx(0.7)

    def test_two_sensors(self):
        # P = 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
        assert combined_detection_probability([0.5, 0.5]) == pytest.approx(0.75)

    def test_five_bands_high_pd(self):
        probs = [0.5, 0.5, 0.5, 0.5, 0.5]
        total = combined_detection_probability(probs)
        # 1 - 0.5^5 = 1 - 0.03125 = 0.96875
        assert total == pytest.approx(0.96875)

    def test_all_zero(self):
        assert combined_detection_probability([0.0, 0.0, 0.0]) == 0.0

    def test_one_perfect_sensor(self):
        assert combined_detection_probability([1.0, 0.3, 0.5]) == pytest.approx(1.0)

    def test_empty_list(self):
        assert combined_detection_probability([]) == 0.0

    def test_clamped_to_valid_range(self):
        result = combined_detection_probability([0.5, 1.5])
        assert 0.0 <= result <= 1.0
