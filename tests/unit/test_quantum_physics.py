"""Tests for quantum illumination physics models."""

import math

from sentinel.sensors.physics import (
    ReceiverType,
    channel_transmissivity,
    classical_detection_probability,
    classical_error_exponent,
    classical_practical_pd,
    entanglement_fidelity,
    qi_detection_probability,
    qi_error_exponent,
    qi_practical_pd,
    qi_snr_advantage_db,
    radar_snr,
    receiver_efficiency,
    thermal_background_photons,
    tmsv_mean_photons,
)


class TestTMSVSource:
    def test_zero_squeeze(self):
        assert tmsv_mean_photons(0.0) == 0.0

    def test_small_squeeze(self):
        # r=0.1 -> N_S ~ 0.01
        ns = tmsv_mean_photons(0.1)
        assert 0.009 < ns < 0.011

    def test_moderate_squeeze(self):
        # r=0.5 -> N_S ~ 0.274
        ns = tmsv_mean_photons(0.5)
        assert 0.25 < ns < 0.30

    def test_large_squeeze(self):
        # r=1.0 -> N_S ~ 1.38
        ns = tmsv_mean_photons(1.0)
        assert 1.3 < ns < 1.5

    def test_increases_with_squeeze(self):
        assert tmsv_mean_photons(0.5) > tmsv_mean_photons(0.1)
        assert tmsv_mean_photons(1.0) > tmsv_mean_photons(0.5)


class TestChannelTransmissivity:
    def test_close_range_high_rcs(self):
        # Large target at close range -> high transmissivity
        eta = channel_transmissivity(
            rcs_m2=100.0,  # 20 dBsm
            antenna_gain=1000.0,  # 30 dBi
            wavelength_m=0.03,  # 10 GHz
            range_m=100.0,
        )
        assert eta > 0.0

    def test_inverse_r4(self):
        # Double range -> 16x lower transmissivity
        eta1 = channel_transmissivity(10.0, 1000.0, 0.03, 1000.0)
        eta2 = channel_transmissivity(10.0, 1000.0, 0.03, 2000.0)
        ratio = eta1 / eta2
        assert abs(ratio - 16.0) < 0.1

    def test_proportional_to_rcs(self):
        eta1 = channel_transmissivity(1.0, 100.0, 0.03, 5000.0)
        eta2 = channel_transmissivity(10.0, 100.0, 0.03, 5000.0)
        assert abs(eta2 / eta1 - 10.0) < 0.01

    def test_zero_range(self):
        assert channel_transmissivity(10.0, 100.0, 0.03, 0.0) == 0.0

    def test_zero_rcs(self):
        assert channel_transmissivity(0.0, 100.0, 0.03, 5000.0) == 0.0

    def test_clamped_to_one(self):
        # Extremely close range with massive RCS
        eta = channel_transmissivity(1e6, 1e6, 1.0, 1.0)
        assert eta == 1.0

    def test_stealth_rcs_low_transmissivity(self):
        # Stealth: -20 dBsm = 0.01 m^2 at 8 km
        eta = channel_transmissivity(
            rcs_m2=0.01,
            antenna_gain=1000.0,
            wavelength_m=0.03,
            range_m=8000.0,
        )
        assert eta < 0.01  # Very low return


class TestThermalBackgroundPhotons:
    def test_microwave_room_temp(self):
        # 10 GHz at 290 K -> N_B >> 1
        nb = thermal_background_photons(10e9, 290.0)
        assert nb > 500  # Expected ~603

    def test_higher_freq_fewer_photons(self):
        nb_low = thermal_background_photons(1e9, 290.0)
        nb_high = thermal_background_photons(100e9, 290.0)
        assert nb_low > nb_high

    def test_higher_temp_more_photons(self):
        nb_cold = thermal_background_photons(10e9, 100.0)
        nb_hot = thermal_background_photons(10e9, 300.0)
        assert nb_hot > nb_cold

    def test_zero_freq(self):
        assert thermal_background_photons(0.0, 290.0) == 0.0

    def test_zero_temp(self):
        assert thermal_background_photons(10e9, 0.0) == 0.0


class TestQIErrorExponent:
    def test_basic(self):
        # beta_QI = M * N_S / N_B
        beta = qi_error_exponent(0.01, 600.0, 10000)
        expected = 10000 * 0.01 / 600.0
        assert abs(beta - expected) < 1e-10

    def test_zero_background(self):
        assert qi_error_exponent(0.01, 0.0, 10000) == float("inf")

    def test_zero_signal(self):
        assert qi_error_exponent(0.0, 600.0, 10000) == 0.0

    def test_increases_with_modes(self):
        b1 = qi_error_exponent(0.01, 600.0, 1000)
        b2 = qi_error_exponent(0.01, 600.0, 10000)
        assert b2 == 10 * b1


class TestClassicalErrorExponent:
    def test_basic(self):
        # beta_C = M * N_S^2 / (4 * N_B)
        beta = classical_error_exponent(0.01, 600.0, 10000)
        expected = 10000 * 0.01**2 / (4.0 * 600.0)
        assert abs(beta - expected) < 1e-12

    def test_always_less_than_qi(self):
        # QI always >= classical for N_S < 4
        for ns in [0.001, 0.01, 0.1, 1.0]:
            beta_qi = qi_error_exponent(ns, 600.0, 10000)
            beta_cl = classical_error_exponent(ns, 600.0, 10000)
            assert beta_qi >= beta_cl


class TestQISNRAdvantage:
    def test_small_ns(self):
        # N_S = 0.01 -> advantage = 10*log10(400) ~ 26 dB
        adv = qi_snr_advantage_db(0.01)
        assert abs(adv - 10.0 * math.log10(400)) < 0.01

    def test_ns_one(self):
        # N_S = 1.0 -> advantage = 10*log10(4) ~ 6 dB
        adv = qi_snr_advantage_db(1.0)
        assert abs(adv - 10.0 * math.log10(4)) < 0.01

    def test_cap_at_40db(self):
        adv = qi_snr_advantage_db(1e-10)
        assert adv == 40.0

    def test_zero_signal(self):
        assert qi_snr_advantage_db(0.0) == 40.0


class TestDetectionProbabilities:
    def test_qi_higher_than_classical(self):
        # Core QI result: P_d(QI) > P_d(classical)
        ns = 0.01
        nb = 600.0
        m = 10000
        eta = 1e-8  # Stealth at long range
        pd_qi = qi_detection_probability(ns, nb, m, eta, 1.0)
        pd_cl = classical_detection_probability(ns, nb, m, eta)
        assert pd_qi > pd_cl

    def test_qi_detects_stealth(self):
        # Stealth scenario: QI Pd significantly higher than classical
        ns = 0.01
        nb = 600.0
        m = 1_000_000
        eta = 1e-4
        pd_qi = qi_detection_probability(ns, nb, m, eta, 1.0)
        pd_cl = classical_detection_probability(ns, nb, m, eta)
        assert pd_qi > pd_cl  # QI always better
        assert pd_qi > 0.001  # Non-negligible detection
        assert pd_qi > 100 * pd_cl  # Massive advantage at low N_S

    def test_classical_misses_stealth(self):
        # Same scenario, classical Pd much lower
        ns = 0.01
        nb = 600.0
        m = 1_000_000
        eta = 1e-4
        pd_cl = classical_detection_probability(ns, nb, m, eta)
        pd_qi = qi_detection_probability(ns, nb, m, eta, 1.0)
        assert pd_qi / max(pd_cl, 1e-30) > 10  # QI at least 10x better

    def test_high_transmissivity(self):
        # Easy target: high eta, QI well above classical
        pd_qi = qi_detection_probability(0.01, 600.0, 100000, 0.01, 1.0)
        pd_cl = classical_detection_probability(0.01, 600.0, 100000, 0.01)
        assert pd_qi > pd_cl
        # QI advantage ratio should be large when N_S << 1
        assert pd_qi > 0.01

    def test_receiver_efficiency_scales(self):
        pd_full = qi_detection_probability(0.01, 600.0, 10000, 1e-5, 1.0)
        pd_half = qi_detection_probability(0.01, 600.0, 10000, 1e-5, 0.5)
        # Half efficiency -> lower Pd
        assert pd_full > pd_half


class TestReceiverEfficiency:
    def test_opa(self):
        assert receiver_efficiency(ReceiverType.OPA) == 0.5

    def test_sfg(self):
        assert receiver_efficiency(ReceiverType.SFG) == 0.9

    def test_phase_conjugate(self):
        assert receiver_efficiency(ReceiverType.PHASE_CONJUGATE) == 0.5

    def test_optimal(self):
        assert receiver_efficiency(ReceiverType.OPTIMAL) == 1.0


class TestEntanglementFidelity:
    def test_no_loss(self):
        # eta=1 -> best fidelity
        f = entanglement_fidelity(1.0, 0.01, 600.0)
        assert 0 < f < 1

    def test_total_loss(self):
        # eta=0 -> no fidelity
        f = entanglement_fidelity(0.0, 0.01, 600.0)
        assert f == 0.0

    def test_decreases_with_loss(self):
        f_high = entanglement_fidelity(0.1, 0.01, 600.0)
        f_low = entanglement_fidelity(0.001, 0.01, 600.0)
        assert f_high > f_low

    def test_decreases_with_background(self):
        f_quiet = entanglement_fidelity(0.01, 0.01, 100.0)
        f_noisy = entanglement_fidelity(0.01, 0.01, 10000.0)
        assert f_quiet > f_noisy


class TestRadarSNR:
    def test_at_reference_point(self):
        snr = radar_snr(10.0, 10000.0, 10000.0, 10.0, 15.0)
        assert abs(snr - 15.0) < 0.01

    def test_closer_range_higher_snr(self):
        snr_near = radar_snr(10.0, 5000.0)
        snr_far = radar_snr(10.0, 10000.0)
        # Half range -> 16x more power -> +12 dB
        assert abs((snr_near - snr_far) - 12.04) < 0.1

    def test_lower_rcs_lower_snr(self):
        snr_big = radar_snr(10.0, 5000.0)
        snr_small = radar_snr(0.01, 5000.0)  # -20 dBsm stealth
        assert snr_big > snr_small
        assert abs((snr_big - snr_small) - 30.0) < 0.1

    def test_zero_range(self):
        assert radar_snr(10.0, 0.0) == -100.0


class TestPracticalDetection:
    def test_qi_higher_than_classical_conventional(self):
        # Conventional target: both high, but QI higher
        rcs_m2 = 10.0  # 10 dBsm
        pd_qi = qi_practical_pd(rcs_m2, 5000.0, 0.01, 1.0)
        pd_cl = classical_practical_pd(rcs_m2, 5000.0)
        assert pd_qi >= pd_cl

    def test_qi_detects_stealth_classical_misses(self):
        # Stealth target at medium range: QI detects, classical may not
        rcs_m2 = 0.01  # -20 dBsm
        pd_qi = qi_practical_pd(rcs_m2, 8000.0, 0.01, 1.0)
        pd_cl = classical_practical_pd(rcs_m2, 8000.0)
        assert pd_qi > pd_cl
        # QI should have nontrivial Pd for stealth
        assert pd_qi > 0.01

    def test_conventional_high_pd(self):
        # Large target at close range: should be near 1.0
        pd = qi_practical_pd(100.0, 2000.0, 0.01, 1.0)
        assert pd > 0.9

    def test_receiver_efficiency_affects_pd(self):
        rcs_m2 = 0.01
        pd_full = qi_practical_pd(rcs_m2, 5000.0, 0.01, 1.0)
        pd_half = qi_practical_pd(rcs_m2, 5000.0, 0.01, 0.5)
        assert pd_full >= pd_half

    def test_pd_decreases_with_range(self):
        rcs_m2 = 0.1  # Small target so Pd doesn't saturate
        pd_near = qi_practical_pd(rcs_m2, 3000.0, 0.01, 1.0)
        pd_far = qi_practical_pd(rcs_m2, 15000.0, 0.01, 1.0)
        assert pd_near > pd_far
