"""Unit tests for electronic warfare module (sensors/ew.py)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import RadarBand
from sentinel.sensors.ew import (
    ChaffCloud,
    DecoySource,
    ECCMConfig,
    EWModel,
    JammerSource,
    apply_eccm_to_js,
    burn_through_range_m,
    chaff_radar_return,
    deceptive_jam_false_targets,
    decoy_radar_return,
    decoy_thermal_return,
    jammer_to_signal_ratio_db,
    noise_jamming_snr_reduction_db,
    quantum_jamming_resistance_db,
)


# ===================================================================
# J/S Ratio
# ===================================================================


class TestJammerToSignalRatio:
    def test_positive_js_when_jammer_dominates(self):
        """High ERP jammer at close range should give positive J/S."""
        js = jammer_to_signal_ratio_db(
            jammer_erp_w=1e5,
            jammer_range_m=5000.0,
            jammer_bw_hz=1e6,
            radar_peak_power_w=1e6,
            radar_range_m=20000.0,
            radar_bw_hz=1e6,
        )
        assert js > 0, f"J/S should be positive when jammer dominates, got {js}"

    def test_negative_js_when_radar_dominates(self):
        """Weak jammer at long range should give negative J/S."""
        js = jammer_to_signal_ratio_db(
            jammer_erp_w=100.0,
            jammer_range_m=50000.0,
            jammer_bw_hz=1e6,
            radar_peak_power_w=1e6,
            radar_range_m=5000.0,
            radar_bw_hz=1e6,
        )
        assert js < 0, f"J/S should be negative when radar dominates, got {js}"

    def test_js_increases_with_jammer_erp(self):
        """Higher jammer ERP → higher J/S."""
        js_low = jammer_to_signal_ratio_db(
            jammer_erp_w=1e3, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        js_high = jammer_to_signal_ratio_db(
            jammer_erp_w=1e5, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        assert js_high > js_low

    def test_js_increases_with_target_range(self):
        """Target at longer range → radar signal weaker → higher J/S."""
        js_near = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=5000.0, radar_bw_hz=1e6,
        )
        js_far = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=30000.0, radar_bw_hz=1e6,
        )
        assert js_far > js_near

    def test_js_decreases_with_jammer_range(self):
        """Jammer farther away → less effective → lower J/S."""
        js_near = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=5000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        js_far = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=30000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        assert js_near > js_far

    def test_zero_erp_returns_very_low(self):
        js = jammer_to_signal_ratio_db(
            jammer_erp_w=0.0, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        assert js <= -90

    def test_zero_jammer_range_returns_low(self):
        js = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=0.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        assert js <= -90

    def test_bandwidth_ratio_effect(self):
        """Wider jammer bandwidth → less power per Hz → lower J/S."""
        js_narrow = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=10000.0, jammer_bw_hz=1e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        js_wide = jammer_to_signal_ratio_db(
            jammer_erp_w=1e4, jammer_range_m=10000.0, jammer_bw_hz=10e6,
            radar_peak_power_w=1e6, radar_range_m=10000.0, radar_bw_hz=1e6,
        )
        assert js_narrow > js_wide


# ===================================================================
# Noise Jamming SNR Reduction
# ===================================================================


class TestNoiseJammingSNRReduction:
    def test_zero_js_gives_about_3db_reduction(self):
        """J/S = 0 dB → SNR reduced by ~3 dB (factor of 2)."""
        reduction = noise_jamming_snr_reduction_db(0.0)
        assert 2.5 < reduction < 3.5

    def test_high_js_gives_large_reduction(self):
        """J/S = 20 dB → SNR reduced by ~20 dB."""
        reduction = noise_jamming_snr_reduction_db(20.0)
        assert reduction > 19.0

    def test_very_negative_js_gives_zero_reduction(self):
        """Very low J/S → negligible SNR reduction."""
        reduction = noise_jamming_snr_reduction_db(-60.0)
        assert reduction < 0.01

    def test_moderate_js(self):
        """J/S = 10 dB → reduction ~10.4 dB."""
        reduction = noise_jamming_snr_reduction_db(10.0)
        assert 10.0 < reduction < 11.0

    def test_reduction_increases_with_js(self):
        r_low = noise_jamming_snr_reduction_db(5.0)
        r_high = noise_jamming_snr_reduction_db(15.0)
        assert r_high > r_low


# ===================================================================
# Burn-Through Range
# ===================================================================


class TestBurnThroughRange:
    def test_burn_through_range_finite(self):
        """Should return a finite positive range."""
        bt = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=10.0,
            jammer_erp_w=1e4, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        assert 0 < bt < 100000

    def test_stronger_jammer_reduces_burn_through(self):
        """Stronger jammer → need to be closer → smaller burn-through range."""
        bt_weak = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=10.0,
            jammer_erp_w=1e3, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        bt_strong = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=10.0,
            jammer_erp_w=1e5, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        assert bt_weak > bt_strong

    def test_larger_rcs_increases_burn_through(self):
        """Bigger target → easier to detect → larger burn-through range."""
        bt_small = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=1.0,
            jammer_erp_w=1e4, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        bt_large = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=100.0,
            jammer_erp_w=1e4, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        assert bt_large > bt_small

    def test_zero_rcs_returns_inf(self):
        bt = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=0.0,
            jammer_erp_w=1e4, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        assert bt == float("inf")

    def test_zero_jammer_returns_inf(self):
        bt = burn_through_range_m(
            radar_peak_power_w=1e6, radar_gain_db=30.0, rcs_m2=10.0,
            jammer_erp_w=0.0, jammer_range_m=20000.0,
            jammer_bw_hz=1e6, radar_bw_hz=1e6,
        )
        assert bt == float("inf")


# ===================================================================
# Deceptive Jamming
# ===================================================================


class TestDeceptiveJamming:
    def test_generates_correct_number_of_false_targets(self):
        results = deceptive_jam_false_targets(
            jammer_pos=np.array([10000.0, 0.0]),
            sensor_pos=np.array([0.0, 0.0]),
            n_false=5,
            rng=np.random.default_rng(42),
        )
        assert len(results) == 5

    def test_false_targets_have_required_keys(self):
        results = deceptive_jam_false_targets(
            jammer_pos=np.array([10000.0, 0.0]),
            sensor_pos=np.array([0.0, 0.0]),
            rng=np.random.default_rng(42),
        )
        for rt in results:
            assert "range_m" in rt
            assert "azimuth_deg" in rt
            assert "rcs_dbsm" in rt
            assert "source_id" in rt

    def test_false_targets_near_jammer_bearing(self):
        """False targets should be near the jammer's true bearing."""
        results = deceptive_jam_false_targets(
            jammer_pos=np.array([10000.0, 0.0]),
            sensor_pos=np.array([0.0, 0.0]),
            azimuth_spread_deg=2.0,
            rng=np.random.default_rng(42),
        )
        # True azimuth to jammer: atan2(10000, 0) ≈ 90 deg
        true_az = math.degrees(math.atan2(10000.0, 0.0)) % 360.0
        for rt in results:
            diff = abs(rt["azimuth_deg"] - true_az)
            diff = min(diff, 360.0 - diff)
            assert diff < 5.0, f"False target azimuth {rt['azimuth_deg']} too far from true {true_az}"

    def test_range_offsets_within_bounds(self):
        results = deceptive_jam_false_targets(
            jammer_pos=np.array([10000.0, 0.0]),
            sensor_pos=np.array([0.0, 0.0]),
            range_offset_m=(1000.0, 3000.0),
            n_false=20,
            rng=np.random.default_rng(42),
        )
        for rt in results:
            assert rt["range_m"] >= 100.0  # Clamped minimum

    def test_default_rng(self):
        """Should work without explicit rng."""
        results = deceptive_jam_false_targets(
            jammer_pos=np.array([5000.0, 5000.0]),
            sensor_pos=np.array([0.0, 0.0]),
        )
        assert len(results) == 3


# ===================================================================
# Chaff Cloud
# ===================================================================


class TestChaffCloud:
    @pytest.fixture
    def chaff(self):
        return ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([-200.0, 0.0]),
            deploy_time=1000.0,
            initial_rcs_dbsm=30.0,
            lifetime_s=60.0,
            drag_coefficient=0.5,
        )

    def test_position_at_deploy_time(self, chaff):
        pos = chaff.current_position(1000.0)
        np.testing.assert_allclose(pos, [5000.0, 0.0])

    def test_position_moves_with_drag(self, chaff):
        pos = chaff.current_position(1002.0)  # 2s after deploy
        # With drag=0.5, after 2s: displacement = v0/drag * (1-exp(-drag*2))
        # = -200/0.5 * (1-exp(-1)) = -400 * 0.632 = -252.8
        assert pos[0] < 5000.0  # Moved left
        assert pos[0] > 4700.0  # But not too far (drag slowing)

    def test_velocity_decays_with_drag(self, chaff):
        vel0 = chaff.current_velocity(1000.0)
        vel2 = chaff.current_velocity(1002.0)
        assert abs(vel2[0]) < abs(vel0[0])

    def test_rcs_at_deploy_time(self, chaff):
        rcs = chaff.current_rcs_dbsm(1000.0)
        assert abs(rcs - 30.0) < 0.1

    def test_rcs_decays_over_time(self, chaff):
        rcs_early = chaff.current_rcs_dbsm(1010.0)
        rcs_late = chaff.current_rcs_dbsm(1050.0)
        assert rcs_late < rcs_early

    def test_rcs_very_low_after_lifetime(self, chaff):
        rcs = chaff.current_rcs_dbsm(1060.1)
        assert rcs < -90

    def test_is_active_before_lifetime(self, chaff):
        assert chaff.is_active(1030.0)

    def test_not_active_after_lifetime(self, chaff):
        assert not chaff.is_active(1061.0)


class TestChaffRadarReturn:
    def test_returns_dict_for_active_chaff(self):
        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
        )
        ret = chaff_radar_return(chaff, np.array([0.0, 0.0]), 10.0, RadarBand.X_BAND)
        assert ret is not None
        assert "range_m" in ret
        assert "azimuth_deg" in ret
        assert "rcs_dbsm" in ret
        assert "source_id" in ret

    def test_returns_none_for_expired_chaff(self):
        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
            lifetime_s=30.0,
        )
        ret = chaff_radar_return(chaff, np.array([0.0, 0.0]), 31.0, RadarBand.VHF)
        assert ret is None

    def test_rcs_uniform_across_bands(self):
        """Key chaff physics: RCS is the same across all bands."""
        chaff = ChaffCloud(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
        )
        rcs_values = {}
        for band in RadarBand:
            ret = chaff_radar_return(chaff, np.array([0.0, 0.0]), 5.0, band)
            rcs_values[band] = ret["rcs_dbsm"]
        # All should be within 0.1 dB of each other
        rcs_list = list(rcs_values.values())
        assert max(rcs_list) - min(rcs_list) < 0.1

    def test_range_computed_correctly(self):
        chaff = ChaffCloud(
            position=np.array([3000.0, 4000.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
        )
        ret = chaff_radar_return(chaff, np.array([0.0, 0.0]), 0.0, RadarBand.X_BAND)
        assert abs(ret["range_m"] - 5000.0) < 1.0


# ===================================================================
# Decoys
# ===================================================================


class TestDecoySource:
    def test_current_position_at_deploy(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([-100.0, 50.0]),
            deploy_time=100.0,
        )
        pos = decoy.current_position(100.0)
        np.testing.assert_allclose(pos, [5000.0, 0.0])

    def test_current_position_after_time(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([-100.0, 50.0]),
            deploy_time=100.0,
        )
        pos = decoy.current_position(110.0)
        np.testing.assert_allclose(pos, [4000.0, 500.0])

    def test_is_active_within_lifetime(self):
        decoy = DecoySource(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
            lifetime_s=120.0,
        )
        assert decoy.is_active(60.0)

    def test_not_active_after_lifetime(self):
        decoy = DecoySource(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
            lifetime_s=120.0,
        )
        assert not decoy.is_active(121.0)


class TestDecoyRadarReturn:
    def test_returns_dict_for_active_decoy(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
        )
        ret = decoy_radar_return(decoy, np.array([0.0, 0.0]), 0.0)
        assert ret is not None
        assert ret["rcs_dbsm"] == 10.0
        assert abs(ret["range_m"] - 5000.0) < 1.0

    def test_returns_none_for_expired_decoy(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=0.0,
            lifetime_s=60.0,
        )
        ret = decoy_radar_return(decoy, np.array([0.0, 0.0]), 61.0)
        assert ret is None


class TestDecoyThermalReturn:
    def test_no_thermal_without_ir(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            has_thermal_signature=False,
        )
        ret = decoy_thermal_return(decoy, np.array([0.0, 0.0]), 0.0)
        assert ret is None

    def test_thermal_with_ir(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            has_thermal_signature=True,
            thermal_temperature_k=800.0,
        )
        ret = decoy_thermal_return(decoy, np.array([0.0, 0.0]), 0.0)
        assert ret is not None
        assert ret["temperature_k"] == 800.0
        assert "azimuth_deg" in ret

    def test_no_thermal_after_lifetime(self):
        decoy = DecoySource(
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            has_thermal_signature=True,
            deploy_time=0.0,
            lifetime_s=60.0,
        )
        ret = decoy_thermal_return(decoy, np.array([0.0, 0.0]), 61.0)
        assert ret is None


# ===================================================================
# Jammer Source
# ===================================================================


class TestJammerSource:
    def test_broadband_affects_all_bands(self):
        jammer = JammerSource(
            position=np.array([10000.0, 0.0]),
            erp_watts=1e4,
            bandwidth_hz=1e6,
            target_bands=None,
        )
        for band in RadarBand:
            assert jammer.affects_band(band)

    def test_narrowband_only_affects_target_bands(self):
        jammer = JammerSource(
            position=np.array([10000.0, 0.0]),
            erp_watts=1e4,
            bandwidth_hz=1e6,
            target_bands=[RadarBand.X_BAND],
        )
        assert jammer.affects_band(RadarBand.X_BAND)
        assert not jammer.affects_band(RadarBand.VHF)
        assert not jammer.affects_band(RadarBand.S_BAND)

    def test_inactive_affects_nothing(self):
        jammer = JammerSource(
            position=np.array([10000.0, 0.0]),
            erp_watts=1e4,
            bandwidth_hz=1e6,
            active=False,
        )
        assert not jammer.affects_band(RadarBand.X_BAND)


# ===================================================================
# ECCM
# ===================================================================


class TestECCM:
    def test_sidelobe_blanking_rejects_sidelobe_jamming(self):
        eccm = ECCMConfig(sidelobe_blanking=True, sidelobe_blanking_threshold_db=-10.0)
        # Sidelobe jamming with low J/S should be rejected
        result = apply_eccm_to_js(-15.0, eccm, is_sidelobe=True)
        assert result <= -90

    def test_sidelobe_blanking_no_effect_on_mainlobe(self):
        eccm = ECCMConfig(sidelobe_blanking=True)
        result = apply_eccm_to_js(10.0, eccm, is_sidelobe=False)
        assert result == 10.0

    def test_frequency_agility_reduces_js(self):
        eccm = ECCMConfig(frequency_agility=True, frequency_agility_bands=4)
        result = apply_eccm_to_js(20.0, eccm)
        # Should reduce by 10*log10(4) ≈ 6 dB
        expected = 20.0 - 10.0 * math.log10(4)
        assert abs(result - expected) < 0.1

    def test_burn_through_reduces_js(self):
        eccm = ECCMConfig(burn_through_mode=True, burn_through_power_factor=4.0)
        result = apply_eccm_to_js(20.0, eccm)
        expected = 20.0 - 10.0 * math.log10(4.0)
        assert abs(result - expected) < 0.1

    def test_combined_eccm(self):
        eccm = ECCMConfig(
            frequency_agility=True, frequency_agility_bands=3,
            burn_through_mode=True, burn_through_power_factor=2.0,
        )
        result = apply_eccm_to_js(20.0, eccm)
        expected = 20.0 - 10.0 * math.log10(3) - 10.0 * math.log10(2.0)
        assert abs(result - expected) < 0.1

    def test_no_eccm_no_change(self):
        eccm = ECCMConfig()
        result = apply_eccm_to_js(15.0, eccm)
        assert result == 15.0


class TestQuantumJammingResistance:
    def test_low_squeeze_gives_high_advantage(self):
        """Low squeeze (small N_S) → large QI advantage."""
        adv = quantum_jamming_resistance_db(0.1, 600.0)
        # N_S = sinh(0.1)^2 ≈ 0.01, ratio = 4/0.01 = 400 → ~26 dB
        assert adv > 20.0

    def test_higher_squeeze_less_advantage(self):
        adv_low = quantum_jamming_resistance_db(0.1, 600.0)
        adv_high = quantum_jamming_resistance_db(0.5, 600.0)
        assert adv_low > adv_high

    def test_zero_squeeze_returns_zero(self):
        adv = quantum_jamming_resistance_db(0.0, 600.0)
        assert adv == 0.0


# ===================================================================
# EWModel Facade
# ===================================================================


class TestEWModel:
    def test_empty_model_no_effect(self):
        model = EWModel()
        assert model.noise_jamming_snr_reduction(10000.0, 10e9) == 0.0
        assert model.get_deceptive_false_targets(np.array([0.0, 0.0])) == []
        assert model.get_chaff_returns(np.array([0.0, 0.0]), 0.0, RadarBand.X_BAND) == []
        assert model.get_decoy_radar_returns(np.array([0.0, 0.0]), 0.0) == []
        assert model.get_decoy_thermal_returns(np.array([0.0, 0.0]), 0.0) == []

    def test_noise_jammer_causes_reduction(self):
        model = EWModel(
            jammers=[JammerSource(
                position=np.array([10000.0, 0.0]),
                erp_watts=1e5,
                bandwidth_hz=1e6,
                jam_type="noise",
            )],
        )
        reduction = model.noise_jamming_snr_reduction(20000.0, 10e9)
        assert reduction > 0

    def test_deceptive_jammer_generates_false_targets(self):
        model = EWModel(
            jammers=[JammerSource(
                position=np.array([10000.0, 0.0]),
                erp_watts=1e4,
                bandwidth_hz=1e6,
                jam_type="deceptive",
                n_false_targets=4,
            )],
        )
        targets = model.get_deceptive_false_targets(
            np.array([0.0, 0.0]), rng=np.random.default_rng(42),
        )
        assert len(targets) == 4

    def test_chaff_returns(self):
        model = EWModel(
            chaff_clouds=[ChaffCloud(
                position=np.array([5000.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
                deploy_time=0.0,
            )],
        )
        returns = model.get_chaff_returns(np.array([0.0, 0.0]), 10.0, RadarBand.X_BAND)
        assert len(returns) == 1

    def test_decoy_radar_returns(self):
        model = EWModel(
            decoys=[DecoySource(
                position=np.array([8000.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
            )],
        )
        returns = model.get_decoy_radar_returns(np.array([0.0, 0.0]), 0.0)
        assert len(returns) == 1

    def test_decoy_thermal_returns_empty_without_ir(self):
        model = EWModel(
            decoys=[DecoySource(
                position=np.array([8000.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
                has_thermal_signature=False,
            )],
        )
        returns = model.get_decoy_thermal_returns(np.array([0.0, 0.0]), 0.0)
        assert len(returns) == 0

    def test_decoy_thermal_returns_with_ir(self):
        model = EWModel(
            decoys=[DecoySource(
                position=np.array([8000.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
                has_thermal_signature=True,
                thermal_temperature_k=700.0,
            )],
        )
        returns = model.get_decoy_thermal_returns(np.array([0.0, 0.0]), 0.0)
        assert len(returns) == 1
        assert returns[0]["temperature_k"] == 700.0

    def test_narrowband_jammer_only_affects_target_band(self):
        model = EWModel(
            jammers=[JammerSource(
                position=np.array([10000.0, 0.0]),
                erp_watts=1e5,
                bandwidth_hz=1e6,
                jam_type="noise",
                target_bands=[RadarBand.X_BAND],
            )],
        )
        # X-band: affected
        reduction_x = model.noise_jamming_snr_reduction(20000.0, 10e9, band=RadarBand.X_BAND)
        # VHF: not affected
        reduction_vhf = model.noise_jamming_snr_reduction(20000.0, 150e6, band=RadarBand.VHF)
        assert reduction_x > 0
        assert reduction_vhf == 0.0

    def test_quantum_advantage_without_eccm(self):
        model = EWModel()
        assert model.effective_quantum_advantage_db(10.0) == 10.0

    def test_quantum_advantage_with_eccm(self):
        model = EWModel(eccm=ECCMConfig(quantum_eccm=True, quantum_eccm_advantage_db=6.0))
        assert model.effective_quantum_advantage_db(10.0) == 16.0

    def test_multiple_jammers_accumulate(self):
        j1 = JammerSource(
            position=np.array([10000.0, 0.0]), erp_watts=1e4,
            bandwidth_hz=1e6, jam_type="noise",
        )
        j2 = JammerSource(
            position=np.array([15000.0, 0.0]), erp_watts=1e4,
            bandwidth_hz=1e6, jam_type="noise",
        )
        model_one = EWModel(jammers=[j1])
        model_two = EWModel(jammers=[j1, j2])
        r_one = model_one.noise_jamming_snr_reduction(20000.0, 10e9)
        r_two = model_two.noise_jamming_snr_reduction(20000.0, 10e9)
        assert r_two > r_one


class TestEWModelFromOmegaConf:
    def test_from_empty_config(self):
        cfg = OmegaConf.create({})
        model = EWModel.from_omegaconf(cfg)
        assert len(model.jammers) == 0
        assert len(model.chaff_clouds) == 0
        assert len(model.decoys) == 0

    def test_from_full_config(self):
        cfg = OmegaConf.create({
            "jammers": [{
                "position": [10000.0, 0.0],
                "erp_watts": 1e5,
                "bandwidth_hz": 1e6,
                "jam_type": "noise",
                "jammer_id": "j1",
            }],
            "chaff_clouds": [{
                "position": [5000.0, 0.0],
                "velocity": [-100.0, 0.0],
                "deploy_time": 0.0,
                "cloud_id": "c1",
            }],
            "decoys": [{
                "position": [8000.0, 0.0],
                "velocity": [-50.0, 0.0],
                "rcs_dbsm": 12.0,
                "has_thermal_signature": True,
                "thermal_temperature_k": 600.0,
                "decoy_id": "d1",
            }],
            "eccm": {
                "quantum_eccm": True,
                "quantum_eccm_advantage_db": 6.0,
            },
            "radar_peak_power_w": 2e6,
        })
        model = EWModel.from_omegaconf(cfg)
        assert len(model.jammers) == 1
        assert model.jammers[0].jammer_id == "j1"
        assert len(model.chaff_clouds) == 1
        assert model.chaff_clouds[0].cloud_id == "c1"
        assert len(model.decoys) == 1
        assert model.decoys[0].has_thermal_signature is True
        assert model.eccm.quantum_eccm is True
        assert model.radar_peak_power_w == 2e6

    def test_narrowband_jammer_from_config(self):
        cfg = OmegaConf.create({
            "jammers": [{
                "position": [10000.0, 0.0],
                "erp_watts": 1e4,
                "bandwidth_hz": 1e6,
                "jam_type": "noise",
                "target_bands": ["x_band"],
            }],
        })
        model = EWModel.from_omegaconf(cfg)
        assert model.jammers[0].target_bands == [RadarBand.X_BAND]
        assert model.jammers[0].affects_band(RadarBand.X_BAND)
        assert not model.jammers[0].affects_band(RadarBand.VHF)
