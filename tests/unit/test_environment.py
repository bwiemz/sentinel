"""Unit tests for the environment modeling module (terrain, atmosphere, weather, clutter)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import RadarBand, ThermalBand
from sentinel.sensors.environment import (
    BAND_CENTER_FREQ_HZ,
    EnvironmentModel,
    TerrainGrid,
    WeatherConditions,
    atmospheric_attenuation_db_per_km,
    clutter_false_alarm_rate_multiplier,
    line_of_sight,
    rain_attenuation_db_per_km,
    rain_clutter_snr_reduction_db,
    surface_clutter_snr_reduction_db,
    thermal_atmospheric_transmission,
    total_propagation_loss_db,
    weather_thermal_contrast_factor,
    weather_visibility_range_factor,
)


# ===================================================================
# TerrainGrid
# ===================================================================


class TestTerrainGridFlat:
    def test_flat_shape(self):
        t = TerrainGrid.flat(extent_m=1000.0, resolution_m=100.0)
        assert t.elevation_data.shape == (10, 10)

    def test_flat_all_zeros(self):
        t = TerrainGrid.flat(extent_m=500.0, resolution_m=50.0)
        assert np.all(t.elevation_data == 0.0)

    def test_flat_origin_centered(self):
        t = TerrainGrid.flat(extent_m=1000.0, resolution_m=100.0)
        assert t.origin_x_m == -500.0
        assert t.origin_y_m == -500.0


class TestTerrainGridProceduralHills:
    def test_shape(self):
        t = TerrainGrid.procedural_hills(extent_m=5000.0, resolution_m=100.0, max_elevation_m=300.0)
        assert t.elevation_data.shape == (50, 50)

    def test_nonnegative(self):
        t = TerrainGrid.procedural_hills(extent_m=5000.0, resolution_m=100.0, max_elevation_m=300.0)
        assert np.all(t.elevation_data >= 0.0)

    def test_within_max_elevation(self):
        t = TerrainGrid.procedural_hills(extent_m=5000.0, resolution_m=100.0, max_elevation_m=300.0)
        assert np.all(t.elevation_data <= 300.0)

    def test_deterministic_with_seed(self):
        t1 = TerrainGrid.procedural_hills(seed=123)
        t2 = TerrainGrid.procedural_hills(seed=123)
        np.testing.assert_array_equal(t1.elevation_data, t2.elevation_data)

    def test_different_seeds_differ(self):
        t1 = TerrainGrid.procedural_hills(seed=1)
        t2 = TerrainGrid.procedural_hills(seed=2)
        assert not np.array_equal(t1.elevation_data, t2.elevation_data)

    def test_has_nonzero_terrain(self):
        t = TerrainGrid.procedural_hills(max_elevation_m=500.0)
        assert np.max(t.elevation_data) > 0.0


class TestTerrainGridRidge:
    def test_ridge_peak_at_specified_x(self):
        t = TerrainGrid.ridge(
            extent_m=10000.0, resolution_m=50.0,
            ridge_x_m=3000.0, ridge_height_m=500.0, ridge_width_m=200.0,
        )
        # Peak should be near x=3000
        elev_at_peak = t.elevation_at(3000.0, 0.0)
        assert elev_at_peak > 400.0

    def test_ridge_drops_off(self):
        t = TerrainGrid.ridge(
            extent_m=10000.0, resolution_m=50.0,
            ridge_x_m=3000.0, ridge_height_m=500.0, ridge_width_m=200.0,
        )
        elev_peak = t.elevation_at(3000.0, 0.0)
        elev_far = t.elevation_at(0.0, 0.0)
        assert elev_far < elev_peak * 0.1

    def test_ridge_uniform_in_y(self):
        t = TerrainGrid.ridge(
            extent_m=10000.0, resolution_m=50.0,
            ridge_x_m=2000.0, ridge_height_m=300.0, ridge_width_m=300.0,
        )
        e1 = t.elevation_at(2000.0, 0.0)
        e2 = t.elevation_at(2000.0, 1000.0)
        assert abs(e1 - e2) < 1.0  # Same height at different y


class TestTerrainGridElevationAt:
    def test_center_of_flat(self):
        t = TerrainGrid.flat(extent_m=1000.0, resolution_m=100.0)
        assert t.elevation_at(0.0, 0.0) == 0.0

    def test_outside_grid_returns_zero(self):
        t = TerrainGrid.flat(extent_m=1000.0, resolution_m=100.0)
        assert t.elevation_at(10000.0, 10000.0) == 0.0

    def test_interpolation_between_cells(self):
        t = TerrainGrid(
            elevation_data=np.array([[0.0, 10.0], [0.0, 10.0]]),
            resolution_m=1.0,
            origin_x_m=0.0,
            origin_y_m=0.0,
        )
        # Midpoint between 0 and 10
        assert abs(t.elevation_at(0.5, 0.0) - 5.0) < 0.01


# ===================================================================
# Line of sight
# ===================================================================


class TestLineOfSight:
    def test_flat_terrain_always_visible(self):
        t = TerrainGrid.flat(extent_m=20000.0)
        assert line_of_sight(t, (0, 0, 10), (5000, 0, 10)) is True

    def test_ridge_blocks_target_behind(self):
        t = TerrainGrid.ridge(
            extent_m=20000.0, resolution_m=100.0,
            ridge_x_m=3000.0, ridge_height_m=500.0, ridge_width_m=200.0,
        )
        # Sensor at origin (ground level), target behind ridge
        assert line_of_sight(t, (0, 0, 0), (5000, 0, 0)) is False

    def test_ridge_does_not_block_high_altitude(self):
        t = TerrainGrid.ridge(
            extent_m=20000.0, resolution_m=100.0,
            ridge_x_m=3000.0, ridge_height_m=500.0, ridge_width_m=200.0,
        )
        # Target at high altitude clears the ridge
        assert line_of_sight(t, (0, 0, 0), (5000, 0, 1000)) is True

    def test_sensor_on_hilltop_sees_over_ridge(self):
        t = TerrainGrid.ridge(
            extent_m=20000.0, resolution_m=100.0,
            ridge_x_m=3000.0, ridge_height_m=300.0, ridge_width_m=200.0,
        )
        # Sensor at 600m can see a high-altitude target past a 300m ridge
        assert line_of_sight(t, (0, 0, 600), (5000, 0, 400)) is True

    def test_same_position_always_visible(self):
        t = TerrainGrid.flat(extent_m=10000.0)
        assert line_of_sight(t, (0, 0, 10), (0, 0, 10)) is True


# ===================================================================
# Atmospheric propagation
# ===================================================================


class TestAtmosphericAttenuation:
    def test_vhf_very_low(self):
        atten = atmospheric_attenuation_db_per_km(150e6)
        assert atten < 0.001  # essentially negligible

    def test_xband_moderate(self):
        atten = atmospheric_attenuation_db_per_km(10e9)
        assert 0.001 < atten < 0.5

    def test_60ghz_peak(self):
        # O₂ absorption peak near 60 GHz
        atten = atmospheric_attenuation_db_per_km(60e9)
        assert atten > atmospheric_attenuation_db_per_km(10e9)

    def test_higher_freq_more_attenuation(self):
        a_vhf = atmospheric_attenuation_db_per_km(BAND_CENTER_FREQ_HZ[RadarBand.VHF])
        a_xband = atmospheric_attenuation_db_per_km(BAND_CENTER_FREQ_HZ[RadarBand.X_BAND])
        assert a_xband > a_vhf

    def test_higher_humidity_more_attenuation(self):
        a_dry = atmospheric_attenuation_db_per_km(10e9, humidity_pct=10.0)
        a_humid = atmospheric_attenuation_db_per_km(10e9, humidity_pct=90.0)
        assert a_humid > a_dry

    def test_zero_freq_returns_zero(self):
        assert atmospheric_attenuation_db_per_km(0.0) == 0.0

    def test_nonnegative(self):
        for freq in [1e6, 1e9, 10e9, 60e9]:
            assert atmospheric_attenuation_db_per_km(freq) >= 0.0


class TestRainAttenuation:
    def test_no_rain_no_attenuation(self):
        assert rain_attenuation_db_per_km(10e9, 0.0) == 0.0

    def test_xband_heavy_rain(self):
        atten = rain_attenuation_db_per_km(10e9, 16.0)
        assert atten > 0.01

    def test_vhf_negligible_in_rain(self):
        atten = rain_attenuation_db_per_km(150e6, 50.0)
        assert atten < 0.01

    def test_more_rain_more_attenuation(self):
        a_light = rain_attenuation_db_per_km(10e9, 2.0)
        a_heavy = rain_attenuation_db_per_km(10e9, 50.0)
        assert a_heavy > a_light

    def test_higher_freq_more_rain_loss(self):
        a_vhf = rain_attenuation_db_per_km(150e6, 10.0)
        a_xband = rain_attenuation_db_per_km(10e9, 10.0)
        assert a_xband > a_vhf

    def test_nonnegative(self):
        assert rain_attenuation_db_per_km(10e9, 10.0) >= 0.0


class TestTotalPropagationLoss:
    def test_zero_range_zero_loss(self):
        assert total_propagation_loss_db(10e9, 0.0) == 0.0

    def test_scales_with_range(self):
        loss_5km = total_propagation_loss_db(10e9, 5000.0)
        loss_10km = total_propagation_loss_db(10e9, 10000.0)
        assert loss_10km > loss_5km

    def test_two_way(self):
        # Should be 2 * range_km * atten_per_km
        atten = atmospheric_attenuation_db_per_km(10e9) + rain_attenuation_db_per_km(10e9, 5.0)
        expected = 2.0 * 10.0 * atten
        actual = total_propagation_loss_db(10e9, 10000.0, rain_rate_mm_h=5.0)
        assert abs(actual - expected) < 1e-10

    def test_rain_adds_to_clear_air(self):
        clear = total_propagation_loss_db(10e9, 10000.0, rain_rate_mm_h=0.0)
        rainy = total_propagation_loss_db(10e9, 10000.0, rain_rate_mm_h=16.0)
        assert rainy > clear


class TestThermalAtmosphericTransmission:
    def test_short_range_high_transmission(self):
        trans = thermal_atmospheric_transmission(ThermalBand.MWIR, 100.0)
        assert trans > 0.95

    def test_long_range_reduced(self):
        trans = thermal_atmospheric_transmission(ThermalBand.MWIR, 50000.0)
        assert trans < 0.5

    def test_mwir_better_than_lwir_in_humidity(self):
        mwir = thermal_atmospheric_transmission(ThermalBand.MWIR, 20000.0, humidity_pct=80.0)
        lwir = thermal_atmospheric_transmission(ThermalBand.LWIR, 20000.0, humidity_pct=80.0)
        assert mwir > lwir

    def test_rain_reduces_transmission(self):
        clear = thermal_atmospheric_transmission(ThermalBand.MWIR, 10000.0, rain_rate_mm_h=0.0)
        rainy = thermal_atmospheric_transmission(ThermalBand.MWIR, 10000.0, rain_rate_mm_h=20.0)
        assert rainy < clear

    def test_low_visibility_reduces_transmission(self):
        good = thermal_atmospheric_transmission(ThermalBand.MWIR, 10000.0, visibility_km=20.0)
        poor = thermal_atmospheric_transmission(ThermalBand.MWIR, 10000.0, visibility_km=1.0)
        assert poor < good

    def test_bounded_zero_one(self):
        for band in ThermalBand:
            trans = thermal_atmospheric_transmission(band, 10000.0)
            assert 0.0 <= trans <= 1.0


# ===================================================================
# Weather effects
# ===================================================================


class TestWeatherConditions:
    def test_defaults_are_clear(self):
        w = WeatherConditions()
        assert w.rain_rate_mm_h == 0.0
        assert w.visibility_km == 20.0
        assert w.cloud_cover_pct == 0.0
        assert w.sea_state == 0


class TestWeatherThermalContrastFactor:
    def test_clear_sky_full_contrast(self):
        w = WeatherConditions(cloud_cover_pct=0.0)
        assert weather_thermal_contrast_factor(w) == 1.0

    def test_overcast_reduces_contrast(self):
        w = WeatherConditions(cloud_cover_pct=100.0)
        factor = weather_thermal_contrast_factor(w)
        assert 0.5 < factor < 0.7

    def test_partial_cloud(self):
        w = WeatherConditions(cloud_cover_pct=50.0)
        factor = weather_thermal_contrast_factor(w)
        assert 0.7 < factor < 0.9

    def test_monotonic_with_cloud_cover(self):
        f25 = weather_thermal_contrast_factor(WeatherConditions(cloud_cover_pct=25.0))
        f75 = weather_thermal_contrast_factor(WeatherConditions(cloud_cover_pct=75.0))
        assert f75 < f25


class TestWeatherVisibilityRangeFactor:
    def test_good_visibility_full_range(self):
        w = WeatherConditions(visibility_km=50.0)
        factor = weather_visibility_range_factor(w, max_range_m=20000.0)
        assert factor == 1.0

    def test_poor_visibility_reduced_range(self):
        w = WeatherConditions(visibility_km=5.0)
        factor = weather_visibility_range_factor(w, max_range_m=50000.0)
        assert factor < 0.2

    def test_very_poor_visibility(self):
        w = WeatherConditions(visibility_km=1.0)
        factor = weather_visibility_range_factor(w, max_range_m=50000.0)
        assert factor < 0.05

    def test_zero_max_range(self):
        w = WeatherConditions(visibility_km=20.0)
        assert weather_visibility_range_factor(w, max_range_m=0.0) == 0.0


# ===================================================================
# Clutter model
# ===================================================================


class TestSurfaceClutter:
    def test_high_elevation_no_clutter(self):
        assert surface_clutter_snr_reduction_db(10.0, 5000.0) == 0.0

    def test_low_elevation_has_clutter(self):
        reduction = surface_clutter_snr_reduction_db(0.0, 5000.0)
        assert reduction > 0.0

    def test_sea_state_increases_clutter(self):
        calm = surface_clutter_snr_reduction_db(2.0, 5000.0, sea_state=0)
        rough = surface_clutter_snr_reduction_db(2.0, 5000.0, sea_state=6)
        assert rough > calm

    def test_zero_range_no_clutter(self):
        assert surface_clutter_snr_reduction_db(0.0, 0.0) == 0.0

    def test_nonnegative(self):
        assert surface_clutter_snr_reduction_db(1.0, 10000.0) >= 0.0

    def test_capped_at_30db(self):
        reduction = surface_clutter_snr_reduction_db(0.0, 100000.0, sea_state=9)
        assert reduction <= 30.0


class TestRainClutter:
    def test_no_rain_no_clutter(self):
        assert rain_clutter_snr_reduction_db(10e9, 0.0, 5000.0) == 0.0

    def test_heavy_rain_has_clutter(self):
        reduction = rain_clutter_snr_reduction_db(10e9, 30.0, 10000.0)
        assert reduction > 0.0

    def test_higher_freq_more_rain_clutter(self):
        vhf = rain_clutter_snr_reduction_db(150e6, 20.0, 10000.0)
        xband = rain_clutter_snr_reduction_db(10e9, 20.0, 10000.0)
        assert xband > vhf

    def test_capped_at_20db(self):
        reduction = rain_clutter_snr_reduction_db(60e9, 100.0, 100000.0)
        assert reduction <= 20.0


class TestClutterFalseAlarmRate:
    def test_no_clutter_same_rate(self):
        result = clutter_false_alarm_rate_multiplier(0.01, elevation_angle_deg=10.0)
        assert result == 0.01

    def test_low_elevation_increases_rate(self):
        result = clutter_false_alarm_rate_multiplier(0.01, elevation_angle_deg=0.0)
        assert result > 0.01

    def test_sea_state_increases_rate(self):
        calm = clutter_false_alarm_rate_multiplier(0.01, elevation_angle_deg=0.0, sea_state=0)
        rough = clutter_false_alarm_rate_multiplier(0.01, elevation_angle_deg=0.0, sea_state=6)
        assert rough > calm

    def test_rain_increases_rate(self):
        dry = clutter_false_alarm_rate_multiplier(0.01, rain_rate_mm_h=0.0)
        wet = clutter_false_alarm_rate_multiplier(0.01, rain_rate_mm_h=20.0)
        assert wet > dry


# ===================================================================
# EnvironmentModel
# ===================================================================


class TestEnvironmentModelDefaults:
    def test_all_features_off_by_default(self):
        env = EnvironmentModel()
        assert not env.use_terrain_masking
        assert not env.use_atmospheric_propagation
        assert not env.use_weather_effects
        assert not env.use_clutter

    def test_visible_when_masking_off(self):
        env = EnvironmentModel()
        assert env.is_target_visible(5000.0, 0.0) is True

    def test_zero_snr_adjustment_when_off(self):
        env = EnvironmentModel()
        assert env.radar_snr_adjustment_db(10e9, 10000.0) == 0.0

    def test_full_thermal_factor_when_off(self):
        env = EnvironmentModel()
        assert env.thermal_detection_factor(ThermalBand.MWIR, 10000.0) == 1.0

    def test_unchanged_max_range_when_off(self):
        env = EnvironmentModel()
        assert env.effective_thermal_max_range(50000.0) == 50000.0

    def test_unchanged_far_when_off(self):
        env = EnvironmentModel()
        assert env.effective_false_alarm_rate(0.01) == 0.01


class TestEnvironmentModelTerrainMasking:
    def test_flat_terrain_visible(self):
        env = EnvironmentModel(
            terrain=TerrainGrid.flat(20000.0),
            use_terrain_masking=True,
            sensor_position=(0.0, 0.0, 10.0),
        )
        assert env.is_target_visible(5000.0, 0.0, 10.0) is True

    def test_ridge_blocks_target(self):
        env = EnvironmentModel(
            terrain=TerrainGrid.ridge(
                extent_m=20000.0, ridge_x_m=3000.0,
                ridge_height_m=500.0, ridge_width_m=200.0,
            ),
            use_terrain_masking=True,
            sensor_position=(0.0, 0.0, 0.0),
        )
        assert env.is_target_visible(5000.0, 0.0, 0.0) is False

    def test_masking_off_ignores_terrain(self):
        env = EnvironmentModel(
            terrain=TerrainGrid.ridge(
                extent_m=20000.0, ridge_x_m=3000.0,
                ridge_height_m=500.0, ridge_width_m=200.0,
            ),
            use_terrain_masking=False,
            sensor_position=(0.0, 0.0, 0.0),
        )
        assert env.is_target_visible(5000.0, 0.0, 0.0) is True


class TestEnvironmentModelAtmosphere:
    def test_snr_reduced_with_atmosphere(self):
        env = EnvironmentModel(
            weather=WeatherConditions(rain_rate_mm_h=16.0),
            use_atmospheric_propagation=True,
        )
        adj = env.radar_snr_adjustment_db(10e9, 15000.0)
        assert adj < 0.0

    def test_vhf_less_affected_than_xband(self):
        env = EnvironmentModel(
            weather=WeatherConditions(rain_rate_mm_h=16.0),
            use_atmospheric_propagation=True,
        )
        vhf_adj = env.radar_snr_adjustment_db(BAND_CENTER_FREQ_HZ[RadarBand.VHF], 15000.0)
        x_adj = env.radar_snr_adjustment_db(BAND_CENTER_FREQ_HZ[RadarBand.X_BAND], 15000.0)
        assert x_adj < vhf_adj  # More negative = more loss

    def test_thermal_factor_reduced(self):
        env = EnvironmentModel(
            weather=WeatherConditions(humidity_pct=80.0, rain_rate_mm_h=10.0),
            use_atmospheric_propagation=True,
        )
        factor = env.thermal_detection_factor(ThermalBand.MWIR, 20000.0)
        assert factor < 1.0


class TestEnvironmentModelWeather:
    def test_cloud_cover_reduces_thermal_factor(self):
        env = EnvironmentModel(
            weather=WeatherConditions(cloud_cover_pct=100.0),
            use_weather_effects=True,
        )
        factor = env.thermal_detection_factor(ThermalBand.MWIR, 1000.0)
        assert factor < 1.0

    def test_poor_visibility_reduces_max_range(self):
        env = EnvironmentModel(
            weather=WeatherConditions(visibility_km=2.0),
            use_weather_effects=True,
        )
        effective = env.effective_thermal_max_range(50000.0)
        assert effective < 50000.0


class TestEnvironmentModelClutter:
    def test_clutter_increases_far(self):
        env = EnvironmentModel(
            weather=WeatherConditions(sea_state=5),
            use_clutter=True,
        )
        far = env.effective_false_alarm_rate(0.01, elevation_angle_deg=0.0)
        assert far > 0.01

    def test_snr_reduced_by_clutter(self):
        env = EnvironmentModel(
            weather=WeatherConditions(sea_state=5, rain_rate_mm_h=10.0),
            use_clutter=True,
        )
        adj = env.radar_snr_adjustment_db(10e9, 10000.0, elevation_angle_deg=0.0)
        assert adj < 0.0


class TestEnvironmentModelFromOmegaconf:
    def test_all_disabled_by_default(self):
        cfg = OmegaConf.create({})
        env = EnvironmentModel.from_omegaconf(cfg)
        assert not env.use_terrain_masking
        assert not env.use_atmospheric_propagation
        assert not env.use_weather_effects
        assert not env.use_clutter

    def test_terrain_enabled(self):
        cfg = OmegaConf.create({
            "terrain": {"enabled": True, "type": "flat", "extent_m": 10000.0},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.use_terrain_masking is True
        assert env.terrain is not None

    def test_procedural_terrain(self):
        cfg = OmegaConf.create({
            "terrain": {"enabled": True, "type": "procedural", "max_elevation_m": 200.0, "seed": 7},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.terrain is not None
        assert np.max(env.terrain.elevation_data) > 0.0

    def test_weather_config(self):
        cfg = OmegaConf.create({
            "weather": {"enabled": True, "rain_rate_mm_h": 8.0, "visibility_km": 5.0},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.use_weather_effects is True
        assert env.weather.rain_rate_mm_h == 8.0
        assert env.weather.visibility_km == 5.0

    def test_atmosphere_config(self):
        cfg = OmegaConf.create({
            "atmosphere": {"enabled": True},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.use_atmospheric_propagation is True

    def test_clutter_config(self):
        cfg = OmegaConf.create({
            "clutter": {"enabled": True},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.use_clutter is True

    def test_sensor_position(self):
        cfg = OmegaConf.create({
            "sensor_position": {"x_m": 100.0, "y_m": 200.0, "altitude_m": 50.0},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.sensor_position == (100.0, 200.0, 50.0)

    def test_full_config(self):
        cfg = OmegaConf.create({
            "terrain": {"enabled": True, "type": "flat"},
            "weather": {"enabled": True, "rain_rate_mm_h": 4.0, "cloud_cover_pct": 50.0},
            "atmosphere": {"enabled": True},
            "clutter": {"enabled": True},
            "sensor_position": {"x_m": 0.0, "y_m": 0.0, "altitude_m": 10.0},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.use_terrain_masking
        assert env.use_atmospheric_propagation
        assert env.use_weather_effects
        assert env.use_clutter
        assert env.weather.rain_rate_mm_h == 4.0


# ===================================================================
# Band center frequencies
# ===================================================================


class TestBandCenterFrequencies:
    def test_all_bands_present(self):
        for band in RadarBand:
            assert band in BAND_CENTER_FREQ_HZ

    def test_vhf_lowest(self):
        freqs = list(BAND_CENTER_FREQ_HZ.values())
        assert BAND_CENTER_FREQ_HZ[RadarBand.VHF] == min(freqs)

    def test_xband_highest(self):
        freqs = list(BAND_CENTER_FREQ_HZ.values())
        assert BAND_CENTER_FREQ_HZ[RadarBand.X_BAND] == max(freqs)
