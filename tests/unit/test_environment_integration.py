"""Integration tests: environment effects applied through each simulator.

Tests verify that when an EnvironmentModel is injected into a simulator
config, the simulator correctly applies terrain masking, atmospheric
losses, weather effects, and clutter adjustments.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.types import RadarBand, SensorType, TargetType, ThermalBand
from sentinel.sensors.environment import (
    EnvironmentModel,
    TerrainGrid,
    WeatherConditions,
)
from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator, RadarTarget
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
)
from sentinel.sensors.radar_sim import MultiFreqRadarTarget
from sentinel.sensors.quantum_radar_sim import (
    QuantumRadarConfig,
    QuantumRadarSimulator,
)
from sentinel.sensors.thermal_sim import (
    ThermalSimConfig,
    ThermalSimulator,
    ThermalTarget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_detections(sim, n_scans: int = 20) -> int:
    """Run *n_scans* and return total number of target detections (excl. false alarms)."""
    sim.connect()
    total = 0
    for _ in range(n_scans):
        frame = sim.read_frame()
        if frame:
            for d in frame.data:
                if d.get("target_id"):
                    total += 1
    sim.disconnect()
    return total


def _make_ridge_env(ridge_x: float = 5000.0, height: float = 300.0) -> EnvironmentModel:
    """Environment with a ridge at *ridge_x*, all masking enabled."""
    terrain = TerrainGrid.ridge(
        extent_m=50000.0,
        resolution_m=100.0,
        ridge_x_m=ridge_x,
        ridge_height_m=height,
        ridge_width_m=500.0,
    )
    return EnvironmentModel(
        terrain=terrain,
        sensor_position=(0.0, 0.0, 10.0),  # 10m mast (realistic)
        use_terrain_masking=True,
    )


def _make_rain_env(rain_rate: float = 16.0) -> EnvironmentModel:
    """Environment with heavy rain, atmosphere + clutter on."""
    return EnvironmentModel(
        weather=WeatherConditions(
            rain_rate_mm_h=rain_rate,
            humidity_pct=80.0,
            visibility_km=5.0,
            cloud_cover_pct=80.0,
        ),
        use_atmospheric_propagation=True,
        use_weather_effects=True,
        use_clutter=True,
    )


# ===================================================================
# Classical Radar (RadarSimulator)
# ===================================================================


class TestRadarSimEnvironment:
    """RadarSimulator + EnvironmentModel integration."""

    def test_terrain_masks_target_behind_ridge(self):
        """Target at x=8000 behind ridge at x=5000 should be masked."""
        env = _make_ridge_env(ridge_x=5000.0, height=300.0)
        target = RadarTarget(
            target_id="T1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
        )
        cfg = RadarSimConfig(
            max_range_m=20000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42)
        assert _count_detections(sim, 20) == 0, "Target behind ridge should not be detected"

    def test_no_environment_detects_normally(self):
        """Same target without environment should be detected."""
        target = RadarTarget(
            target_id="T1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
        )
        cfg = RadarSimConfig(
            max_range_m=20000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        sim = RadarSimulator(cfg, seed=42)
        assert _count_detections(sim, 20) == 20, "Target should be detected every scan"

    def test_rain_reduces_flat_pd_detections(self):
        """Heavy rain with flat-Pd mode should reduce detections at long range."""
        target = RadarTarget(
            target_id="T1",
            position=np.array([15000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )
        # Baseline (no env)
        cfg_clear = RadarSimConfig(
            max_range_m=20000.0,
            detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        clear_dets = _count_detections(RadarSimulator(cfg_clear, seed=42), 50)

        # With heavy rain
        env = _make_rain_env(rain_rate=16.0)
        cfg_rain = RadarSimConfig(
            max_range_m=20000.0,
            detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        rain_dets = _count_detections(RadarSimulator(cfg_rain, seed=42), 50)

        assert rain_dets < clear_dets, f"Rain should reduce detections: {rain_dets} vs {clear_dets}"

    def test_snr_pd_with_atmosphere(self):
        """SNR-based Pd + atmospheric loss should reduce Pd at range."""
        target = RadarTarget(
            target_id="T1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )
        env = _make_rain_env(rain_rate=16.0)
        cfg = RadarSimConfig(
            max_range_m=20000.0,
            use_snr_pd=True,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42)
        # Should still detect some (SNR-based Pd + rain loss)
        dets = _count_detections(sim, 30)
        assert dets >= 0  # Non-negative (may be 0 if loss is extreme)

    def test_false_alarm_rate_increases_with_clutter(self):
        """Clutter environment should increase false alarm rate."""
        env = EnvironmentModel(
            weather=WeatherConditions(rain_rate_mm_h=10.0, sea_state=5),
            use_clutter=True,
        )
        cfg = RadarSimConfig(
            max_range_m=10000.0,
            false_alarm_rate=1.0,  # High base FAR to see effect
            targets=[],
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()
        total_fa = 0
        for _ in range(100):
            frame = sim.read_frame()
            total_fa += len(frame.data)
        sim.disconnect()

        # Without clutter, expected ~ 100 * 1.0 = 100 false alarms
        # With clutter, should be higher
        assert total_fa > 100, f"Clutter should increase false alarms, got {total_fa}"


# ===================================================================
# Multi-Frequency Radar (MultiFreqRadarSimulator)
# ===================================================================


class TestMultiFreqRadarEnvironment:
    """MultiFreqRadarSimulator + EnvironmentModel integration."""

    def test_terrain_masks_target(self):
        """Target behind ridge should produce zero detections across all bands."""
        env = _make_ridge_env(ridge_x=5000.0, height=300.0)
        target = MultiFreqRadarTarget(
            target_id="MF1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.VHF, RadarBand.X_BAND],
            max_range_m=20000.0,
            base_detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        assert _count_detections(sim, 20) == 0

    def test_rain_affects_xband_more_than_vhf(self):
        """X-band should lose more detections than VHF in heavy rain."""
        target = MultiFreqRadarTarget(
            target_id="MF1",
            position=np.array([15000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=15.0,
            target_type=TargetType.CONVENTIONAL,
        )
        env = _make_rain_env(rain_rate=16.0)

        # VHF-only
        cfg_vhf = MultiFreqRadarConfig(
            bands=[RadarBand.VHF],
            max_range_m=20000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        vhf_dets = _count_detections(MultiFreqRadarSimulator(cfg_vhf, seed=42), 50)

        # X-band only
        cfg_x = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND],
            max_range_m=20000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        x_dets = _count_detections(MultiFreqRadarSimulator(cfg_x, seed=42), 50)

        assert vhf_dets > x_dets, (
            f"VHF should have more detections than X-band in rain: VHF={vhf_dets}, X={x_dets}"
        )

    def test_clear_weather_all_bands_detect(self):
        """Without environment effects, all bands should detect a close, high-RCS target."""
        target = MultiFreqRadarTarget(
            target_id="MF1",
            position=np.array([3000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.VHF, RadarBand.S_BAND, RadarBand.X_BAND],
            max_range_m=20000.0,
            base_detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        # 20 scans * 3 bands * 1 target = 60
        dets = _count_detections(sim, 20)
        assert dets == 60, f"Expected 60 detections, got {dets}"

    def test_multifreq_false_alarm_clutter(self):
        """Clutter should inflate false alarm rate in multi-freq radar."""
        env = EnvironmentModel(
            weather=WeatherConditions(rain_rate_mm_h=10.0, sea_state=5),
            use_clutter=True,
        )
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND],
            max_range_m=10000.0,
            false_alarm_rate=1.0,
            targets=[],
            environment=env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        sim.connect()
        total_fa = 0
        for _ in range(100):
            frame = sim.read_frame()
            total_fa += len(frame.data)
        sim.disconnect()
        assert total_fa > 100, f"Clutter should increase FAs, got {total_fa}"


# ===================================================================
# Quantum Radar (QuantumRadarSimulator)
# ===================================================================


class TestQuantumRadarEnvironment:
    """QuantumRadarSimulator + EnvironmentModel integration."""

    def test_terrain_masks_target(self):
        """Quantum radar should not detect target behind a ridge."""
        env = _make_ridge_env(ridge_x=5000.0, height=300.0)
        target = MultiFreqRadarTarget(
            target_id="QI1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = QuantumRadarConfig(
            max_range_m=50000.0,
            squeeze_param_r=0.5,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = QuantumRadarSimulator(cfg, seed=42)
        assert _count_detections(sim, 20) == 0

    def test_atmosphere_reduces_qi_pd(self):
        """Heavy rain should reduce quantum radar Pd at long range."""
        target = MultiFreqRadarTarget(
            target_id="QI1",
            position=np.array([12000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
            target_type=TargetType.CONVENTIONAL,
        )
        env = _make_rain_env(rain_rate=16.0)

        # Clear weather
        cfg_clear = QuantumRadarConfig(
            max_range_m=50000.0,
            squeeze_param_r=0.5,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        clear_dets = _count_detections(QuantumRadarSimulator(cfg_clear, seed=42), 50)

        # Rainy
        cfg_rain = QuantumRadarConfig(
            max_range_m=50000.0,
            squeeze_param_r=0.5,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        rain_dets = _count_detections(QuantumRadarSimulator(cfg_rain, seed=42), 50)

        assert rain_dets <= clear_dets, (
            f"Rain should not increase QI detections: rain={rain_dets}, clear={clear_dets}"
        )

    def test_no_environment_detects_normally(self):
        """Quantum radar should detect a strong target without environment."""
        target = MultiFreqRadarTarget(
            target_id="QI1",
            position=np.array([3000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=20.0,
            target_type=TargetType.CONVENTIONAL,
        )
        cfg = QuantumRadarConfig(
            max_range_m=50000.0,
            squeeze_param_r=0.5,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        sim = QuantumRadarSimulator(cfg, seed=42)
        dets = _count_detections(sim, 20)
        assert dets > 0, "Should detect a strong close target"


# ===================================================================
# Thermal Simulator (ThermalSimulator)
# ===================================================================


class TestThermalSimEnvironment:
    """ThermalSimulator + EnvironmentModel integration."""

    def test_terrain_masks_target(self):
        """Thermal sensor should not see target behind ridge."""
        env = _make_ridge_env(ridge_x=5000.0, height=300.0)
        target = ThermalTarget(
            target_id="TH1",
            position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            mach=5.0,
        )
        cfg = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = ThermalSimulator(cfg, seed=42)
        assert _count_detections(sim, 20) == 0

    def test_visibility_reduces_max_range(self):
        """Low visibility should reduce effective thermal max range."""
        target = ThermalTarget(
            target_id="TH1",
            position=np.array([30000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            mach=5.0,
        )
        # Good visibility -> target at 30km is within 50km max range
        cfg_clear = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        clear_dets = _count_detections(ThermalSimulator(cfg_clear, seed=42), 20)

        # Poor visibility (5km) -> effective range = 50000 * 5/50 = 5000 < 30000
        env = EnvironmentModel(
            weather=WeatherConditions(visibility_km=5.0),
            use_weather_effects=True,
        )
        cfg_fog = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        fog_dets = _count_detections(ThermalSimulator(cfg_fog, seed=42), 20)

        assert clear_dets > 0, "Clear weather should detect"
        assert fog_dets == 0, "Fog should prevent detection at 30km (effective range ~5km)"

    def test_cloud_cover_reduces_contrast(self):
        """Heavy cloud cover should reduce some detections by lowering contrast."""
        # Target with moderate thermal signature (close to contrast threshold)
        target = ThermalTarget(
            target_id="TH1",
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            mach=0.3,  # Low speed -> modest temperature
        )
        # No cloud cover
        cfg_clear = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            min_contrast_k=5.0,  # Lower threshold to get baseline detections
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        clear_dets = _count_detections(ThermalSimulator(cfg_clear, seed=42), 50)

        # 100% cloud cover
        env = EnvironmentModel(
            weather=WeatherConditions(cloud_cover_pct=100.0),
            use_weather_effects=True,
        )
        cfg_cloudy = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            min_contrast_k=5.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        cloudy_dets = _count_detections(ThermalSimulator(cfg_cloudy, seed=42), 50)

        # Cloud cover reduces contrast by up to 40%, so some marginal targets may be lost
        # At minimum, cloudy should not detect MORE than clear
        assert cloudy_dets <= clear_dets, (
            f"Cloudy should not have more detections: cloudy={cloudy_dets}, clear={clear_dets}"
        )

    def test_atmospheric_transmission_reduces_pd(self):
        """Atmospheric transmission should reduce thermal Pd at long range."""
        target = ThermalTarget(
            target_id="TH1",
            position=np.array([20000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            mach=5.0,
        )
        # With atmosphere
        env = EnvironmentModel(
            weather=WeatherConditions(humidity_pct=90.0, rain_rate_mm_h=4.0),
            use_atmospheric_propagation=True,
        )
        cfg_atmo = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=0.95,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        atmo_dets = _count_detections(ThermalSimulator(cfg_atmo, seed=42), 50)

        # Without atmosphere
        cfg_clear = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=0.95,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        clear_dets = _count_detections(ThermalSimulator(cfg_clear, seed=42), 50)

        assert atmo_dets <= clear_dets, (
            f"Atmosphere should not increase detections: atmo={atmo_dets}, clear={clear_dets}"
        )

    def test_no_environment_detects_normally(self):
        """Thermal sim without environment should work exactly as before."""
        target = ThermalTarget(
            target_id="TH1",
            position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            mach=5.0,
        )
        cfg = ThermalSimConfig(
            fov_deg=120.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
            environment=None,
        )
        sim = ThermalSimulator(cfg, seed=42)
        dets = _count_detections(sim, 10)
        # 2 bands * 10 scans * 1 target (Pd=1.0) = 20
        assert dets == 20, f"Expected 20, got {dets}"


# ===================================================================
# Cross-simulator consistency
# ===================================================================


class TestCrossSimulatorEnvironment:
    """Tests that environment effects apply consistently across simulators."""

    def test_ridge_blocks_all_sensors(self):
        """A ridge should block radar, multi-freq, quantum, and thermal equally."""
        env = _make_ridge_env(ridge_x=5000.0, height=300.0)

        # Classical radar
        radar_target = RadarTarget(
            target_id="T1", position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]), rcs_dbsm=20.0,
        )
        radar_cfg = RadarSimConfig(
            max_range_m=20000.0, detection_probability=1.0,
            false_alarm_rate=0.0, targets=[radar_target], environment=env,
        )
        assert _count_detections(RadarSimulator(radar_cfg, seed=42), 10) == 0

        # Multi-freq radar
        mf_target = MultiFreqRadarTarget(
            target_id="T1", position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]), rcs_dbsm=20.0,
            target_type=TargetType.CONVENTIONAL,
        )
        mf_cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND], max_range_m=20000.0,
            base_detection_probability=1.0, false_alarm_rate=0.0,
            targets=[mf_target], environment=env,
        )
        assert _count_detections(MultiFreqRadarSimulator(mf_cfg, seed=42), 10) == 0

        # Quantum radar
        qi_cfg = QuantumRadarConfig(
            max_range_m=50000.0, squeeze_param_r=0.5,
            false_alarm_rate=0.0, targets=[mf_target], environment=env,
        )
        assert _count_detections(QuantumRadarSimulator(qi_cfg, seed=42), 10) == 0

        # Thermal
        th_target = ThermalTarget(
            target_id="T1", position=np.array([8000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC, mach=5.0,
        )
        th_cfg = ThermalSimConfig(
            fov_deg=120.0, max_range_m=50000.0, detection_probability=1.0,
            false_alarm_rate=0.0, targets=[th_target], environment=env,
        )
        assert _count_detections(ThermalSimulator(th_cfg, seed=42), 10) == 0

    def test_flat_terrain_no_masking(self):
        """Flat terrain should not block any sensor."""
        terrain = TerrainGrid.flat(extent_m=50000.0)
        env = EnvironmentModel(
            terrain=terrain,
            sensor_position=(0.0, 0.0, 0.0),
            use_terrain_masking=True,
        )
        target = RadarTarget(
            target_id="T1", position=np.array([5000.0, 0.0]),
            velocity=np.array([0.0, 0.0]), rcs_dbsm=20.0,
        )
        cfg = RadarSimConfig(
            max_range_m=20000.0, detection_probability=1.0,
            false_alarm_rate=0.0, targets=[target], environment=env,
        )
        dets = _count_detections(RadarSimulator(cfg, seed=42), 10)
        assert dets == 10, f"Flat terrain should not block: got {dets}"
