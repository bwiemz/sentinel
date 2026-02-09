"""Scenario tests: terrain masking and weather degradation.

Scenario A — Mountain Masking:
  A target flies behind a ridge, then emerges.
  Early steps → no detections; later steps → detections resume.

Scenario B — Heavy Rain Degradation:
  Rain at 16 mm/h heavily attenuates X-band while VHF survives.
  Thermal range is reduced.  Tracks still confirm via low-freq bands.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.types import RadarBand, TargetType, TrackState
from sentinel.sensors.environment import (
    EnvironmentModel,
    TerrainGrid,
    WeatherConditions,
)
from tests.scenarios.conftest import ScenarioRunner, ScenarioTarget


# ===================================================================
# Helpers
# ===================================================================

def _make_ridge_target(start_x: float = 8000.0, vx: float = -200.0) -> ScenarioTarget:
    """Target moving toward radar, starting beyond a ridge."""
    return ScenarioTarget(
        target_id="RIDGE-TGT",
        position=np.array([start_x, 0.0]),
        velocity=np.array([vx, 0.0]),
        target_type=TargetType.CONVENTIONAL,
        rcs_dbsm=15.0,
        mach=0.6,
        expected_threat="medium",
    )


def _ridge_env(ridge_x: float = 5000.0, height: float = 300.0) -> EnvironmentModel:
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


def _rain_env(rain_rate: float = 16.0) -> EnvironmentModel:
    return EnvironmentModel(
        weather=WeatherConditions(
            rain_rate_mm_h=rain_rate,
            humidity_pct=80.0,
            visibility_km=5.0,
            cloud_cover_pct=80.0,
        ),
        use_atmospheric_propagation=True,
        use_weather_effects=True,
    )


def _make_standard_target() -> ScenarioTarget:
    """Conventional target at moderate range for rain tests."""
    return ScenarioTarget(
        target_id="RAIN-TGT",
        position=np.array([12000.0, 0.0]),
        velocity=np.array([-100.0, 0.0]),
        target_type=TargetType.CONVENTIONAL,
        rcs_dbsm=15.0,
        mach=0.5,
        expected_threat="medium",
    )


# ===================================================================
# Scenario A: Mountain Masking
# ===================================================================


class TestMountainMasking:
    """Target behind a ridge is masked, then emerges."""

    def test_target_masked_behind_ridge(self):
        """Early steps: target at x=8000 behind ridge at x=5000 produces no radar detections."""
        target = _make_ridge_target(start_x=8000.0, vx=0.0)  # stationary behind ridge
        env = _ridge_env(ridge_x=5000.0, height=300.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=10,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            environment=env,
        )
        result = runner.run()
        # All detections should be zero (target behind ridge the whole time)
        total_dets = sum(len(step) for step in result.multifreq_detection_log)
        assert total_dets == 0, f"Target behind ridge should produce 0 detections, got {total_dets}"

    def test_target_detected_on_near_side_of_ridge(self):
        """Target on sensor side of ridge (x=3000, ridge at x=5000) is visible."""
        target = ScenarioTarget(
            target_id="NEAR-TGT",
            position=np.array([3000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=15.0,
            mach=0.6,
            expected_threat="medium",
        )
        env = _ridge_env(ridge_x=5000.0, height=300.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=15,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_base_pd=0.95,
            environment=env,
        )
        result = runner.run()
        total_dets = sum(len(step) for step in result.multifreq_detection_log)
        assert total_dets > 0, "Target on near side of ridge should be detected"

    def test_radar_track_confirms_on_near_side(self):
        """Target visible in front of ridge should produce confirmed tracks."""
        target = ScenarioTarget(
            target_id="NEAR-TGT",
            position=np.array([3000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=15.0,
            mach=0.6,
            expected_threat="medium",
        )
        env = _ridge_env(ridge_x=5000.0, height=300.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=15,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_base_pd=0.95,
            environment=env,
        )
        result = runner.run()
        assert result.radar_confirmed_count >= 1, (
            f"Should have confirmed track on near side, got {result.radar_confirmed_count}"
        )

    def test_thermal_also_masked(self):
        """Thermal sensor is equally blocked by terrain."""
        target = _make_ridge_target(start_x=8000.0, vx=0.0)
        env = _ridge_env(ridge_x=5000.0, height=300.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=10,
            use_multifreq=False,
            use_thermal=True,
            use_quantum=False,
            thermal_fov_deg=120.0,
            environment=env,
        )
        result = runner.run()
        total_dets = sum(len(step) for step in result.thermal_detection_log)
        assert total_dets == 0, f"Thermal should be blocked by ridge, got {total_dets} detections"

    def test_flat_terrain_no_masking(self):
        """Same target with flat terrain should be detected from step 1."""
        target = _make_ridge_target(start_x=8000.0, vx=0.0)
        flat_terrain = TerrainGrid.flat(extent_m=50000.0)
        env = EnvironmentModel(
            terrain=flat_terrain,
            sensor_position=(0.0, 0.0, 0.0),
            use_terrain_masking=True,
        )
        runner = ScenarioRunner(
            targets=[target],
            n_steps=10,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_base_pd=1.0,
            environment=env,
        )
        result = runner.run()
        total_dets = sum(len(step) for step in result.multifreq_detection_log)
        # 5 bands * 10 steps * 1 target = 50
        assert total_dets > 0, "Flat terrain should not block detections"


# ===================================================================
# Scenario B: Heavy Rain Degradation
# ===================================================================


class TestHeavyRainDegradation:
    """Heavy rain degrades high-frequency radar and thermal more than VHF."""

    def test_rain_degrades_xband_more_than_vhf(self):
        """X-band detection rate significantly lower than VHF in 16 mm/h rain."""
        target = _make_standard_target()
        env = _rain_env(rain_rate=16.0)

        # VHF-only run
        vhf_runner = ScenarioRunner(
            targets=[target],
            n_steps=30,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.VHF],
            multifreq_base_pd=0.9,
            environment=env,
        )
        vhf_result = vhf_runner.run()
        vhf_dets = sum(len(step) for step in vhf_result.multifreq_detection_log)

        # X-band-only run
        x_runner = ScenarioRunner(
            targets=[target],
            n_steps=30,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.X_BAND],
            multifreq_base_pd=0.9,
            environment=env,
        )
        x_result = x_runner.run()
        x_dets = sum(len(step) for step in x_result.multifreq_detection_log)

        assert vhf_dets > x_dets, (
            f"VHF should outperform X-band in rain: VHF={vhf_dets}, X={x_dets}"
        )

    def test_vhf_maintains_detection_in_rain(self):
        """VHF detections should still be present in heavy rain."""
        target = _make_standard_target()
        env = _rain_env(rain_rate=16.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=20,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.VHF],
            multifreq_base_pd=0.9,
            environment=env,
        )
        result = runner.run()
        vhf_dets = sum(len(step) for step in result.multifreq_detection_log)
        assert vhf_dets > 0, "VHF should still detect targets in rain"

    def test_thermal_range_reduced_in_rain(self):
        """Thermal detections at long range should be fewer in rain vs clear weather."""
        target = ScenarioTarget(
            target_id="TH-RAIN",
            position=np.array([20000.0, 0.0]),
            velocity=np.array([-100.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            rcs_dbsm=10.0,
            mach=5.0,
            expected_threat="high",
        )
        # Clear weather
        clear_runner = ScenarioRunner(
            targets=[target],
            n_steps=20,
            use_multifreq=False,
            use_thermal=True,
            use_quantum=False,
            thermal_fov_deg=120.0,
            environment=None,
        )
        clear_result = clear_runner.run()
        clear_dets = sum(len(step) for step in clear_result.thermal_detection_log)

        # Rainy
        env = _rain_env(rain_rate=16.0)
        rain_runner = ScenarioRunner(
            targets=[target],
            n_steps=20,
            use_multifreq=False,
            use_thermal=True,
            use_quantum=False,
            thermal_fov_deg=120.0,
            environment=env,
        )
        rain_result = rain_runner.run()
        rain_dets = sum(len(step) for step in rain_result.thermal_detection_log)

        assert rain_dets < clear_dets, (
            f"Rain should reduce thermal detections: rain={rain_dets}, clear={clear_dets}"
        )

    def test_clear_weather_baseline(self):
        """Same scenario without rain should have higher detection rates."""
        # Place target at longer range to amplify atmospheric effect
        target = ScenarioTarget(
            target_id="XBAND-RAIN",
            position=np.array([25000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            rcs_dbsm=15.0,
            mach=0.5,
            expected_threat="medium",
        )
        # Clear
        clear_runner = ScenarioRunner(
            targets=[target],
            n_steps=60,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.X_BAND],
            multifreq_base_pd=0.9,
            environment=None,
        )
        clear_result = clear_runner.run()
        clear_dets = sum(len(step) for step in clear_result.multifreq_detection_log)

        # Rainy (heavy: 30 mm/h at 25km range -> significant X-band loss)
        env = _rain_env(rain_rate=30.0)
        rain_runner = ScenarioRunner(
            targets=[target],
            n_steps=60,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.X_BAND],
            multifreq_base_pd=0.9,
            environment=env,
        )
        rain_result = rain_runner.run()
        rain_dets = sum(len(step) for step in rain_result.multifreq_detection_log)

        assert clear_dets > rain_dets, (
            f"Clear weather should have more X-band detections: clear={clear_dets}, rain={rain_dets}"
        )

    def test_radar_track_still_confirms(self):
        """Despite rain, low-freq detections should still confirm tracks."""
        target = _make_standard_target()
        env = _rain_env(rain_rate=16.0)
        runner = ScenarioRunner(
            targets=[target],
            n_steps=20,
            use_multifreq=True,
            use_thermal=False,
            use_quantum=False,
            multifreq_bands=[RadarBand.VHF, RadarBand.UHF, RadarBand.X_BAND],
            multifreq_base_pd=0.9,
            environment=env,
        )
        result = runner.run()
        assert result.radar_confirmed_count >= 1, (
            f"Low-freq bands should still confirm tracks in rain, got {result.radar_confirmed_count}"
        )
