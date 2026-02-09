"""End-to-end geodetic integration tests.

Tests the full pipeline with geodetic reference: simulator targets defined
in lat/lon, tracking in ENU, output converted back to geodetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.clock import SimClock
from sentinel.core.types import (
    Detection,
    RadarBand,
    SensorType,
    TargetType,
    TrackState,
)
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
    MultiFreqRadarTarget,
    multifreq_radar_frame_to_detections,
)
from sentinel.sensors.radar_sim import (
    RadarSimConfig,
    RadarSimulator,
    RadarTarget,
    radar_frame_to_detections,
)
from sentinel.sensors.thermal_sim import (
    ThermalSimConfig,
    ThermalSimulator,
    ThermalTarget,
    thermal_frame_to_detections,
)
from sentinel.tracking.radar_track_manager import RadarTrackManager
from sentinel.tracking.thermal_track_manager import ThermalTrackManager
from sentinel.utils.geo_context import GeoContext
from sentinel.utils.geodetic import haversine_distance


# ── Helpers ──────────────────────────────────────────────────────


DC_GEO = GeoContext(lat0_deg=38.8977, lon0_deg=-77.0365, alt0_m=0.0, name="DC")
TOKYO_GEO = GeoContext(lat0_deg=35.6762, lon0_deg=139.6503, alt0_m=0.0, name="Tokyo")


def _radar_cfg(**overrides):
    base = {
        "filter": {"dt": 0.1, "type": "ekf"},
        "association": {"gate_threshold": 9.21},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    }
    return OmegaConf.create({**base, **overrides})


def _thermal_cfg(**overrides):
    base = {
        "filter": {"type": "bearing_ekf", "dt": 0.1, "assumed_initial_range_m": 10000.0},
        "association": {"gate_threshold": 6.635},
        "track_management": {
            "confirm_hits": 2,
            "max_coast_frames": 5,
            "max_tracks": 50,
        },
    }
    return OmegaConf.create({**base, **overrides})


# ── Radar with GeoContext ────────────────────────────────────────


class TestRadarGeodeticE2E:
    """Full radar pipeline with geodetic context."""

    def test_target_position_geo_output(self):
        """A tracked target should report meaningful geodetic position."""
        gc = DC_GEO
        # Target 5 km east of DC
        target = RadarTarget("TGT-1", np.array([5000.0, 0.0]), np.array([0.0, 0.0]),
                             rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        track = confirmed[0]

        # Should have geodetic position
        geo = track.position_geo
        assert geo is not None
        lat, lon, alt = geo

        # Should be east of DC (lon > DC lon, lat ~= DC lat)
        assert lon > DC_GEO.lon0_deg  # east => less negative longitude
        assert lat == pytest.approx(DC_GEO.lat0_deg, abs=0.05)  # within ~5km latitude

        # Distance from origin should be ~5 km
        dist = haversine_distance(DC_GEO.lat0_deg, DC_GEO.lon0_deg, lat, lon)
        assert 4000 < dist < 6000

    def test_to_dict_has_geodetic(self):
        """Track to_dict should include position_geo when GeoContext is set."""
        gc = DC_GEO
        target = RadarTarget("TGT-1", np.array([3000.0, 2000.0]), np.array([0.0, 0.0]),
                             rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        d = confirmed[0].to_dict()
        assert "position_geo" in d
        assert "lat" in d["position_geo"]
        assert "lon" in d["position_geo"]
        assert "alt" in d["position_geo"]

    def test_no_geodetic_without_context(self):
        """Without GeoContext, tracks should not have position_geo."""
        target = RadarTarget("TGT-1", np.array([3000.0, 1000.0]), np.array([0.0, 0.0]),
                             rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg())  # no geo_context

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        assert confirmed[0].position_geo is None
        assert "position_geo" not in confirmed[0].to_dict()

    def test_geodetic_target_from_config_roundtrip(self):
        """Target defined at known lat/lon → tracked → output lat/lon matches."""
        gc = DC_GEO
        # Place target ~3 km east-northeast of DC (within radar FOV)
        target_lat = 38.8977 + (1000.0 / 111_000.0)  # ~1km north
        target_lon = -77.0365 + (3000.0 / (111_000.0 * np.cos(np.radians(38.8977))))  # ~3km east
        xy = gc.target_geodetic_to_xy(target_lat, target_lon)

        target = RadarTarget("TGT-GEO", xy, np.array([0.0, 0.0]), rcs_dbsm=20.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        lat, lon, alt = confirmed[0].position_geo

        # Should be close to original target coords (within ~500m)
        dist_to_target = haversine_distance(target_lat, target_lon, lat, lon)
        assert dist_to_target < 500.0

    def test_two_targets_different_geodetic_positions(self):
        """Two targets at different geodetic positions produce distinct geodetic outputs."""
        gc = DC_GEO
        # Both within FOV (azimuth < 60° from +x)
        # Target 1: 5 km at ~20° (mostly east, slightly north)
        # Target 2: 5 km at ~50° (more north, still in FOV)
        t1 = RadarTarget("TGT-E", np.array([4700.0, 1700.0]), np.array([0.0, 0.0]), rcs_dbsm=15.0)
        t2 = RadarTarget("TGT-N", np.array([3200.0, 3800.0]), np.array([0.0, 0.0]), rcs_dbsm=15.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[t1, t2],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 2

        geos = [t.position_geo for t in confirmed]
        assert all(g is not None for g in geos)

        # Geodetic positions should be distinct
        d = haversine_distance(geos[0][0], geos[0][1], geos[1][0], geos[1][1])
        assert d > 2000  # at least 2 km apart


# ── Thermal with GeoContext ──────────────────────────────────────


class TestThermalGeodeticE2E:
    """Thermal bearing-only pipeline with geodetic context."""

    def test_thermal_track_has_geodetic(self):
        """Thermal tracks should report geodetic position when context is set."""
        gc = DC_GEO
        clock = SimClock()
        target = ThermalTarget(
            target_id="TH-1",
            position=np.array([3000.0, 3000.0]),
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.CONVENTIONAL,
            mach=0.8,
        )
        cfg = ThermalSimConfig(
            fov_deg=90.0,
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = ThermalSimulator(cfg, seed=42, clock=clock)
        sim.connect()
        mgr = ThermalTrackManager(_thermal_cfg(), geo_context=gc)

        for _ in range(10):
            clock.step(0.1)
            frame = sim.read_frame()
            dets = thermal_frame_to_detections(frame)
            mgr.step(dets)

        active = mgr.active_tracks
        assert len(active) >= 1
        geo = active[0].position_geo
        assert geo is not None
        lat, lon, alt = geo
        # Should be NE of DC
        assert lat > DC_GEO.lat0_deg or lon > DC_GEO.lon0_deg


# ── MultiFreq Radar with GeoContext ──────────────────────────────


class TestMultiFreqGeodeticE2E:
    """Multi-frequency radar pipeline with geodetic context."""

    def test_multifreq_radar_geodetic(self):
        """MultiFreq radar + tracking with geo context produces geodetic output."""
        gc = TOKYO_GEO
        clock = SimClock()
        target = MultiFreqRadarTarget(
            target_id="MF-1",
            position=np.array([8000.0, 2000.0]),
            velocity=np.array([-100.0, 0.0]),
            rcs_dbsm=10.0,
            target_type=TargetType.CONVENTIONAL,
            mach=0.85,
        )
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND],
            max_range_m=50000.0,
            fov_deg=120.0,
            base_detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42, clock=clock)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            clock.step(0.1)
            frame = sim.read_frame()
            dets = multifreq_radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        geo = confirmed[0].position_geo
        assert geo is not None
        lat, lon, alt = geo
        # Should be near Tokyo (within ~50km)
        dist = haversine_distance(TOKYO_GEO.lat0_deg, TOKYO_GEO.lon0_deg, lat, lon)
        assert dist < 50000


# ── ScenarioRunner with GeoContext ───────────────────────────────


class TestScenarioRunnerGeodetic:
    """ScenarioRunner with geodetic target specification and output."""

    def test_scenario_with_geo_context(self):
        """ScenarioRunner with geo_context gives tracks with geodetic positions."""
        from tests.scenarios.conftest import ScenarioRunner, ScenarioTarget

        gc = DC_GEO
        targets = [
            ScenarioTarget(
                target_id="GEO-1",
                position=np.array([5000.0, 3000.0]),
                velocity=np.array([-50.0, 10.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=10.0,
                mach=0.85,
                expected_threat="MEDIUM",
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=15,
            use_thermal=True,
            use_quantum=False,
            multifreq_base_pd=1.0,
            geo_context=gc,
        )
        result = runner.run()

        # Radar tracks should have geodetic positions
        for track in result.radar_tracks:
            if hasattr(track, "position_geo") and track.geo_context is not None:
                geo = track.position_geo
                assert geo is not None
                dist = haversine_distance(DC_GEO.lat0_deg, DC_GEO.lon0_deg,
                                          geo[0], geo[1])
                assert dist < 20000  # within 20km

    def test_scenario_with_position_geo_targets(self):
        """Targets specified via position_geo are resolved to ENU."""
        from tests.scenarios.conftest import ScenarioRunner, ScenarioTarget

        gc = DC_GEO
        target_lat = 38.92  # ~2.5 km north of DC
        target_lon = -77.00  # ~3.1 km east of DC
        targets = [
            ScenarioTarget(
                target_id="GEO-POS",
                position=np.array([0.0, 0.0]),  # placeholder, will be overwritten
                velocity=np.array([0.0, 0.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=15.0,
                mach=0.5,
                expected_threat="MEDIUM",
                position_geo=(target_lat, target_lon, 0.0),
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=15,
            use_thermal=False,
            use_quantum=False,
            multifreq_base_pd=1.0,
            geo_context=gc,
        )

        # After init, target position should be resolved from geodetic
        resolved = runner.targets[0].position
        assert np.linalg.norm(resolved) > 100  # not still at origin

        result = runner.run()
        assert result.radar_confirmed_count >= 1

        # Confirmed tracks should be near the geodetic target
        for track in result.radar_tracks:
            if hasattr(track, "position_geo") and track.geo_context is not None:
                geo = track.position_geo
                if geo is not None:
                    dist = haversine_distance(target_lat, target_lon, geo[0], geo[1])
                    assert dist < 2000  # within 2km of target

    def test_scenario_without_geo_context_unchanged(self):
        """Without geo_context, ScenarioRunner behaves exactly as before."""
        from tests.scenarios.conftest import ScenarioRunner, ScenarioTarget

        targets = [
            ScenarioTarget(
                target_id="PLAIN-1",
                position=np.array([5000.0, 3000.0]),
                velocity=np.array([-50.0, 10.0]),
                target_type=TargetType.CONVENTIONAL,
                rcs_dbsm=10.0,
                mach=0.85,
                expected_threat="MEDIUM",
            ),
        ]
        runner = ScenarioRunner(
            targets=targets,
            n_steps=15,
            use_thermal=False,
            use_quantum=False,
            multifreq_base_pd=1.0,
            # no geo_context
        )
        result = runner.run()
        assert result.radar_confirmed_count >= 1

        # Tracks should NOT have geodetic positions
        for track in result.radar_tracks:
            assert track.position_geo is None


# ── GeoContext at Different Locations ────────────────────────────


class TestGeoContextLocations:
    """Verify geodetic output at different reference locations."""

    def test_equator_reference(self):
        """GeoContext at equator/prime meridian produces correct geodetic."""
        gc = GeoContext(lat0_deg=0.0, lon0_deg=0.0, alt0_m=0.0, name="Equator")
        target = RadarTarget("EQ-1", np.array([10000.0, 0.0]), np.array([0.0, 0.0]),
                             rcs_dbsm=20.0)
        sim_cfg = RadarSimConfig(
            max_range_m=20000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        lat, lon, alt = confirmed[0].position_geo
        # 10km east on equator → lon ~0.09°, lat ~0°
        assert abs(lat) < 0.01
        assert lon > 0.05
        assert lon < 0.15

    def test_high_latitude_reference(self):
        """GeoContext at high latitude (Norway) works correctly."""
        gc = GeoContext(lat0_deg=70.0, lon0_deg=25.0, alt0_m=0.0, name="Norway")
        # Target 5 km at ~30° (within FOV, northeast)
        target = RadarTarget("NO-1", np.array([4330.0, 2500.0]), np.array([0.0, 0.0]),
                             rcs_dbsm=20.0)
        sim_cfg = RadarSimConfig(
            max_range_m=10000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim = RadarSimulator(sim_cfg, seed=42)
        sim.connect()
        mgr = RadarTrackManager(_radar_cfg(), geo_context=gc)

        for _ in range(10):
            frame = sim.read_frame()
            dets = radar_frame_to_detections(frame)
            mgr.step(dets)

        confirmed = mgr.confirmed_tracks
        assert len(confirmed) >= 1
        lat, lon, alt = confirmed[0].position_geo
        # Target is northeast → lat should increase, lon should increase
        assert lat > 70.0
        assert lon > 25.0
        # Distance from origin should be ~5 km
        dist = haversine_distance(70.0, 25.0, lat, lon)
        assert 3000 < dist < 7000
