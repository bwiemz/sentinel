"""Tests for geodetic output on tracks."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.utils.geo_context import GeoContext
from sentinel.core.types import Detection, SensorType


# ── Helper ──────────────────────────────────────────────────────


def _make_geo_context():
    return GeoContext(lat0_deg=38.8977, lon0_deg=-77.0365, alt0_m=0.0, name="DC")


def _radar_detection(range_m=5000.0, azimuth_deg=45.0, ts=0.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=0.0,
        rcs_dbsm=10.0,
    )


def _thermal_detection(azimuth_deg=30.0, ts=0.0):
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=ts,
        azimuth_deg=azimuth_deg,
        temperature_k=350.0,
    )


# ── RadarTrack with GeoContext ──────────────────────────────────


class TestRadarTrackGeodetic:
    def test_position_geo_none_without_context(self):
        from sentinel.tracking.radar_track import RadarTrack
        track = RadarTrack(_radar_detection(), dt=0.1)
        assert track.position_geo is None

    def test_position_geo_with_context(self):
        from sentinel.tracking.radar_track import RadarTrack
        gc = _make_geo_context()
        track = RadarTrack(_radar_detection(), dt=0.1, geo_context=gc)
        geo = track.position_geo
        assert geo is not None
        lat, lon, alt = geo
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(alt, float)

    def test_position_geo_roundtrip(self):
        """Track at known ENU → geodetic should match reference."""
        gc = _make_geo_context()
        from sentinel.tracking.radar_track import RadarTrack
        track = RadarTrack(_radar_detection(range_m=1000, azimuth_deg=0), dt=0.1, geo_context=gc)
        pos = track.position
        lat, lon, alt = track.position_geo
        # Convert back to ENU and verify
        enu = gc.geodetic_to_enu(lat, lon, alt)
        assert enu[0] == pytest.approx(pos[0], abs=1.0)
        assert enu[1] == pytest.approx(pos[1], abs=1.0)

    def test_to_dict_without_context(self):
        from sentinel.tracking.radar_track import RadarTrack
        track = RadarTrack(_radar_detection(), dt=0.1)
        d = track.to_dict()
        assert "position_geo" not in d

    def test_to_dict_with_context(self):
        from sentinel.tracking.radar_track import RadarTrack
        gc = _make_geo_context()
        track = RadarTrack(_radar_detection(), dt=0.1, geo_context=gc)
        d = track.to_dict()
        assert "position_geo" in d
        assert "lat" in d["position_geo"]
        assert "lon" in d["position_geo"]
        assert "alt" in d["position_geo"]

    def test_geo_context_property(self):
        from sentinel.tracking.radar_track import RadarTrack
        gc = _make_geo_context()
        track = RadarTrack(_radar_detection(), dt=0.1, geo_context=gc)
        assert track.geo_context is gc

    def test_geo_context_setter(self):
        from sentinel.tracking.radar_track import RadarTrack
        track = RadarTrack(_radar_detection(), dt=0.1)
        assert track.position_geo is None
        gc = _make_geo_context()
        track.geo_context = gc
        assert track.position_geo is not None


# ── ThermalTrack with GeoContext ────────────────────────────────


class TestThermalTrackGeodetic:
    def test_position_geo_none_without_context(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        track = ThermalTrack(_thermal_detection(), dt=0.033)
        assert track.position_geo is None

    def test_position_geo_with_context(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        gc = _make_geo_context()
        track = ThermalTrack(_thermal_detection(), dt=0.033, geo_context=gc)
        geo = track.position_geo
        assert geo is not None
        lat, lon, alt = geo

    def test_to_dict_without_context(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        track = ThermalTrack(_thermal_detection(), dt=0.033)
        d = track.to_dict()
        assert "position_geo" not in d

    def test_to_dict_with_context(self):
        from sentinel.tracking.thermal_track import ThermalTrack
        gc = _make_geo_context()
        track = ThermalTrack(_thermal_detection(), dt=0.033, geo_context=gc)
        d = track.to_dict()
        assert "position_geo" in d


# ── Track Manager GeoContext Propagation ────────────────────────


class TestTrackManagerGeoContext:
    def test_radar_manager_passes_geo_context(self):
        from omegaconf import OmegaConf
        from sentinel.tracking.radar_track_manager import RadarTrackManager
        cfg = OmegaConf.create({
            "filter": {"dt": 0.1, "type": "ekf"},
            "association": {"gate_threshold": 9.21},
            "track_management": {"confirm_hits": 2, "max_coast_frames": 5, "max_tracks": 50},
        })
        gc = _make_geo_context()
        mgr = RadarTrackManager(cfg, geo_context=gc)
        dets = [_radar_detection(range_m=5000, azimuth_deg=30)]
        mgr.step(dets)
        if mgr.active_tracks:
            assert mgr.active_tracks[0].geo_context is gc

    def test_thermal_manager_passes_geo_context(self):
        from omegaconf import OmegaConf
        from sentinel.tracking.thermal_track_manager import ThermalTrackManager
        cfg = OmegaConf.create({
            "filter": {"type": "bearing_ekf", "dt": 0.033, "assumed_initial_range_m": 10000.0},
            "association": {"gate_threshold": 6.635},
            "track_management": {"confirm_hits": 2, "max_coast_frames": 5, "max_tracks": 50},
        })
        gc = _make_geo_context()
        mgr = ThermalTrackManager(cfg, geo_context=gc)
        dets = [_thermal_detection(azimuth_deg=30)]
        mgr.step(dets)
        if mgr.active_tracks:
            assert mgr.active_tracks[0].geo_context is gc
