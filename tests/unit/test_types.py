"""Tests for core data types."""

import numpy as np

from sentinel.core.types import (
    Detection,
    RadarBand,
    SensorType,
    TargetType,
    ThermalBand,
    TrackState,
    generate_track_id,
)


class TestDetection:
    def test_camera_detection(self):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=100.0,
            bbox=np.array([10, 20, 110, 120], dtype=np.float32),
            class_id=0,
            class_name="person",
            confidence=0.95,
        )
        assert det.sensor_type == SensorType.CAMERA
        assert det.class_name == "person"
        assert det.confidence == 0.95

    def test_bbox_center(self):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=0,
            bbox=np.array([100, 200, 300, 400], dtype=np.float32),
        )
        center = det.bbox_center
        assert center is not None
        np.testing.assert_array_almost_equal(center, [200, 300])

    def test_bbox_center_none(self):
        det = Detection(sensor_type=SensorType.RADAR, timestamp=0)
        assert det.bbox_center is None

    def test_bbox_area(self):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=0,
            bbox=np.array([0, 0, 100, 50], dtype=np.float32),
        )
        assert det.bbox_area == 5000.0

    def test_radar_detection(self):
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=200.0,
            range_m=5000.0,
            azimuth_deg=45.0,
            velocity_mps=120.0,
            rcs_dbsm=10.0,
        )
        assert det.sensor_type == SensorType.RADAR
        assert det.range_m == 5000.0

    def test_to_dict(self):
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=1.0,
            bbox=np.array([10, 20, 30, 40], dtype=np.float32),
            class_name="car",
            confidence=0.8,
        )
        d = det.to_dict()
        assert d["sensor_type"] == "camera"
        assert d["class_name"] == "car"
        assert "bbox" in d


    def test_thermal_detection(self):
        det = Detection(
            sensor_type=SensorType.THERMAL,
            timestamp=300.0,
            azimuth_deg=10.0,
            elevation_deg=2.5,
            temperature_k=1800.0,
            thermal_band="mwir",
            intensity=0.95,
        )
        assert det.sensor_type == SensorType.THERMAL
        assert det.temperature_k == 1800.0
        assert det.thermal_band == "mwir"
        assert det.elevation_deg == 2.5
        assert det.range_m is None  # bearing-only

    def test_multifreq_radar_detection(self):
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=200.0,
            range_m=3000.0,
            azimuth_deg=15.0,
            velocity_mps=100.0,
            rcs_dbsm=-10.0,
            radar_band="vhf",
        )
        assert det.radar_band == "vhf"
        assert det.rcs_dbsm == -10.0

    def test_thermal_to_dict(self):
        det = Detection(
            sensor_type=SensorType.THERMAL,
            timestamp=1.0,
            azimuth_deg=5.0,
            elevation_deg=1.0,
            temperature_k=2000.0,
            thermal_band="lwir",
        )
        d = det.to_dict()
        assert d["sensor_type"] == "thermal"
        assert d["temperature_k"] == 2000.0
        assert d["thermal_band"] == "lwir"
        assert d["elevation_deg"] == 1.0

    def test_radar_band_to_dict(self):
        det = Detection(
            sensor_type=SensorType.RADAR,
            timestamp=1.0,
            range_m=5000.0,
            azimuth_deg=10.0,
            velocity_mps=50.0,
            radar_band="x_band",
        )
        d = det.to_dict()
        assert d["radar_band"] == "x_band"


class TestRadarBand:
    def test_bands_exist(self):
        assert RadarBand.VHF.value == "vhf"
        assert RadarBand.UHF.value == "uhf"
        assert RadarBand.L_BAND.value == "l_band"
        assert RadarBand.S_BAND.value == "s_band"
        assert RadarBand.X_BAND.value == "x_band"

    def test_band_count(self):
        assert len(RadarBand) == 5


class TestThermalBand:
    def test_bands_exist(self):
        assert ThermalBand.SWIR.value == "swir"
        assert ThermalBand.MWIR.value == "mwir"
        assert ThermalBand.LWIR.value == "lwir"


class TestTargetType:
    def test_types_exist(self):
        assert TargetType.CONVENTIONAL.value == "conventional"
        assert TargetType.STEALTH.value == "stealth"
        assert TargetType.HYPERSONIC.value == "hypersonic"


class TestTrackState:
    def test_states_exist(self):
        assert TrackState.TENTATIVE.value == "tentative"
        assert TrackState.CONFIRMED.value == "confirmed"
        assert TrackState.COASTING.value == "coasting"
        assert TrackState.DELETED.value == "deleted"


class TestTrackId:
    def test_uniqueness(self):
        ids = {generate_track_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_format(self):
        tid = generate_track_id()
        assert len(tid) == 8
        assert tid == tid.upper()
