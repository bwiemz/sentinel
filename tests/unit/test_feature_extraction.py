"""Unit tests for feature extraction from EnhancedFusedTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.classification.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureExtractor,
)


# ---------------------------------------------------------------------------
# Lightweight mock objects (avoid importing full track classes)
# ---------------------------------------------------------------------------

def _mock_track(
    position=None,
    velocity=None,
    score=0.7,
    age=10,
    quality_monitor=None,
    last_detection=None,
    ekf=None,
):
    t = MagicMock()
    t.position = np.array(position) if position is not None else None
    t.velocity = np.array(velocity) if velocity is not None else None
    t.score = score
    t.age = age
    t.quality_monitor = quality_monitor
    t.last_detection = last_detection
    t.state_vector = None
    t.ekf = ekf
    return t


def _mock_detection(rcs_dbsm=None, azimuth_deg=None):
    d = MagicMock()
    d.rcs_dbsm = rcs_dbsm
    d.azimuth_deg = azimuth_deg
    return d


def _mock_quality_monitor(consistency_score=0.85):
    m = MagicMock()
    m.consistency_score = consistency_score
    return m


def _mock_correlated_detection(band_rcs: dict[str, float] | None = None):
    cd = MagicMock()
    cd.band_detections = {}
    if band_rcs:
        for band, rcs in band_rcs.items():
            det = MagicMock()
            det.rcs_dbsm = rcs
            cd.band_detections[band] = det
    return cd


def _mock_eft(
    camera_track=None,
    radar_track=None,
    thermal_track=None,
    quantum_radar_track=None,
    correlated_detection=None,
    range_m=None,
    temperature_k=None,
    qi_advantage_db=None,
    confidence=None,
    radar_bands_detected=None,
    is_chaff_candidate=False,
    is_decoy_candidate=False,
    is_stealth_candidate=False,
    has_quantum_confirmation=False,
    is_hypersonic_candidate=False,
    fusion_quality=0.5,
    position_m=None,
):
    eft = MagicMock()
    eft.camera_track = camera_track
    eft.radar_track = radar_track
    eft.thermal_track = thermal_track
    eft.quantum_radar_track = quantum_radar_track
    eft.correlated_detection = correlated_detection
    eft.range_m = range_m
    eft.temperature_k = temperature_k
    eft.qi_advantage_db = qi_advantage_db
    eft.confidence = confidence
    eft.radar_bands_detected = radar_bands_detected or []
    eft.is_chaff_candidate = is_chaff_candidate
    eft.is_decoy_candidate = is_decoy_candidate
    eft.is_stealth_candidate = is_stealth_candidate
    eft.has_quantum_confirmation = has_quantum_confirmation
    eft.is_hypersonic_candidate = is_hypersonic_candidate
    eft.fusion_quality = fusion_quality
    eft.position_m = np.array(position_m) if position_m is not None else None

    count = sum(
        1
        for t in [camera_track, radar_track, thermal_track, quantum_radar_track]
        if t is not None
    )
    eft.sensor_count = count
    return eft


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_feature_count_matches_names(self):
        assert len(FEATURE_NAMES) == FEATURE_COUNT

    def test_feature_count_is_28(self):
        assert FEATURE_COUNT == 28

    def test_no_duplicate_names(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


class TestFeatureExtractorOutput:
    def test_output_shape(self):
        ext = FeatureExtractor()
        eft = _mock_eft(radar_track=_mock_track(position=[5000, 3000], velocity=[100, 50]))
        features = ext.extract(eft)
        assert features.shape == (FEATURE_COUNT,)

    def test_output_dtype_float(self):
        ext = FeatureExtractor()
        eft = _mock_eft(radar_track=_mock_track(position=[5000, 3000], velocity=[100, 50]))
        features = ext.extract(eft)
        assert features.dtype == np.float64


class TestKinematicFeatures:
    def test_speed_computation(self):
        ext = FeatureExtractor()
        vel = [300, 400]
        eft = _mock_eft(radar_track=_mock_track(position=[5000, 0], velocity=vel))
        f = ext.extract(eft)
        assert f[0] == pytest.approx(500.0)

    def test_mach_computation(self):
        ext = FeatureExtractor(speed_of_sound_mps=343.0)
        eft = _mock_eft(radar_track=_mock_track(position=[5000, 0], velocity=[343, 0]))
        f = ext.extract(eft)
        assert f[1] == pytest.approx(1.0)

    def test_heading_computation(self):
        ext = FeatureExtractor()
        eft = _mock_eft(radar_track=_mock_track(position=[5000, 0], velocity=[0, 100]))
        f = ext.extract(eft)
        assert f[2] == pytest.approx(np.pi / 2)

    def test_range_from_eft(self):
        ext = FeatureExtractor()
        eft = _mock_eft(range_m=8000.0)
        f = ext.extract(eft)
        assert f[3] == pytest.approx(8000.0)

    def test_range_from_position(self):
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[3000, 4000], velocity=[10, 0]),
            position_m=[3000, 4000],
        )
        f = ext.extract(eft)
        assert f[3] == pytest.approx(5000.0)

    def test_approach_rate_closing(self):
        """Target at [5000, 0] moving [-100, 0] → approach_rate ≈ +100."""
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[-100, 0]),
            position_m=[5000, 0],
        )
        f = ext.extract(eft)
        assert f[4] == pytest.approx(100.0, abs=1.0)

    def test_approach_rate_receding(self):
        """Target at [5000, 0] moving [100, 0] → approach_rate ≈ -100."""
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[100, 0]),
            position_m=[5000, 0],
        )
        f = ext.extract(eft)
        assert f[4] == pytest.approx(-100.0, abs=1.0)

    def test_zero_velocity(self):
        """Zero velocity doesn't cause division errors."""
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[0, 0]),
            position_m=[5000, 0],
        )
        f = ext.extract(eft)
        assert f[0] == pytest.approx(0.0)
        assert np.isfinite(f[4])


class TestSignatureFeatures:
    def test_rcs_from_detection(self):
        det = _mock_detection(rcs_dbsm=10.0)
        radar = _mock_track(position=[5000, 0], velocity=[10, 0], last_detection=det)
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(radar_track=radar))
        assert f[8] == pytest.approx(10.0)

    def test_rcs_variance_from_correlated(self):
        cd = _mock_correlated_detection({"vhf": 5.0, "x_band": 20.0})
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(correlated_detection=cd))
        assert f[9] == pytest.approx(15.0)

    def test_temperature(self):
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(temperature_k=2000.0))
        assert f[10] == pytest.approx(2000.0)

    def test_qi_advantage(self):
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(qi_advantage_db=6.0))
        assert f[11] == pytest.approx(6.0)

    def test_camera_confidence(self):
        cam = _mock_track(score=0.9)
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(camera_track=cam, confidence=0.85))
        assert f[12] == pytest.approx(0.85)

    def test_radar_bands_count(self):
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(radar_bands_detected=["vhf", "s_band", "x_band"]))
        assert f[13] == pytest.approx(3.0)

    def test_low_freq_only_flag(self):
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(radar_bands_detected=["vhf", "uhf"]))
        assert f[14] == pytest.approx(1.0)

    def test_not_low_freq_only(self):
        ext = FeatureExtractor()
        f = ext.extract(_mock_eft(radar_bands_detected=["vhf", "x_band"]))
        assert f[14] == pytest.approx(0.0)


class TestSensorFeatures:
    def test_sensor_count(self):
        ext = FeatureExtractor()
        eft = _mock_eft(
            camera_track=_mock_track(),
            radar_track=_mock_track(position=[5000, 0], velocity=[10, 0]),
            thermal_track=_mock_track(position=[5000, 0], velocity=[10, 0]),
        )
        f = ext.extract(eft)
        assert f[15] == pytest.approx(3.0)
        assert f[16] == pytest.approx(1.0)  # has_camera
        assert f[17] == pytest.approx(1.0)  # has_radar
        assert f[18] == pytest.approx(1.0)  # has_thermal
        assert f[19] == pytest.approx(0.0)  # no quantum
        assert f[20] == pytest.approx(0.0)  # no correlated

    def test_ew_flags(self):
        ext = FeatureExtractor()
        eft = _mock_eft(is_chaff_candidate=True, is_stealth_candidate=True)
        f = ext.extract(eft)
        assert f[21] == pytest.approx(1.0)
        assert f[22] == pytest.approx(0.0)
        assert f[23] == pytest.approx(1.0)


class TestQualityFeatures:
    def test_track_score(self):
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[10, 0], score=0.9),
            thermal_track=_mock_track(position=[5000, 0], velocity=[10, 0], score=0.6),
        )
        f = ext.extract(eft)
        assert f[24] == pytest.approx(0.9)

    def test_track_age(self):
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[10, 0], age=20),
        )
        f = ext.extract(eft)
        assert f[25] == pytest.approx(20.0)

    def test_filter_health(self):
        qm = _mock_quality_monitor(0.92)
        ext = FeatureExtractor()
        eft = _mock_eft(
            radar_track=_mock_track(
                position=[5000, 0], velocity=[10, 0], quality_monitor=qm
            ),
        )
        f = ext.extract(eft)
        assert f[26] == pytest.approx(0.92)

    def test_fusion_quality(self):
        ext = FeatureExtractor()
        eft = _mock_eft(fusion_quality=0.75)
        f = ext.extract(eft)
        assert f[27] == pytest.approx(0.75)


class TestMissingData:
    def test_no_sensors_at_all(self):
        """Track with no sensor data produces valid vector with defaults/NaN."""
        ext = FeatureExtractor()
        eft = _mock_eft()
        f = ext.extract(eft)
        assert f.shape == (FEATURE_COUNT,)
        # Speed should be 0 (no velocity)
        assert f[0] == pytest.approx(0.0)
        # No inf values
        assert not np.any(np.isinf(f))

    def test_camera_only_track(self):
        """Camera-only: kinematic speed = 0 (camera velocity is pixels/frame)."""
        ext = FeatureExtractor()
        eft = _mock_eft(camera_track=_mock_track(score=0.8), confidence=0.9)
        f = ext.extract(eft)
        assert f[16] == pytest.approx(1.0)  # has_camera
        assert f[17] == pytest.approx(0.0)  # no radar


class TestBatchExtraction:
    def test_batch_shape(self):
        ext = FeatureExtractor()
        efts = [
            _mock_eft(radar_track=_mock_track(position=[5000, 0], velocity=[100, 0])),
            _mock_eft(radar_track=_mock_track(position=[3000, 4000], velocity=[50, 50])),
        ]
        batch = ext.extract_batch(efts)
        assert batch.shape == (2, FEATURE_COUNT)

    def test_empty_batch(self):
        ext = FeatureExtractor()
        batch = ext.extract_batch([])
        assert batch.shape == (0, FEATURE_COUNT)
