"""Integration tests for ML threat classification pipeline.

Tests the full flow: feature extraction → classification → integration
with MultiSensorFusion, verifying backward compatibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.classification.features import FEATURE_COUNT, FeatureExtractor
from sentinel.classification.intent_estimator import IntentEstimator
from sentinel.classification.threat_classifier import ThreatClassifier
from sentinel.core.types import IntentType, ThreatLevel
from sentinel.fusion.multi_sensor_fusion import (
    EnhancedFusedTrack,
    MultiSensorFusion,
    THREAT_CRITICAL,
    THREAT_HIGH,
    THREAT_LOW,
    THREAT_MEDIUM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_radar_track(position, velocity, rcs_dbsm=10.0, score=0.7, age=10):
    """Create a minimal mock radar track."""
    from sentinel.tracking.radar_track import RadarTrack
    from sentinel.core.types import Detection, SensorType

    det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        range_m=float(np.linalg.norm(position)),
        azimuth_deg=float(np.degrees(np.arctan2(position[1], position[0]))),
        rcs_dbsm=rcs_dbsm,
    )

    config = OmegaConf.create({
        "filter": {"type": "ekf"},
        "association": {"method": "hungarian", "gate": 50.0},
        "filter_params": {"range_std_m": 50.0, "azimuth_std_deg": 1.0},
    })
    from sentinel.tracking.radar_track_manager import RadarTrackManager
    mgr = RadarTrackManager(config)
    mgr.predict(0.1)
    mgr.update([det])
    # Force track to have specific properties via mock overlay
    if mgr.active_tracks:
        track = mgr.active_tracks[0]
        return track

    # Fallback: create a manual mock
    t = MagicMock()
    t.position = np.array(position)
    t.velocity = np.array(velocity)
    t.score = score
    t.age = age
    t.azimuth_deg = float(np.degrees(np.arctan2(position[1], position[0])))
    t.range_m = float(np.linalg.norm(position))
    t.last_detection = det
    t.quality_monitor = None
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuleBasedDefault:
    """Verify rule-based path (default) still works with new fields."""

    def test_rule_based_default_produces_threat_method(self):
        """Default fusion uses rule-based method."""
        fusion = MultiSensorFusion()
        # Rule-based is the default
        assert fusion._threat_classifier is None

    def test_enhanced_fused_track_new_fields_default(self):
        """New fields have correct defaults."""
        eft = EnhancedFusedTrack(fused_id="test")
        assert eft.threat_confidence == 0.0
        assert eft.threat_probabilities == {}
        assert eft.threat_method == "rule_based"
        assert eft.intent == "unknown"
        assert eft.intent_confidence == 0.0

    def test_to_dict_includes_new_fields(self):
        """to_dict() includes new Phase 18 fields."""
        eft = EnhancedFusedTrack(fused_id="test")
        d = eft.to_dict()
        assert "threat_confidence" in d
        assert "threat_method" in d
        assert "intent" in d


class TestMLClassifierIntegration:
    """Test ML classifier integration with mock model."""

    def test_ml_mode_with_mock_model(self):
        """ML method produces threat_method='ml' when model is loaded."""
        fusion = MultiSensorFusion(threat_classification_method="ml")
        # Manually inject a mock model
        mock_model = MagicMock()
        proba = np.array([[0.05, 0.1, 0.8, 0.05]])
        mock_model.predict_proba.return_value = proba
        mock_model.feature_importances_ = np.ones(FEATURE_COUNT) / FEATURE_COUNT
        fusion._threat_classifier._model = mock_model
        fusion._threat_classifier._model_loaded = True

        # Create a simple EFT
        eft = EnhancedFusedTrack(fused_id="test")
        rule_level = "MEDIUM"

        result = fusion._threat_classifier.classify(eft, rule_level)
        assert result.predicted_level == "HIGH"
        assert result.method_used == "ml"
        assert result.confidence == pytest.approx(0.8)

    def test_ml_mode_no_model_falls_back(self):
        """ML mode without model file gracefully falls back."""
        fusion = MultiSensorFusion(
            threat_classification_method="ml",
            threat_model_path="/nonexistent/model.joblib",
        )
        # Classifier created but no model loaded
        assert fusion._threat_classifier is not None
        assert fusion._threat_classifier.is_model_loaded is False


class TestIntentEstimation:
    """Test intent estimator integration."""

    def test_intent_estimation_enabled(self):
        """Intent estimator is created when enabled."""
        fusion = MultiSensorFusion(intent_estimation_enabled=True)
        assert fusion._intent_estimator is not None

    def test_intent_estimation_disabled_default(self):
        """Intent estimator is not created by default."""
        fusion = MultiSensorFusion()
        assert fusion._intent_estimator is None


class TestEnumsExist:
    """Verify new enums are available."""

    def test_threat_level_enum(self):
        assert ThreatLevel.LOW.value == "LOW"
        assert ThreatLevel.CRITICAL.value == "CRITICAL"
        assert len(ThreatLevel) == 4

    def test_intent_type_enum(self):
        assert IntentType.ATTACK.value == "attack"
        assert IntentType.UNKNOWN.value == "unknown"
        assert len(IntentType) == 6


class TestFeatureExtractionEndToEnd:
    """Test feature extraction on real-ish EnhancedFusedTrack."""

    def test_extract_from_eft_with_real_fields(self):
        """Feature extraction works on a real EnhancedFusedTrack dataclass."""
        eft = EnhancedFusedTrack(
            fused_id="test",
            range_m=8000.0,
            temperature_k=450.0,
            qi_advantage_db=6.0,
            is_stealth_candidate=True,
            fusion_quality=0.7,
            radar_bands_detected=["vhf", "uhf"],
        )
        ext = FeatureExtractor()
        features = ext.extract(eft)
        assert features.shape == (FEATURE_COUNT,)
        assert features[3] == pytest.approx(8000.0)  # range_m
        assert features[10] == pytest.approx(450.0)  # temperature_k
        assert features[23] == pytest.approx(1.0)  # is_stealth_candidate
