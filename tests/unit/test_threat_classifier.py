"""Unit tests for ML threat classifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel.classification.features import FEATURE_COUNT, FEATURE_NAMES
from sentinel.classification.threat_classifier import (
    THREAT_LABELS,
    ClassificationResult,
    ThreatClassifier,
)


def _mock_eft_with_features():
    """Create a mock EFT that produces valid features."""
    eft = MagicMock()
    eft.radar_track = MagicMock()
    eft.radar_track.position = np.array([5000.0, 3000.0])
    eft.radar_track.velocity = np.array([-100.0, 50.0])
    eft.radar_track.score = 0.8
    eft.radar_track.age = 10
    eft.radar_track.quality_monitor = MagicMock()
    eft.radar_track.quality_monitor.consistency_score = 0.9
    eft.radar_track.last_detection = MagicMock()
    eft.radar_track.last_detection.rcs_dbsm = 10.0
    eft.radar_track.state_vector = None
    eft.radar_track.ekf = MagicMock()
    eft.radar_track.ekf.x = np.array([5000, -100, 3000, 50])  # 4D state
    eft.thermal_track = None
    eft.quantum_radar_track = None
    eft.camera_track = None
    eft.correlated_detection = None
    eft.position_m = np.array([5000.0, 3000.0])
    eft.range_m = 5831.0
    eft.temperature_k = None
    eft.qi_advantage_db = None
    eft.confidence = None
    eft.radar_bands_detected = ["s_band", "x_band"]
    eft.is_chaff_candidate = False
    eft.is_decoy_candidate = False
    eft.is_stealth_candidate = False
    eft.fusion_quality = 0.6
    eft.sensor_count = 1
    return eft


def _mock_sklearn_model(predicted_class=2, n_classes=4):
    """Create a mock sklearn model with predict_proba."""
    model = MagicMock()
    proba = np.zeros(n_classes)
    proba[predicted_class] = 0.85
    remaining = 0.15 / max(1, n_classes - 1)
    for i in range(n_classes):
        if i != predicted_class:
            proba[i] = remaining
    model.predict_proba.return_value = proba.reshape(1, -1)
    model.feature_importances_ = np.random.rand(FEATURE_COUNT)
    model.feature_importances_ /= model.feature_importances_.sum()
    return model


class TestClassificationResult:
    def test_structure(self):
        result = ClassificationResult(
            predicted_level="HIGH",
            confidence=0.85,
            probabilities={"LOW": 0.05, "MEDIUM": 0.05, "HIGH": 0.85, "CRITICAL": 0.05},
            method_used="ml",
        )
        assert result.predicted_level == "HIGH"
        assert result.confidence == 0.85
        assert result.method_used == "ml"
        assert sum(result.probabilities.values()) == pytest.approx(1.0)


class TestThreatClassifierNoModel:
    def test_no_model_returns_rule_based(self):
        clf = ThreatClassifier()
        eft = _mock_eft_with_features()
        result = clf.classify(eft, "MEDIUM")
        assert result.predicted_level == "MEDIUM"
        assert result.method_used == "rule_based"
        assert result.confidence == 1.0

    def test_load_nonexistent_model(self):
        clf = ThreatClassifier()
        ok = clf.load_model("/nonexistent/path/model.joblib")
        assert ok is False
        assert clf.is_model_loaded is False

    def test_is_model_loaded_false(self):
        clf = ThreatClassifier()
        assert clf.is_model_loaded is False

    def test_rule_based_probabilities(self):
        clf = ThreatClassifier()
        result = clf.classify(_mock_eft_with_features(), "CRITICAL")
        assert result.probabilities["CRITICAL"] == 1.0
        assert result.probabilities["LOW"] == 0.0

    def test_empty_feature_importances(self):
        clf = ThreatClassifier()
        assert clf.get_global_feature_importances() == {}


class TestThreatClassifierWithModel:
    def _make_classifier_with_mock(self, predicted_class=2, confidence=0.85):
        clf = ThreatClassifier(confidence_threshold=0.6)
        clf._model = _mock_sklearn_model(predicted_class)
        clf._model_loaded = True
        return clf

    def test_classify_with_mock_model(self):
        clf = self._make_classifier_with_mock(predicted_class=2)  # HIGH
        result = clf.classify(_mock_eft_with_features(), "MEDIUM")
        assert result.predicted_level == "HIGH"
        assert result.method_used == "ml"

    def test_probabilities_sum_to_one(self):
        clf = self._make_classifier_with_mock()
        result = clf.classify(_mock_eft_with_features(), "MEDIUM")
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_confidence_matches_max_prob(self):
        clf = self._make_classifier_with_mock()
        result = clf.classify(_mock_eft_with_features(), "MEDIUM")
        max_prob = max(result.probabilities.values())
        assert result.confidence == pytest.approx(max_prob)

    def test_fallback_on_low_confidence(self):
        """ML confidence < threshold → use rule-based level."""
        clf = ThreatClassifier(confidence_threshold=0.9)
        model = _mock_sklearn_model(predicted_class=3)  # CRITICAL at 0.85
        clf._model = model
        clf._model_loaded = True
        result = clf.classify(_mock_eft_with_features(), "MEDIUM")
        # 0.85 < 0.9 threshold → fallback to rule_based
        assert result.predicted_level == "MEDIUM"
        assert result.method_used == "rule_based"

    def test_method_used_ml(self):
        clf = self._make_classifier_with_mock()
        result = clf.classify(_mock_eft_with_features(), "LOW")
        assert result.method_used == "ml"

    def test_feature_importances_available(self):
        clf = self._make_classifier_with_mock()
        imps = clf.get_global_feature_importances()
        assert len(imps) == FEATURE_COUNT
        assert set(imps.keys()) == set(FEATURE_NAMES)

    def test_batch_classify(self):
        clf = self._make_classifier_with_mock()
        efts = [_mock_eft_with_features(), _mock_eft_with_features()]
        results = clf.classify_batch(efts, ["LOW", "MEDIUM"])
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_backward_compatible_string_levels(self):
        """Output levels match string constants used in existing code."""
        clf = self._make_classifier_with_mock()
        result = clf.classify(_mock_eft_with_features(), "LOW")
        assert result.predicted_level in THREAT_LABELS
