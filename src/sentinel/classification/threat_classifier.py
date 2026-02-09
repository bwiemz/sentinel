"""ML-based threat level classifier with rule-based fallback.

Uses scikit-learn GradientBoostingClassifier for 4-class threat level
prediction.  Falls back to rule-based classification when:
- No model is loaded
- Model prediction confidence is below threshold
- Any error occurs during ML inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel.classification.features import FEATURE_COUNT, FEATURE_NAMES, FeatureExtractor

logger = logging.getLogger(__name__)

THREAT_LABELS: list[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(THREAT_LABELS)}


@dataclass
class ClassificationResult:
    """Result of a threat classification."""

    predicted_level: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    method_used: str = "rule_based"


class ThreatClassifier:
    """ML threat classifier with automatic fallback to rule-based.

    Args:
        model_path: Path to saved model (joblib format).
        confidence_threshold: Minimum confidence to trust ML result.
        feature_extractor: Feature extraction instance.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.6,
        feature_extractor: FeatureExtractor | None = None,
    ):
        self._model = None
        self._model_loaded = False
        self._confidence_threshold = confidence_threshold
        self._feature_extractor = feature_extractor or FeatureExtractor()

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, path: str | Path) -> bool:
        """Load a pre-trained model from disk.

        Returns:
            True if model loaded successfully.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Threat model not found at %s", path)
            return False
        try:
            import joblib

            model = joblib.load(path)
            if not hasattr(model, "predict_proba"):
                logger.error("Model at %s lacks predict_proba method", path)
                return False
            self._model = model
            self._model_loaded = True
            logger.info("Threat classifier model loaded from %s", path)
            return True
        except Exception:
            logger.exception("Failed to load threat model from %s", path)
            return False

    def classify(
        self,
        eft,
        rule_based_level: str,
    ) -> ClassificationResult:
        """Classify a single fused track.

        Args:
            eft: The fused track to classify.
            rule_based_level: The rule-based classification (fallback).

        Returns:
            ClassificationResult with predicted level and confidence.
        """
        if not self._model_loaded:
            return self._make_rule_based_result(rule_based_level)

        try:
            features = self._feature_extractor.extract(eft)
            return self._predict_ml(features, rule_based_level)
        except Exception:
            logger.debug("ML classification failed, using rule-based", exc_info=True)
            return self._make_rule_based_result(rule_based_level)

    def classify_batch(
        self,
        tracks: list,
        rule_based_levels: list[str],
    ) -> list[ClassificationResult]:
        """Classify multiple tracks."""
        return [
            self.classify(eft, level)
            for eft, level in zip(tracks, rule_based_levels)
        ]

    def _predict_ml(
        self, features: np.ndarray, rule_based_level: str
    ) -> ClassificationResult:
        """Run ML model prediction on a feature vector."""
        X = features.reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        max_idx = int(np.argmax(proba))
        confidence = float(proba[max_idx])
        predicted = THREAT_LABELS[max_idx]

        probabilities = {
            label: float(p) for label, p in zip(THREAT_LABELS, proba)
        }

        if confidence < self._confidence_threshold:
            return ClassificationResult(
                predicted_level=rule_based_level,
                confidence=confidence,
                probabilities=probabilities,
                method_used="rule_based",
            )

        return ClassificationResult(
            predicted_level=predicted,
            confidence=confidence,
            probabilities=probabilities,
            method_used="ml",
        )

    @staticmethod
    def _make_rule_based_result(level: str) -> ClassificationResult:
        """Wrap a rule-based result in ClassificationResult."""
        probs = {label: 0.0 for label in THREAT_LABELS}
        if level in probs:
            probs[level] = 1.0
        return ClassificationResult(
            predicted_level=level,
            confidence=1.0,
            probabilities=probs,
            method_used="rule_based",
        )

    def get_global_feature_importances(self) -> dict[str, float]:
        """Return model's global feature importances."""
        if not self._model_loaded:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(FEATURE_NAMES, importances))

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded
