"""Threat classification and intent estimation for SENTINEL.

Provides ML-based threat level classification (gradient boosted trees)
with automatic fallback to rule-based when model is unavailable or
confidence is low, plus kinematic-based intent estimation.
"""

from sentinel.classification.features import FEATURE_COUNT, FEATURE_NAMES, FeatureExtractor
from sentinel.classification.intent_estimator import IntentEstimate, IntentEstimator
from sentinel.classification.threat_classifier import ClassificationResult, ThreatClassifier

__all__ = [
    "ClassificationResult",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "IntentEstimate",
    "IntentEstimator",
    "ThreatClassifier",
]
