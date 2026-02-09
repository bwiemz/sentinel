"""Unit tests for training data generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sentinel.classification.features import FEATURE_COUNT
from sentinel.classification.threat_classifier import LABEL_TO_INDEX, THREAT_LABELS
from sentinel.classification.training_data import (
    TrainingDataGenerator,
    TrainingScenario,
    default_scenarios,
)


class TestTrainingDataGenerator:
    def test_generate_shape(self):
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate(n_samples_per_class=50)
        assert X.ndim == 2
        assert X.shape[1] == FEATURE_COUNT
        assert len(y) == len(X)

    def test_all_classes_present(self):
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate(n_samples_per_class=50)
        unique_labels = set(y.tolist())
        expected = set(LABEL_TO_INDEX.values())
        assert unique_labels == expected

    def test_balanced_classes(self):
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate(n_samples_per_class=100)
        for label_idx in LABEL_TO_INDEX.values():
            count = int(np.sum(y == label_idx))
            # Allow some imbalance due to rounding across scenarios
            assert count >= 50, f"Class {label_idx} has only {count} samples"

    def test_features_no_inf(self):
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate(n_samples_per_class=100)
        assert not np.any(np.isinf(X))

    def test_csv_roundtrip(self):
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate(n_samples_per_class=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            gen.save_csv(X, y, path)
            X2, y2 = gen.load_csv(path)
            np.testing.assert_array_almost_equal(X, X2, decimal=10)
            np.testing.assert_array_equal(y, y2)

    def test_default_scenarios_cover_all_levels(self):
        scenarios = default_scenarios()
        levels = {s.ground_truth_threat for s in scenarios}
        assert levels == set(THREAT_LABELS)

    def test_reproducible_with_seed(self):
        gen1 = TrainingDataGenerator(seed=123)
        gen2 = TrainingDataGenerator(seed=123)
        X1, y1 = gen1.generate(n_samples_per_class=10)
        X2, y2 = gen2.generate(n_samples_per_class=10)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
