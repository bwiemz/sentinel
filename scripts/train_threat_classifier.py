"""Train the threat classification model.

Usage:
    python scripts/train_threat_classifier.py [--samples 2000] [--output models/threat_classifier.joblib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SENTINEL threat classifier")
    parser.add_argument(
        "--samples", type=int, default=2000, help="Samples per class (default: 2000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/threat_classifier.joblib",
        help="Output model path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import cross_val_score, train_test_split
    except ImportError:
        print("ERROR: scikit-learn and joblib required. Install with:")
        print("  pip install scikit-learn joblib")
        sys.exit(1)

    from sentinel.classification.features import FEATURE_NAMES
    from sentinel.classification.threat_classifier import THREAT_LABELS
    from sentinel.classification.training_data import TrainingDataGenerator

    # Generate training data
    print("Generating synthetic training data...")
    gen = TrainingDataGenerator(seed=args.seed)
    X, y = gen.generate(n_samples_per_class=args.samples)
    print(f"  Generated {X.shape[0]} samples with {X.shape[1]} features")
    for i, label in enumerate(THREAT_LABELS):
        count = int(np.sum(y == i))
        print(f"  {label}: {count} samples")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train
    print("\nTraining GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=THREAT_LABELS,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Feature importances
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Feature Importances:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
