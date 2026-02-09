"""Synthetic training data generator for threat classifier.

Generates feature vectors with ground truth labels by simulating
various target types at different ranges, speeds, and sensor
configurations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel.classification.features import FEATURE_COUNT, FEATURE_NAMES
from sentinel.classification.threat_classifier import LABEL_TO_INDEX, THREAT_LABELS


@dataclass
class TrainingScenario:
    """Defines a category of synthetic targets for training."""

    ground_truth_threat: str
    speed_range: tuple[float, float]
    rcs_range: tuple[float, float]
    range_range: tuple[float, float] = (3000.0, 30000.0)
    temperature_range: tuple[float, float] | None = None
    qi_advantage_range: tuple[float, float] | None = None
    sensor_count_range: tuple[int, int] = (1, 4)
    is_stealth: bool = False
    is_chaff: bool = False
    is_decoy: bool = False
    has_quantum: bool = False
    num_radar_bands_range: tuple[int, int] = (0, 3)
    low_freq_only: bool = False


def default_scenarios() -> list[TrainingScenario]:
    """Default training scenarios covering all threat levels."""
    return [
        # CRITICAL: Hypersonic targets
        TrainingScenario(
            ground_truth_threat="CRITICAL",
            speed_range=(1500.0, 3000.0),
            rcs_range=(0.0, 15.0),
            temperature_range=(1500.0, 5000.0),
            sensor_count_range=(2, 4),
            num_radar_bands_range=(1, 3),
        ),
        # CRITICAL: Quantum-confirmed stealth
        TrainingScenario(
            ground_truth_threat="CRITICAL",
            speed_range=(200.0, 400.0),
            rcs_range=(-30.0, -10.0),
            temperature_range=(400.0, 800.0),
            is_stealth=True,
            has_quantum=True,
            sensor_count_range=(2, 4),
            num_radar_bands_range=(1, 2),
            low_freq_only=True,
        ),
        # HIGH: Stealth without quantum
        TrainingScenario(
            ground_truth_threat="HIGH",
            speed_range=(200.0, 400.0),
            rcs_range=(-30.0, -10.0),
            temperature_range=(400.0, 800.0),
            is_stealth=True,
            sensor_count_range=(1, 3),
            num_radar_bands_range=(1, 2),
            low_freq_only=True,
        ),
        # HIGH: Quantum-only (no classical radar)
        TrainingScenario(
            ground_truth_threat="HIGH",
            speed_range=(100.0, 350.0),
            rcs_range=(-20.0, 5.0),
            has_quantum=True,
            sensor_count_range=(1, 2),
            num_radar_bands_range=(0, 0),
        ),
        # MEDIUM: Conventional multi-sensor
        TrainingScenario(
            ground_truth_threat="MEDIUM",
            speed_range=(50.0, 300.0),
            rcs_range=(5.0, 25.0),
            temperature_range=(300.0, 600.0),
            sensor_count_range=(2, 4),
            num_radar_bands_range=(1, 3),
        ),
        # LOW: Single-sensor conventional
        TrainingScenario(
            ground_truth_threat="LOW",
            speed_range=(20.0, 150.0),
            rcs_range=(5.0, 30.0),
            sensor_count_range=(1, 1),
            num_radar_bands_range=(0, 2),
        ),
        # LOW: Chaff
        TrainingScenario(
            ground_truth_threat="LOW",
            speed_range=(5.0, 50.0),
            rcs_range=(25.0, 40.0),
            is_chaff=True,
            sensor_count_range=(1, 2),
            num_radar_bands_range=(1, 3),
        ),
        # LOW: Decoy
        TrainingScenario(
            ground_truth_threat="LOW",
            speed_range=(50.0, 200.0),
            rcs_range=(5.0, 20.0),
            is_decoy=True,
            sensor_count_range=(1, 1),
            num_radar_bands_range=(1, 2),
        ),
    ]


class TrainingDataGenerator:
    """Generates synthetic training data for the threat classifier.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    def generate(
        self,
        n_samples_per_class: int = 500,
        scenarios: list[TrainingScenario] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate training data.

        Returns:
            (X, y) where X is (N, 28) features and y is (N,) integer labels.
        """
        scenarios = scenarios or default_scenarios()
        rng = np.random.default_rng(self._seed)

        # Group scenarios by class to balance
        class_scenarios: dict[str, list[TrainingScenario]] = {}
        for sc in scenarios:
            class_scenarios.setdefault(sc.ground_truth_threat, []).append(sc)

        all_X = []
        all_y = []

        for label, sc_list in class_scenarios.items():
            label_idx = LABEL_TO_INDEX[label]
            n_per_scenario = max(1, n_samples_per_class // len(sc_list))

            for sc in sc_list:
                for _ in range(n_per_scenario):
                    features = self._generate_sample(sc, rng)
                    all_X.append(features)
                    all_y.append(label_idx)

        X = np.array(all_X)
        y = np.array(all_y, dtype=np.int64)

        # Shuffle
        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    def _generate_sample(
        self,
        scenario: TrainingScenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a single synthetic feature vector."""
        f = np.full(FEATURE_COUNT, np.nan)

        # Kinematic
        speed = rng.uniform(*scenario.speed_range)
        heading = rng.uniform(-np.pi, np.pi)
        range_m = rng.uniform(*scenario.range_range)

        f[0] = speed
        f[1] = speed / 343.0
        f[2] = heading
        f[3] = range_m

        # Approach rate: random fraction of speed
        approach_frac = rng.uniform(-0.8, 0.8)
        f[4] = speed * approach_frac
        f[5] = speed * np.sqrt(max(0, 1 - approach_frac**2))

        # Acceleration: occasional for high-speed targets
        if speed > 500 and rng.random() > 0.5:
            f[6] = rng.uniform(5.0, 50.0)
            f[7] = 1.0
        else:
            f[7] = 0.0

        # Signature
        f[8] = rng.uniform(*scenario.rcs_range)

        if scenario.is_stealth:
            f[9] = rng.uniform(15.0, 30.0)
        elif scenario.is_chaff:
            f[9] = rng.uniform(0.0, 5.0)
        elif rng.random() > 0.5:
            f[9] = rng.uniform(0.0, 10.0)

        if scenario.temperature_range is not None:
            f[10] = rng.uniform(*scenario.temperature_range)

        if scenario.qi_advantage_range is not None:
            f[11] = rng.uniform(*scenario.qi_advantage_range)
        elif scenario.has_quantum:
            f[11] = rng.uniform(3.0, 12.0)

        # Camera confidence (if multi-sensor, sometimes have camera)
        sc_min, sc_max = scenario.sensor_count_range
        sensor_count = int(rng.integers(sc_min, sc_max + 1))
        has_camera = sensor_count >= 2 and rng.random() > 0.3
        if has_camera:
            f[12] = rng.uniform(0.5, 0.95)

        nb_min, nb_max = scenario.num_radar_bands_range
        n_bands = int(rng.integers(nb_min, nb_max + 1))
        f[13] = float(n_bands)
        f[14] = 1.0 if scenario.low_freq_only else 0.0

        # Sensor coverage
        has_radar = n_bands > 0 or rng.random() > 0.3
        has_thermal = scenario.temperature_range is not None and rng.random() > 0.3
        has_quantum = scenario.has_quantum

        f[15] = float(sensor_count)
        f[16] = 1.0 if has_camera else 0.0
        f[17] = 1.0 if has_radar else 0.0
        f[18] = 1.0 if has_thermal else 0.0
        f[19] = 1.0 if has_quantum else 0.0
        f[20] = 1.0 if n_bands >= 2 else 0.0

        # EW flags
        f[21] = 1.0 if scenario.is_chaff else 0.0
        f[22] = 1.0 if scenario.is_decoy else 0.0
        f[23] = 1.0 if scenario.is_stealth else 0.0

        # Quality
        f[24] = rng.uniform(0.4, 0.95)
        f[25] = float(rng.integers(5, 50))
        f[26] = rng.uniform(0.5, 1.0)
        f[27] = rng.uniform(0.3, 0.9)

        return f

    def save_csv(self, X: np.ndarray, y: np.ndarray, path: str | Path) -> None:
        """Save training data to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(FEATURE_NAMES + ["label"])
            for i in range(len(X)):
                row = list(X[i]) + [int(y[i])]
                writer.writerow(row)

    def load_csv(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load training data from CSV file."""
        path = Path(path)
        with open(path, "r") as fp:
            reader = csv.reader(fp)
            header = next(reader)
            data = []
            labels = []
            for row in reader:
                data.append([float(x) if x != "" else np.nan for x in row[:-1]])
                labels.append(int(row[-1]))
        return np.array(data), np.array(labels, dtype=np.int64)
