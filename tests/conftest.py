"""Shared pytest fixtures for SENTINEL tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, SensorType


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def config_path(project_root: Path) -> Path:
    return project_root / "config" / "default.yaml"


@pytest.fixture
def default_config(config_path: Path):
    return OmegaConf.load(config_path)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 720p synthetic test frame with some colored shapes."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add some rectangles to simulate objects
    import cv2

    cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 255), -1)
    cv2.rectangle(frame, (600, 300), (800, 500), (255, 0, 0), -1)
    cv2.circle(frame, (1000, 400), 80, (0, 255, 0), -1)
    return frame


@pytest.fixture
def sample_detections() -> list[Detection]:
    """A set of test detections."""
    return [
        Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=1000.0,
            bbox=np.array([200, 200, 400, 400], dtype=np.float32),
            class_id=0,
            class_name="person",
            confidence=0.92,
        ),
        Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=1000.0,
            bbox=np.array([600, 300, 800, 500], dtype=np.float32),
            class_id=2,
            class_name="car",
            confidence=0.85,
        ),
    ]
