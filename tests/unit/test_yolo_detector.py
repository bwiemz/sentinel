"""Tests for YOLO detector.

These tests require ultralytics and will download a model on first run.
Tests are skipped if ultralytics is not installed.
"""

import numpy as np
import pytest

from sentinel.core.types import SensorType

try:
    from sentinel.detection.yolo import YOLODetector

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


@pytest.mark.skipif(not HAS_YOLO, reason="ultralytics not installed")
class TestYOLODetector:
    @pytest.fixture(scope="class")
    def detector(self):
        det = YOLODetector(model_path="yolov8n.pt", confidence=0.25, device="cpu")
        return det

    def test_detect_returns_list(self, detector, sample_frame):
        results = detector.detect(sample_frame, timestamp=1.0)
        assert isinstance(results, list)

    def test_detect_on_blank_frame(self, detector):
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect(blank, timestamp=0.0)
        assert isinstance(results, list)
        # Blank frame should produce zero or very few detections
        assert len(results) < 5

    def test_detection_fields(self, detector, sample_frame):
        results = detector.detect(sample_frame, timestamp=2.0)
        for det in results:
            assert det.sensor_type == SensorType.CAMERA
            assert det.timestamp == 2.0
            assert det.bbox is not None
            assert len(det.bbox) == 4
            assert det.confidence is not None
            assert 0 <= det.confidence <= 1
            assert det.class_name is not None

    def test_warmup(self, detector):
        detector.warmup()  # Should not raise

    def test_auto_device(self):
        device = YOLODetector._auto_device()
        assert device in ("cpu", "cuda:0", "mps")
