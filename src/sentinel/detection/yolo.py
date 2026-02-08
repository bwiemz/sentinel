"""YOLOv8/v9/v11 object detector using the Ultralytics library."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from sentinel.core.types import Detection, SensorType
from sentinel.detection.base import AbstractDetector

logger = logging.getLogger(__name__)


class YOLODetector(AbstractDetector):
    """Object detector wrapping Ultralytics YOLO models.

    Supports YOLOv8, YOLOv9, YOLOv11, and any Ultralytics-compatible model.

    Args:
        model_path: Path to YOLO weights file (.pt) or model name.
        confidence: Minimum detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        device: Inference device -- "auto", "cpu", "cuda:0", "mps".
        classes: Optional list of class IDs to filter. None = all classes.
        max_detections: Maximum detections per frame.
        image_size: Input image size for the model.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        classes: Optional[list[int]] = None,
        max_detections: int = 100,
        image_size: int = 640,
    ):
        from ultralytics import YOLO

        self._model_path = model_path
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._classes = classes
        self._max_detections = max_detections
        self._image_size = image_size

        # Resolve device
        if device == "auto":
            self._device = self._auto_device()
        else:
            self._device = device

        logger.info("Loading YOLO model: %s on %s", model_path, self._device)
        self._model = YOLO(model_path)
        self._class_names: dict[int, str] = {}

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Run YOLO inference on a BGR frame."""
        results = self._model(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            device=self._device,
            classes=self._classes,
            max_det=self._max_detections,
            imgsz=self._image_size,
            verbose=False,
        )

        detections: list[Detection] = []
        result = results[0]

        # Cache class names from first result
        if not self._class_names and hasattr(result, "names"):
            self._class_names = result.names

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append(Detection(
                sensor_type=SensorType.CAMERA,
                timestamp=timestamp,
                bbox=box.xyxy[0].cpu().numpy().astype(np.float32),
                class_id=cls_id,
                class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                confidence=float(box.conf[0].item()),
            ))

        return detections

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.info("YOLO warmup complete on %s", self._device)

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    @staticmethod
    def _auto_device() -> str:
        """Detect best available inference device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
