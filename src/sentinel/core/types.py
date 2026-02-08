"""Core data types for the SENTINEL tracking system."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class SensorType(enum.Enum):
    CAMERA = "camera"
    RADAR = "radar"


class TrackState(enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class Detection:
    """Unified detection from any sensor modality."""

    sensor_type: SensorType
    timestamp: float  # epoch seconds

    # Camera fields
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2] pixels
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None

    # Radar fields
    range_m: Optional[float] = None
    azimuth_deg: Optional[float] = None
    velocity_mps: Optional[float] = None
    rcs_dbsm: Optional[float] = None

    # Shared / fused
    position_3d: Optional[np.ndarray] = None  # [x, y, z] world frame

    @property
    def bbox_center(self) -> Optional[np.ndarray]:
        """Return center [cx, cy] of bounding box."""
        if self.bbox is None:
            return None
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    @property
    def bbox_area(self) -> float:
        if self.bbox is None:
            return 0.0
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(max(w, 0) * max(h, 0))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sensor_type": self.sensor_type.value,
            "timestamp": self.timestamp,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox.tolist()
        if self.range_m is not None:
            d["range_m"] = self.range_m
            d["azimuth_deg"] = self.azimuth_deg
            d["velocity_mps"] = self.velocity_mps
        return d


def generate_track_id() -> str:
    return uuid.uuid4().hex[:8].upper()
