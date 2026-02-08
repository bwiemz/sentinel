"""Sensor frame data container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sentinel.core.types import SensorType


@dataclass
class SensorFrame:
    """A single frame of data from any sensor."""

    data: Any  # np.ndarray for camera, list[dict] for radar
    timestamp: float
    sensor_type: SensorType
    frame_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> tuple[int, int] | None:
        """(height, width) for camera frames."""
        if self.sensor_type == SensorType.CAMERA and isinstance(self.data, np.ndarray):
            return self.data.shape[:2]
        return None
