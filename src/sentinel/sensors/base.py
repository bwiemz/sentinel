"""Abstract base class for all sensor adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from sentinel.sensors.frame import SensorFrame


class AbstractSensor(ABC):
    """Base interface for sensor adapters.

    All sensors (camera, radar, lidar, etc.) implement this interface
    to provide a uniform data ingestion API to the pipeline.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Initialize the sensor connection. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Release sensor resources."""
        ...

    @abstractmethod
    def read_frame(self) -> Optional[SensorFrame]:
        """Read a single frame. Returns None if no data available."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the sensor is currently connected and producing data."""
        ...

    def stream(self) -> Iterator[SensorFrame]:
        """Continuously yield frames from the sensor."""
        while self.is_connected:
            frame = self.read_frame()
            if frame is not None:
                yield frame

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
