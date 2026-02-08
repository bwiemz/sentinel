"""Abstract base class for object detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sentinel.core.types import Detection


class AbstractDetector(ABC):
    """Base interface for all detection algorithms."""

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            timestamp: Frame timestamp in epoch seconds.

        Returns:
            List of Detection objects found in the frame.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Run a warmup inference to initialize the model."""
        ...
