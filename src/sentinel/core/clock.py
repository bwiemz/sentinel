"""System clock and timestamping utilities."""

from __future__ import annotations

import time


class SystemClock:
    """Monotonic clock for consistent timestamping across the pipeline."""

    def __init__(self):
        self._start_time = time.monotonic()
        self._epoch_offset = time.time() - self._start_time

    def now(self) -> float:
        """Current time as epoch seconds (monotonic-based)."""
        return time.monotonic() + self._epoch_offset

    def elapsed(self) -> float:
        """Seconds since clock was created."""
        return time.monotonic() - self._start_time


class FrameTimer:
    """Tracks frame timing and computes FPS."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record a frame timestamp."""
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        """Rolling average FPS over the window."""
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt
