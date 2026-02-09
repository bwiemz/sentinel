"""System clock, simulated clock, and timestamping utilities."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for clocks used by simulators and the pipeline.

    Both SystemClock (real-time) and SimClock (deterministic) implement this.
    """

    def now(self) -> float:
        """Current time as epoch seconds."""
        ...

    def elapsed(self) -> float:
        """Seconds since clock was created/started."""
        ...


class SystemClock:
    """Monotonic clock for consistent timestamping across the pipeline.

    Uses real wall-clock time.  This is the default clock.
    """

    def __init__(self):
        self._start_time = time.monotonic()
        self._epoch_offset = time.time() - self._start_time

    def now(self) -> float:
        """Current time as epoch seconds (monotonic-based)."""
        return time.monotonic() + self._epoch_offset

    def elapsed(self) -> float:
        """Seconds since clock was created."""
        return time.monotonic() - self._start_time


class SimClock:
    """Deterministic clock for reproducible simulation and testing.

    Time only advances when :meth:`step`, :meth:`set_time`, or
    :meth:`set_elapsed` are called.  Provides the same ``now()`` /
    ``elapsed()`` interface as :class:`SystemClock` via the :class:`Clock`
    protocol.

    Args:
        start_epoch: Initial epoch time (what ``now()`` returns at
            ``elapsed=0``).  Defaults to ``1_000_000.0``.
    """

    def __init__(self, start_epoch: float = 1_000_000.0):
        self._start_epoch = start_epoch
        self._elapsed = 0.0

    def now(self) -> float:
        """Current simulated epoch time."""
        return self._start_epoch + self._elapsed

    def elapsed(self) -> float:
        """Simulated seconds since clock was created."""
        return self._elapsed

    def step(self, dt: float) -> None:
        """Advance simulated time by *dt* seconds.

        Raises:
            ValueError: If *dt* is negative.
        """
        if dt < 0:
            raise ValueError(f"SimClock.step() requires dt >= 0, got {dt}")
        self._elapsed += dt

    def set_time(self, epoch_time: float) -> None:
        """Set the absolute simulated epoch time.

        Raises:
            ValueError: If *epoch_time* is before *start_epoch*.
        """
        new_elapsed = epoch_time - self._start_epoch
        if new_elapsed < 0:
            raise ValueError(
                f"epoch_time {epoch_time} is before start_epoch {self._start_epoch}"
            )
        self._elapsed = new_elapsed

    def set_elapsed(self, elapsed: float) -> None:
        """Set the elapsed time directly.

        Raises:
            ValueError: If *elapsed* is negative.
        """
        if elapsed < 0:
            raise ValueError(
                f"SimClock.set_elapsed() requires elapsed >= 0, got {elapsed}"
            )
        self._elapsed = elapsed

    @property
    def start_epoch(self) -> float:
        """The epoch time that corresponds to ``elapsed=0``."""
        return self._start_epoch


def create_clock(config: dict | None = None) -> SystemClock | SimClock:
    """Create a clock from the ``sentinel.time`` config section.

    Returns a :class:`SystemClock` for real-time mode (default) or a
    :class:`SimClock` for simulated mode.
    """
    if config is None:
        return SystemClock()
    mode = config.get("mode", "realtime")
    if mode == "simulated":
        return SimClock(start_epoch=config.get("start_epoch", 1_000_000.0))
    return SystemClock()


class FrameTimer:
    """Tracks frame timing and computes FPS.

    Always uses real wall-clock time (``time.monotonic``).
    """

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
