"""HistoryBuffer — thread-safe ring buffer for HistoryFrame storage."""

from __future__ import annotations

import bisect
import threading

from sentinel.history.frame import HistoryFrame


class HistoryBuffer:
    """Thread-safe ring buffer for HistoryFrame storage.

    Uses a list with eviction of the oldest entry when capacity is reached.
    A parallel ``_timestamps`` list enables O(log n) time-based lookup via
    :func:`bisect.bisect_left`.

    Thread safety: all public methods acquire ``self._lock``.  Since
    :class:`HistoryFrame` is frozen, returned frames are safe to use
    without the lock.
    """

    def __init__(self, max_frames: int = 10000) -> None:
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        self._max_frames = max_frames
        self._frames: list[HistoryFrame] = []
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    @property
    def max_frames(self) -> int:
        return self._max_frames

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def time_range(self) -> tuple[float, float] | None:
        """Return ``(earliest_timestamp, latest_timestamp)`` or *None* if empty."""
        with self._lock:
            if not self._frames:
                return None
            return (self._frames[0].timestamp, self._frames[-1].timestamp)

    def record(self, frame: HistoryFrame) -> None:
        """Append a frame.  Evicts the oldest if at capacity."""
        with self._lock:
            if len(self._frames) >= self._max_frames:
                self._frames.pop(0)
                self._timestamps.pop(0)
            self._frames.append(frame)
            self._timestamps.append(frame.timestamp)

    def get_frame(self, index: int) -> HistoryFrame | None:
        """Get frame by buffer index (0 = oldest).  Negative indexing supported."""
        with self._lock:
            if not self._frames:
                return None
            if -len(self._frames) <= index < len(self._frames):
                return self._frames[index]
            return None

    def get_range(self, start: int, end: int) -> list[HistoryFrame]:
        """Get frames from *start* to *end* (exclusive).  Clamps to bounds."""
        with self._lock:
            n = len(self._frames)
            start = max(0, min(start, n))
            end = max(0, min(end, n))
            return list(self._frames[start:end])

    def get_frame_at_time(self, t: float) -> HistoryFrame | None:
        """Find the frame closest to timestamp *t* using binary search."""
        with self._lock:
            if not self._timestamps:
                return None
            idx = bisect.bisect_left(self._timestamps, t)
            if idx == 0:
                return self._frames[0]
            if idx >= len(self._timestamps):
                return self._frames[-1]
            before = self._timestamps[idx - 1]
            after = self._timestamps[idx]
            if (t - before) <= (after - t):
                return self._frames[idx - 1]
            return self._frames[idx]

    def get_frames_in_time_range(
        self, t_start: float, t_end: float
    ) -> list[HistoryFrame]:
        """Return all frames with timestamps in ``[t_start, t_end]``."""
        with self._lock:
            if not self._timestamps:
                return []
            i_start = bisect.bisect_left(self._timestamps, t_start)
            i_end = bisect.bisect_right(self._timestamps, t_end)
            return list(self._frames[i_start:i_end])

    def clear(self) -> None:
        """Remove all frames."""
        with self._lock:
            self._frames.clear()
            self._timestamps.clear()

    def get_all_frames(self) -> list[HistoryFrame]:
        """Return a shallow copy of all frames (for export)."""
        with self._lock:
            return list(self._frames)

    def load_frames(self, frames: list[HistoryFrame]) -> None:
        """Replace buffer contents with *frames* (for import).

        Frames should be sorted by timestamp.  If ``len(frames)`` exceeds
        ``max_frames``, only the last ``max_frames`` are kept.
        """
        with self._lock:
            if len(frames) > self._max_frames:
                frames = frames[-self._max_frames :]
            self._frames = list(frames)
            self._timestamps = [f.timestamp for f in self._frames]

    @property
    def estimated_memory_bytes(self) -> int:
        """Rough total memory estimate."""
        with self._lock:
            return sum(f.estimated_size_bytes for f in self._frames)
