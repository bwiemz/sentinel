"""Thread-safe state buffer for pipeline -> web server data flow."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateSnapshot:
    """Immutable snapshot of pipeline state at a point in time.

    All fields are plain Python types (dicts, lists, bytes) -- no numpy
    arrays or live track objects.  This makes the snapshot safe to read
    from any thread without holding a lock on pipeline internals.
    """

    timestamp: float = 0.0
    system_status: dict[str, Any] = field(default_factory=dict)
    camera_tracks: list[dict] = field(default_factory=list)
    radar_tracks: list[dict] = field(default_factory=list)
    thermal_tracks: list[dict] = field(default_factory=list)
    fused_tracks: list[dict] = field(default_factory=list)
    enhanced_fused_tracks: list[dict] = field(default_factory=list)
    hud_frame_jpeg: bytes | None = None


class StateBuffer:
    """Thread-safe double-buffered state for pipeline -> web server data flow.

    The pipeline calls :meth:`update` from the main thread.
    The web server calls :meth:`get_snapshot` and :meth:`wait_for_frame`
    from the server thread.

    Uses a :class:`threading.Lock` for mutual exclusion with minimal hold
    time (replacing / reading a single reference).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = StateSnapshot()
        self._frame_event = threading.Event()

    def update(self, snapshot: StateSnapshot) -> None:
        """Publish a new state snapshot (called from the pipeline thread)."""
        with self._lock:
            self._snapshot = snapshot
        if snapshot.hud_frame_jpeg is not None:
            self._frame_event.set()

    def get_snapshot(self) -> StateSnapshot:
        """Return the latest snapshot (called from the web server thread)."""
        with self._lock:
            return self._snapshot

    def wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        """Block until a new HUD frame is available, or *timeout* elapses.

        Returns JPEG bytes or ``None`` on timeout.  Clears the internal
        event so the next call blocks again.
        """
        if self._frame_event.wait(timeout=timeout):
            with self._lock:
                self._frame_event.clear()
                return self._snapshot.hud_frame_jpeg
        return None
