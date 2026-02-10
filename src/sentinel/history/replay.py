"""ReplayController — playback of recorded history frames."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from sentinel.core.types import PlaybackState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.frame import HistoryFrame

logger = logging.getLogger(__name__)

# Valid playback speeds
VALID_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class ReplayController:
    """Controls playback of recorded history frames.

    The replay controller runs its own daemon thread that iterates
    through frames in the :class:`HistoryBuffer` and publishes them
    to a :class:`StateBuffer` (the same buffer the web UI reads from).

    Control methods (play/pause/stop/seek) are called from the web API
    thread.  All shared state is protected by ``self._lock``.
    """

    def __init__(
        self,
        state_buffer: Any | None = None,
        default_speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self._state_buffer = state_buffer
        self._buffer: HistoryBuffer | None = None
        self._state = PlaybackState.STOPPED
        self._speed = default_speed if default_speed in VALID_SPEEDS else 1.0
        self._loop = loop
        self._current_index = 0
        self._lock = threading.Lock()
        self._play_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def current_index(self) -> int:
        with self._lock:
            return self._current_index

    @property
    def total_frames(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.frame_count

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self, buffer: HistoryBuffer) -> None:
        """Load a HistoryBuffer for playback.  Stops any current playback."""
        self.stop()
        with self._lock:
            self._buffer = buffer
            self._current_index = 0
        logger.info("Replay loaded %d frames", buffer.frame_count)

    def set_state_buffer(self, state_buffer: Any) -> None:
        """Set or replace the output StateBuffer."""
        self._state_buffer = state_buffer

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback."""
        with self._lock:
            if self._buffer is None or self._buffer.frame_count == 0:
                logger.warning("Cannot play: no frames loaded")
                return
            self._state = PlaybackState.PLAYING

        self._play_event.set()

        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._playback_loop,
                name="sentinel-replay",
                daemon=True,
            )
            self._thread.start()

        logger.info("Replay playing at %.2fx", self._speed)

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED
        self._play_event.clear()
        logger.info("Replay paused at frame %d", self._current_index)

    def stop(self) -> None:
        """Stop playback and reset to frame 0."""
        self._stop_event.set()
        self._play_event.set()  # unblock if paused

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            self._state = PlaybackState.STOPPED
            self._current_index = 0
        self._thread = None
        logger.info("Replay stopped")

    def step_forward(self) -> HistoryFrame | None:
        """Advance one frame and publish it."""
        with self._lock:
            if self._buffer is None:
                return None
            if self._current_index >= self._buffer.frame_count:
                return None
            self._state = PlaybackState.STEPPING
            frame = self._buffer.get_frame(self._current_index)
            self._current_index += 1

        if frame is not None:
            self._publish_frame(frame)
        return frame

    def step_backward(self) -> HistoryFrame | None:
        """Go back one frame and publish it."""
        with self._lock:
            if self._buffer is None:
                return None
            if self._current_index <= 0:
                return None
            self._state = PlaybackState.STEPPING
            self._current_index -= 1
            frame = self._buffer.get_frame(self._current_index)

        if frame is not None:
            self._publish_frame(frame)
        return frame

    def seek_to_frame(self, index: int) -> HistoryFrame | None:
        """Seek to a specific frame index and publish it."""
        with self._lock:
            if self._buffer is None:
                return None
            index = max(0, min(index, self._buffer.frame_count - 1))
            self._current_index = index
            frame = self._buffer.get_frame(index)

        if frame is not None:
            self._publish_frame(frame)
        return frame

    def seek_to_time(self, t: float) -> HistoryFrame | None:
        """Seek to the frame closest to timestamp *t* and publish it."""
        with self._lock:
            if self._buffer is None:
                return None
            frame = self._buffer.get_frame_at_time(t)
            if frame is not None:
                # Find the buffer index of this frame
                all_frames = self._buffer.get_all_frames()
                for i, f in enumerate(all_frames):
                    if f.frame_number == frame.frame_number:
                        self._current_index = i
                        break

        if frame is not None:
            self._publish_frame(frame)
        return frame

    def set_speed(self, speed: float) -> None:
        """Set playback speed.  Clamps to nearest valid speed."""
        closest = min(VALID_SPEEDS, key=lambda s: abs(s - speed))
        with self._lock:
            self._speed = closest
        logger.info("Replay speed set to %.2fx", closest)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """Main playback thread loop."""
        while not self._stop_event.is_set():
            self._play_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break

            with self._lock:
                if self._state != PlaybackState.PLAYING:
                    continue
                if self._buffer is None:
                    break
                idx = self._current_index
                total = self._buffer.frame_count
                speed = self._speed

            if idx >= total:
                if self._loop:
                    with self._lock:
                        self._current_index = 0
                    continue
                else:
                    with self._lock:
                        self._state = PlaybackState.STOPPED
                    logger.info("Replay finished (end of recording)")
                    break

            frame = self._buffer.get_frame(idx)
            next_frame = (
                self._buffer.get_frame(idx + 1) if idx + 1 < total else None
            )

            if frame is not None:
                self._publish_frame(frame)

            with self._lock:
                self._current_index = idx + 1

            # Compute delay based on frame timestamps
            if frame is not None and next_frame is not None:
                dt = next_frame.timestamp - frame.timestamp
                if dt > 0 and speed > 0:
                    delay = min(dt / speed, 2.0)
                else:
                    delay = 0.01
            else:
                delay = 0.033  # ~30 fps fallback

            # Sleep in small increments for responsive stop/pause
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._stop_event.wait(timeout=min(remaining, 0.05))

    def _publish_frame(self, frame: HistoryFrame) -> None:
        """Publish a HistoryFrame to the StateBuffer for web display."""
        if self._state_buffer is None:
            return

        # Lazy import to avoid circular dependency
        from sentinel.ui.web.state_buffer import StateSnapshot

        status = dict(frame.system_status)
        status["replay_mode"] = True
        status["replay_frame"] = frame.frame_number
        status["replay_index"] = self._current_index
        status["replay_total"] = self.total_frames
        status["replay_speed"] = self._speed
        status["replay_state"] = self._state.value

        snapshot = StateSnapshot(
            timestamp=frame.timestamp,
            system_status=status,
            camera_tracks=frame.camera_tracks,
            radar_tracks=frame.radar_tracks,
            thermal_tracks=frame.thermal_tracks,
            fused_tracks=frame.fused_tracks,
            enhanced_fused_tracks=frame.enhanced_fused_tracks,
            hud_frame_jpeg=None,
        )
        self._state_buffer.update(snapshot)

    def get_status(self) -> dict[str, Any]:
        """Return replay status for web API."""
        with self._lock:
            total = self._buffer.frame_count if self._buffer else 0
            tr = self._buffer.time_range if self._buffer else None
            current_ts = None
            if self._buffer and self._current_index < total:
                f = self._buffer.get_frame(self._current_index)
                if f is not None:
                    current_ts = f.timestamp
            return {
                "state": self._state.value,
                "current_index": self._current_index,
                "total_frames": total,
                "speed": self._speed,
                "loop": self._loop,
                "time_range": list(tr) if tr else None,
                "current_timestamp": current_ts,
            }
