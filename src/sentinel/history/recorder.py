"""HistoryRecorder — captures pipeline state into a HistoryBuffer."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from sentinel.core.types import RecordingState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.config import HistoryConfig
from sentinel.history.frame import HistoryFrame

logger = logging.getLogger(__name__)


class HistoryRecorder:
    """Records pipeline state into a HistoryBuffer.

    Called by the pipeline after fusion + engagement.  Thread-safe:
    the recorder can be started/stopped from any thread while the
    pipeline records from its main thread.
    """

    def __init__(
        self,
        config: HistoryConfig,
        buffer: HistoryBuffer | None = None,
    ) -> None:
        self._config = config
        self._buffer = buffer or HistoryBuffer(max_frames=config.max_frames)
        self._state = RecordingState.IDLE
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._recorded_count = 0
        self._start_time: float | None = None

    @property
    def state(self) -> RecordingState:
        with self._lock:
            return self._state

    @property
    def buffer(self) -> HistoryBuffer:
        return self._buffer

    @property
    def frame_counter(self) -> int:
        with self._lock:
            return self._frame_counter

    @property
    def recorded_count(self) -> int:
        with self._lock:
            return self._recorded_count

    def start(self) -> None:
        """Start recording.  If paused, resumes."""
        with self._lock:
            self._state = RecordingState.RECORDING
            if self._start_time is None:
                self._start_time = time.monotonic()
        logger.info("History recording started")

    def stop(self) -> None:
        """Stop recording."""
        with self._lock:
            self._state = RecordingState.IDLE
        logger.info(
            "History recording stopped (%d frames recorded)", self._recorded_count
        )

    def pause(self) -> None:
        """Pause recording (frames are skipped)."""
        with self._lock:
            if self._state == RecordingState.RECORDING:
                self._state = RecordingState.PAUSED
        logger.info("History recording paused")

    def record_frame(self, pipeline: Any) -> None:
        """Called by the pipeline each frame.  Captures state if recording."""
        with self._lock:
            self._frame_counter += 1
            if self._state != RecordingState.RECORDING:
                return
            if self._frame_counter % self._config.capture_interval != 0:
                return

        frame = self._capture_frame(pipeline)
        self._buffer.record(frame)

        with self._lock:
            self._recorded_count += 1

    def _capture_frame(self, pipeline: Any) -> HistoryFrame:
        """Extract current pipeline state into a HistoryFrame."""
        # Camera tracks
        camera_tracks: list[dict] = []
        for t in getattr(pipeline, "_latest_tracks", []) or []:
            try:
                camera_tracks.append(t.to_dict())
            except Exception:
                pass

        # Radar tracks
        radar_tracks: list[dict] = []
        radar_mgr = getattr(pipeline, "_radar_track_manager", None)
        if radar_mgr is not None:
            for t in getattr(radar_mgr, "active_tracks", []):
                if getattr(t, "is_alive", True):
                    try:
                        radar_tracks.append(t.to_dict())
                    except Exception:
                        pass

        # Thermal tracks
        thermal_tracks: list[dict] = []
        thermal_mgr = getattr(pipeline, "_thermal_track_manager", None)
        if thermal_mgr is not None:
            for t in getattr(thermal_mgr, "active_tracks", []):
                if getattr(t, "is_alive", True):
                    try:
                        thermal_tracks.append(t.to_dict())
                    except Exception:
                        pass

        # Fused tracks
        fused_tracks: list[dict] = []
        for ft in getattr(pipeline, "_latest_fused_tracks", []) or []:
            try:
                fused_tracks.append(ft.to_dict())
            except Exception:
                pass

        # Enhanced fused tracks
        enhanced_fused: list[dict] = []
        for eft in getattr(pipeline, "_latest_enhanced_fused", []) or []:
            try:
                enhanced_fused.append(eft.to_dict())
            except Exception:
                pass

        # System status
        try:
            system_status = pipeline.get_system_status()
        except Exception:
            system_status = {}

        # Clock info
        clock = getattr(pipeline, "_clock", None)
        timestamp = clock.now() if clock else time.time()
        elapsed = clock.elapsed() if clock else 0.0

        return HistoryFrame(
            frame_number=self._frame_counter,
            timestamp=timestamp,
            elapsed=elapsed,
            camera_tracks=camera_tracks,
            radar_tracks=radar_tracks,
            thermal_tracks=thermal_tracks,
            fused_tracks=fused_tracks,
            enhanced_fused_tracks=enhanced_fused,
            system_status=system_status,
        )

    def get_status(self) -> dict[str, Any]:
        """Return recording status for web API / system status."""
        tr = self._buffer.time_range
        with self._lock:
            return {
                "state": self._state.value,
                "frame_counter": self._frame_counter,
                "recorded_count": self._recorded_count,
                "buffer_frames": self._buffer.frame_count,
                "buffer_capacity": self._buffer.max_frames,
                "time_range": list(tr) if tr else None,
                "estimated_memory_bytes": self._buffer.estimated_memory_bytes,
                "capture_interval": self._config.capture_interval,
            }
