"""Tests for sentinel.history.recorder — HistoryRecorder pipeline hook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from sentinel.core.types import RecordingState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.config import HistoryConfig
from sentinel.history.recorder import HistoryRecorder


# ---------------------------------------------------------------------------
# Mock pipeline
# ---------------------------------------------------------------------------


class MockClock:
    def __init__(self, start: float = 1000.0):
        self._now = start
        self._start = start

    def now(self) -> float:
        return self._now

    def elapsed(self) -> float:
        return self._now - self._start

    def advance(self, dt: float) -> None:
        self._now += dt


@dataclass
class MockTrack:
    track_id: str = "T1"
    score: float = 0.8

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "score": self.score}


@dataclass
class MockTrackManager:
    active_tracks: list[MockTrack] = field(default_factory=list)


class MockPipeline:
    def __init__(self, clock: MockClock | None = None):
        self._clock = clock or MockClock()
        self._latest_tracks = [MockTrack("CAM-1")]
        self._latest_fused_tracks: list = []
        self._latest_enhanced_fused: list = []
        self._radar_track_manager = MockTrackManager([MockTrack("RAD-1")])
        self._thermal_track_manager = None

    def get_system_status(self) -> dict[str, Any]:
        return {"fps": 30, "track_count": 2}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestRecorderLifecycle:
    def test_initial_state_idle(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        assert rec.state == RecordingState.IDLE

    def test_start(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        assert rec.state == RecordingState.RECORDING

    def test_stop(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        rec.stop()
        assert rec.state == RecordingState.IDLE

    def test_pause(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        rec.pause()
        assert rec.state == RecordingState.PAUSED

    def test_pause_when_idle_stays_idle(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.pause()
        assert rec.state == RecordingState.IDLE

    def test_resume_from_pause(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        rec.pause()
        rec.start()
        assert rec.state == RecordingState.RECORDING


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecorderCapture:
    def test_records_frame(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        assert rec.buffer.frame_count == 1

    def test_frame_has_camera_tracks(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        f = rec.buffer.get_frame(0)
        assert len(f.camera_tracks) == 1
        assert f.camera_tracks[0]["track_id"] == "CAM-1"

    def test_frame_has_radar_tracks(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        f = rec.buffer.get_frame(0)
        assert len(f.radar_tracks) == 1
        assert f.radar_tracks[0]["track_id"] == "RAD-1"

    def test_frame_has_system_status(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        f = rec.buffer.get_frame(0)
        assert f.system_status["fps"] == 30

    def test_frame_has_timestamp(self):
        cfg = HistoryConfig()
        clock = MockClock(1000.0)
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline(clock)
        rec.record_frame(pipe)
        f = rec.buffer.get_frame(0)
        assert f.timestamp == 1000.0

    def test_does_not_record_when_idle(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        pipe = MockPipeline()
        rec.record_frame(pipe)
        assert rec.buffer.frame_count == 0

    def test_does_not_record_when_paused(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        rec.pause()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        assert rec.buffer.frame_count == 0

    def test_multiple_frames(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        clock = MockClock(1000.0)
        pipe = MockPipeline(clock)
        for _ in range(10):
            rec.record_frame(pipe)
            clock.advance(0.1)
        assert rec.buffer.frame_count == 10
        assert rec.recorded_count == 10


# ---------------------------------------------------------------------------
# Capture interval
# ---------------------------------------------------------------------------


class TestRecorderCaptureInterval:
    def test_interval_1_records_every_frame(self):
        cfg = HistoryConfig(capture_interval=1)
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        for _ in range(5):
            rec.record_frame(pipe)
        assert rec.recorded_count == 5

    def test_interval_3_skips_frames(self):
        cfg = HistoryConfig(capture_interval=3)
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        for _ in range(9):
            rec.record_frame(pipe)
        # Frames 3, 6, 9 recorded
        assert rec.recorded_count == 3
        assert rec.frame_counter == 9

    def test_interval_5(self):
        cfg = HistoryConfig(capture_interval=5)
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        for _ in range(20):
            rec.record_frame(pipe)
        assert rec.recorded_count == 4


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestRecorderStatus:
    def test_status_idle(self):
        cfg = HistoryConfig(max_frames=500)
        rec = HistoryRecorder(cfg)
        s = rec.get_status()
        assert s["state"] == "idle"
        assert s["frame_counter"] == 0
        assert s["recorded_count"] == 0
        assert s["buffer_capacity"] == 500

    def test_status_recording(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)
        s = rec.get_status()
        assert s["state"] == "recording"
        assert s["frame_counter"] == 1
        assert s["recorded_count"] == 1
        assert s["buffer_frames"] == 1

    def test_status_has_time_range(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        clock = MockClock(1000.0)
        pipe = MockPipeline(clock)
        rec.record_frame(pipe)
        clock.advance(1.0)
        rec.record_frame(pipe)
        s = rec.get_status()
        assert s["time_range"] is not None
        assert len(s["time_range"]) == 2


# ---------------------------------------------------------------------------
# Custom buffer
# ---------------------------------------------------------------------------


class TestRecorderCustomBuffer:
    def test_uses_provided_buffer(self):
        buf = HistoryBuffer(max_frames=50)
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg, buffer=buf)
        assert rec.buffer is buf

    def test_creates_buffer_from_config(self):
        cfg = HistoryConfig(max_frames=42)
        rec = HistoryRecorder(cfg)
        assert rec.buffer.max_frames == 42
