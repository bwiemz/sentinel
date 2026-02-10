"""Scenario tests for the history & replay system.

Tests realistic usage patterns: long recording sessions, ring buffer overflow,
multi-sensor capture, capture interval skipping, speed control, and
interrupted recording recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from sentinel.core.types import RecordingState, PlaybackState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.config import HistoryConfig
from sentinel.history.frame import HistoryFrame
from sentinel.history.recorder import HistoryRecorder
from sentinel.history.replay import ReplayController
from sentinel.history.storage import HistoryStorage
from sentinel.ui.web.state_buffer import StateBuffer


# ---------------------------------------------------------------------------
# Helpers
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
    def __init__(self, clock: MockClock, n_cam=2, n_radar=1, n_thermal=0):
        self._clock = clock
        self._latest_tracks = [MockTrack(f"CAM-{i}") for i in range(n_cam)]
        self._latest_fused_tracks = []
        self._latest_enhanced_fused = []
        self._radar_track_manager = MockTrackManager(
            [MockTrack(f"RAD-{i}") for i in range(n_radar)]
        ) if n_radar > 0 else None
        self._thermal_track_manager = MockTrackManager(
            [MockTrack(f"THM-{i}") for i in range(n_thermal)]
        ) if n_thermal > 0 else None

    def get_system_status(self) -> dict[str, Any]:
        return {"fps": 30, "track_count": 5}


# ---------------------------------------------------------------------------
# Scenario 1: Long recording exceeds buffer capacity
# ---------------------------------------------------------------------------


class TestLongRecordingOverflow:
    def test_ring_buffer_drops_oldest(self):
        """Recording 200 frames into a 100-frame buffer keeps only latest 100."""
        cfg = HistoryConfig(max_frames=100)
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock)
        for _ in range(200):
            rec.record_frame(pipe)
            clock.advance(0.033)

        assert rec.buffer.frame_count == 100
        assert rec.recorded_count == 200
        # First frame in buffer should be frame 101 (1-based, oldest 100 evicted)
        f = rec.buffer.get_frame(0)
        assert f.frame_number == 101

    def test_time_range_correct_after_overflow(self):
        cfg = HistoryConfig(max_frames=50)
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock(1000.0)
        pipe = MockPipeline(clock)
        for _ in range(100):
            rec.record_frame(pipe)
            clock.advance(1.0)

        tr = rec.buffer.time_range
        assert tr is not None
        # Should span the last 50 frames
        assert tr[1] - tr[0] == pytest.approx(49.0, abs=0.1)


# ---------------------------------------------------------------------------
# Scenario 2: Multi-sensor capture
# ---------------------------------------------------------------------------


class TestMultiSensorCapture:
    def test_captures_all_sensor_types(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock, n_cam=3, n_radar=2, n_thermal=1)
        rec.record_frame(pipe)

        f = rec.buffer.get_frame(0)
        assert len(f.camera_tracks) == 3
        assert len(f.radar_tracks) == 2
        assert len(f.thermal_tracks) == 1

    def test_missing_sensors_give_empty_lists(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock, n_cam=1, n_radar=0, n_thermal=0)
        rec.record_frame(pipe)

        f = rec.buffer.get_frame(0)
        assert len(f.camera_tracks) == 1
        assert len(f.radar_tracks) == 0
        assert len(f.thermal_tracks) == 0


# ---------------------------------------------------------------------------
# Scenario 3: Capture interval skipping under load
# ---------------------------------------------------------------------------


class TestCaptureIntervalSkipping:
    def test_interval_10_at_30fps_for_60s(self):
        """30 FPS * 60s = 1800 frames, interval 10 → 180 captured."""
        cfg = HistoryConfig(capture_interval=10, max_frames=5000)
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock)
        for _ in range(1800):
            rec.record_frame(pipe)
            clock.advance(1.0 / 30)

        assert rec.frame_counter == 1800
        assert rec.recorded_count == 180


# ---------------------------------------------------------------------------
# Scenario 4: Replay speed changes
# ---------------------------------------------------------------------------


class TestReplaySpeedScenario:
    def _make_buffer(self, n: int = 10) -> HistoryBuffer:
        buf = HistoryBuffer(max_frames=100)
        for i in range(n):
            buf.record(HistoryFrame(
                frame_number=i,
                timestamp=1000.0 + i * 1.0,
                elapsed=float(i),
                camera_tracks=[{"id": f"T{i}"}],
            ))
        return buf

    def test_speed_changes_reflected_in_status(self):
        buf = self._make_buffer(10)
        ctrl = ReplayController(state_buffer=StateBuffer())
        ctrl.load(buf)

        ctrl.set_speed(4.0)
        assert ctrl.get_status()["speed"] == 4.0

        ctrl.set_speed(0.25)
        assert ctrl.get_status()["speed"] == 0.25

    def test_invalid_speed_clamped(self):
        buf = self._make_buffer(5)
        ctrl = ReplayController(state_buffer=StateBuffer())
        ctrl.load(buf)

        ctrl.set_speed(100.0)
        assert ctrl.get_status()["speed"] == 8.0

        ctrl.set_speed(0.01)
        assert ctrl.get_status()["speed"] == 0.25


# ---------------------------------------------------------------------------
# Scenario 5: Export, modify, re-import
# ---------------------------------------------------------------------------


class TestExportReimport:
    def test_export_reimport_different_buffer_size(self, tmp_path):
        """Export 100 frames, reimport into a smaller buffer."""
        buf = HistoryBuffer(max_frames=200)
        for i in range(100):
            buf.record(HistoryFrame(
                frame_number=i,
                timestamp=float(i),
                elapsed=float(i),
            ))

        path = tmp_path / "big.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path, max_frames=30)
        assert loaded.frame_count == 30
        # Should keep the last 30 frames
        assert loaded.get_frame(0).frame_number == 70


# ---------------------------------------------------------------------------
# Scenario 6: Pause/resume recording
# ---------------------------------------------------------------------------


class TestPauseResumeRecording:
    def test_pause_stops_then_resumes(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock)

        # Record 5 frames
        for _ in range(5):
            rec.record_frame(pipe)
            clock.advance(0.1)
        assert rec.recorded_count == 5

        # Pause — recording stops
        rec.pause()
        for _ in range(5):
            rec.record_frame(pipe)
            clock.advance(0.1)
        assert rec.recorded_count == 5  # unchanged

        # Resume
        rec.start()
        for _ in range(5):
            rec.record_frame(pipe)
            clock.advance(0.1)
        assert rec.recorded_count == 10

    def test_pause_state_transitions(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)

        assert rec.state == RecordingState.IDLE
        rec.start()
        assert rec.state == RecordingState.RECORDING
        rec.pause()
        assert rec.state == RecordingState.PAUSED
        rec.start()
        assert rec.state == RecordingState.RECORDING
        rec.stop()
        assert rec.state == RecordingState.IDLE
