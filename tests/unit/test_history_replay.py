"""Tests for sentinel.history.replay — ReplayController playback."""

from __future__ import annotations

import time

import pytest

from sentinel.core.types import PlaybackState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.frame import HistoryFrame
from sentinel.history.replay import ReplayController, VALID_SPEEDS
from sentinel.ui.web.state_buffer import StateBuffer


def _frame(n: int, ts: float) -> HistoryFrame:
    return HistoryFrame(
        frame_number=n,
        timestamp=ts,
        elapsed=ts - 1000.0,
        camera_tracks=[{"track_id": f"T{n}"}],
        system_status={"fps": 30},
    )


def _buffer(n: int = 10) -> HistoryBuffer:
    b = HistoryBuffer(max_frames=100)
    for i in range(n):
        b.record(_frame(i, 1000.0 + i * 0.1))
    return b


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestReplayLifecycle:
    def test_initial_state(self):
        rc = ReplayController()
        assert rc.state == PlaybackState.STOPPED
        assert rc.total_frames == 0
        assert rc.current_index == 0

    def test_load(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        assert rc.total_frames == 5
        assert rc.current_index == 0

    def test_load_resets_index(self):
        rc = ReplayController()
        buf = _buffer(5)
        rc.load(buf)
        rc.step_forward()
        rc.step_forward()
        assert rc.current_index == 2
        rc.load(buf)
        assert rc.current_index == 0

    def test_stop_resets_to_zero(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        rc.step_forward()
        rc.step_forward()
        rc.stop()
        assert rc.current_index == 0
        assert rc.state == PlaybackState.STOPPED


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


class TestReplayStepForward:
    def test_step_forward(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        f = rc.step_forward()
        assert f is not None
        assert f.frame_number == 0
        assert rc.current_index == 1

    def test_step_forward_multiple(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        for i in range(5):
            f = rc.step_forward()
            assert f is not None
            assert f.frame_number == i

    def test_step_forward_past_end(self):
        rc = ReplayController()
        rc.load(_buffer(2))
        rc.step_forward()
        rc.step_forward()
        f = rc.step_forward()
        assert f is None

    def test_step_forward_empty_buffer(self):
        rc = ReplayController()
        f = rc.step_forward()
        assert f is None

    def test_state_is_stepping(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        rc.step_forward()
        assert rc.state == PlaybackState.STEPPING


class TestReplayStepBackward:
    def test_step_backward(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        rc.step_forward()  # index 0 → 1
        rc.step_forward()  # index 1 → 2
        f = rc.step_backward()  # index 2 → 1, returns frame[1]
        assert f is not None
        assert f.frame_number == 1
        assert rc.current_index == 1

    def test_step_backward_at_start(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        f = rc.step_backward()
        assert f is None


# ---------------------------------------------------------------------------
# Seeking
# ---------------------------------------------------------------------------


class TestReplaySeeking:
    def test_seek_to_frame(self):
        rc = ReplayController()
        rc.load(_buffer(10))
        f = rc.seek_to_frame(5)
        assert f is not None
        assert f.frame_number == 5
        assert rc.current_index == 5

    def test_seek_to_frame_clamped_high(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        f = rc.seek_to_frame(100)
        assert f is not None
        assert rc.current_index == 4  # clamped to last

    def test_seek_to_frame_clamped_low(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        f = rc.seek_to_frame(-10)
        assert f is not None
        assert rc.current_index == 0

    def test_seek_to_time(self):
        rc = ReplayController()
        rc.load(_buffer(10))
        # Frames at 1000.0, 1000.1, ..., 1000.9
        f = rc.seek_to_time(1000.5)
        assert f is not None
        assert f.frame_number == 5

    def test_seek_no_buffer(self):
        rc = ReplayController()
        assert rc.seek_to_frame(0) is None
        assert rc.seek_to_time(0.0) is None


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


class TestReplaySpeed:
    def test_default_speed(self):
        rc = ReplayController(default_speed=1.0)
        assert rc.speed == 1.0

    def test_set_speed(self):
        rc = ReplayController()
        rc.set_speed(4.0)
        assert rc.speed == 4.0

    def test_set_speed_clamps_to_nearest(self):
        rc = ReplayController()
        rc.set_speed(3.0)
        assert rc.speed == 2.0  # nearest valid

    def test_set_speed_minimum(self):
        rc = ReplayController()
        rc.set_speed(0.01)
        assert rc.speed == 0.25

    def test_set_speed_maximum(self):
        rc = ReplayController()
        rc.set_speed(100.0)
        assert rc.speed == 8.0

    def test_invalid_default_speed_uses_1x(self):
        rc = ReplayController(default_speed=99.0)
        assert rc.speed == 1.0


# ---------------------------------------------------------------------------
# Publishing to StateBuffer
# ---------------------------------------------------------------------------


class TestReplayPublish:
    def test_step_publishes_to_state_buffer(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb)
        rc.load(_buffer(5))
        rc.step_forward()

        snap = sb.get_snapshot()
        assert snap.timestamp == 1000.0
        assert len(snap.camera_tracks) == 1
        assert snap.system_status.get("replay_mode") is True

    def test_replay_metadata_in_status(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb)
        rc.load(_buffer(5))
        rc.step_forward()

        snap = sb.get_snapshot()
        assert snap.system_status["replay_frame"] == 0
        assert snap.system_status["replay_total"] == 5

    def test_no_hud_frame(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb)
        rc.load(_buffer(5))
        rc.step_forward()
        assert sb.get_snapshot().hud_frame_jpeg is None


# ---------------------------------------------------------------------------
# Playback thread
# ---------------------------------------------------------------------------


class TestReplayPlayback:
    def test_play_advances_frames(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb, default_speed=8.0)
        # 5 frames at 0.1s intervals => should finish in < 1s at 8x
        rc.load(_buffer(5))
        rc.play()
        time.sleep(1.0)
        # Should have finished
        assert rc.state == PlaybackState.STOPPED
        assert rc.current_index == 5

    def test_pause_stops_advancing(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb, default_speed=1.0)
        rc.load(_buffer(100))  # 100 frames at 0.1s = 10s
        rc.play()
        time.sleep(0.2)
        rc.pause()
        assert rc.state == PlaybackState.PAUSED
        idx_at_pause = rc.current_index
        time.sleep(0.3)
        assert rc.current_index == idx_at_pause

    def test_stop_during_play(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb, default_speed=1.0)
        rc.load(_buffer(100))
        rc.play()
        time.sleep(0.15)
        rc.stop()
        assert rc.state == PlaybackState.STOPPED
        assert rc.current_index == 0

    def test_play_without_load(self):
        rc = ReplayController()
        rc.play()
        assert rc.state == PlaybackState.STOPPED  # stays stopped

    def test_loop_mode(self):
        sb = StateBuffer()
        rc = ReplayController(state_buffer=sb, default_speed=8.0, loop=True)
        rc.load(_buffer(3))
        rc.play()
        time.sleep(0.5)
        # Should have looped at least once
        rc.stop()
        # Just verify it didn't crash and is cleanly stopped
        assert rc.state == PlaybackState.STOPPED


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestReplayStatus:
    def test_status_stopped(self):
        rc = ReplayController()
        s = rc.get_status()
        assert s["state"] == "stopped"
        assert s["total_frames"] == 0

    def test_status_with_buffer(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        s = rc.get_status()
        assert s["total_frames"] == 5
        assert s["time_range"] is not None

    def test_status_speed(self):
        rc = ReplayController(default_speed=2.0)
        s = rc.get_status()
        assert s["speed"] == 2.0

    def test_status_current_timestamp(self):
        rc = ReplayController()
        rc.load(_buffer(5))
        rc.step_forward()
        s = rc.get_status()
        assert s["current_timestamp"] is not None
