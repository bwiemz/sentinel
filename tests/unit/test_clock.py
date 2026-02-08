"""Tests for clock and timing utilities."""

import time

from sentinel.core.clock import FrameTimer, SystemClock


class TestSystemClock:
    def test_now_increases(self):
        clock = SystemClock()
        t1 = clock.now()
        time.sleep(0.01)
        t2 = clock.now()
        assert t2 > t1

    def test_elapsed(self):
        clock = SystemClock()
        time.sleep(0.05)
        assert clock.elapsed() >= 0.04  # Allow small tolerance


class TestFrameTimer:
    def test_fps_no_ticks(self):
        timer = FrameTimer()
        assert timer.fps == 0.0

    def test_fps_single_tick(self):
        timer = FrameTimer()
        timer.tick()
        assert timer.fps == 0.0

    def test_fps_multiple_ticks(self):
        timer = FrameTimer(window_size=10)
        for _ in range(5):
            timer.tick()
            time.sleep(0.02)
        fps = timer.fps
        # At 50Hz sleep, expect roughly 40-60 FPS (imprecise due to sleep)
        assert 10 < fps < 100

    def test_window_limit(self):
        timer = FrameTimer(window_size=5)
        for _ in range(20):
            timer.tick()
        assert len(timer._timestamps) <= 5
