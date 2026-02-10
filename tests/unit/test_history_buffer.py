"""Tests for sentinel.history.buffer — HistoryBuffer ring buffer."""

from __future__ import annotations

import threading

import pytest

from sentinel.history.buffer import HistoryBuffer
from sentinel.history.frame import HistoryFrame


def _frame(n: int, ts: float = 0.0) -> HistoryFrame:
    """Helper to create a minimal frame."""
    return HistoryFrame(frame_number=n, timestamp=ts, elapsed=ts)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHistoryBufferInit:
    def test_default_capacity(self):
        b = HistoryBuffer()
        assert b.max_frames == 10000

    def test_custom_capacity(self):
        b = HistoryBuffer(max_frames=50)
        assert b.max_frames == 50

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="max_frames must be >= 1"):
            HistoryBuffer(max_frames=0)

    def test_negative_capacity(self):
        with pytest.raises(ValueError):
            HistoryBuffer(max_frames=-5)

    def test_empty_on_creation(self):
        b = HistoryBuffer(max_frames=100)
        assert b.frame_count == 0
        assert b.time_range is None


# ---------------------------------------------------------------------------
# Basic record / get
# ---------------------------------------------------------------------------


class TestHistoryBufferRecord:
    def test_record_single(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 100.0))
        assert b.frame_count == 1

    def test_get_frame_by_index(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 100.0))
        f = b.get_frame(0)
        assert f is not None
        assert f.frame_number == 1

    def test_get_frame_negative_index(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 100.0))
        b.record(_frame(2, 101.0))
        f = b.get_frame(-1)
        assert f is not None
        assert f.frame_number == 2

    def test_get_frame_out_of_range(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 100.0))
        assert b.get_frame(5) is None

    def test_get_frame_empty_buffer(self):
        b = HistoryBuffer(max_frames=10)
        assert b.get_frame(0) is None

    def test_record_preserves_order(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(5):
            b.record(_frame(i, 100.0 + i))
        assert b.frame_count == 5
        for i in range(5):
            f = b.get_frame(i)
            assert f is not None
            assert f.frame_number == i


# ---------------------------------------------------------------------------
# Ring buffer eviction
# ---------------------------------------------------------------------------


class TestHistoryBufferEviction:
    def test_eviction_at_capacity(self):
        b = HistoryBuffer(max_frames=3)
        b.record(_frame(1, 1.0))
        b.record(_frame(2, 2.0))
        b.record(_frame(3, 3.0))
        assert b.frame_count == 3

        b.record(_frame(4, 4.0))
        assert b.frame_count == 3
        # Oldest (frame 1) should be gone
        assert b.get_frame(0).frame_number == 2
        assert b.get_frame(-1).frame_number == 4

    def test_eviction_multiple(self):
        b = HistoryBuffer(max_frames=2)
        for i in range(10):
            b.record(_frame(i, float(i)))
        assert b.frame_count == 2
        assert b.get_frame(0).frame_number == 8
        assert b.get_frame(1).frame_number == 9

    def test_time_range_after_eviction(self):
        b = HistoryBuffer(max_frames=3)
        for i in range(5):
            b.record(_frame(i, 100.0 + i))
        tr = b.time_range
        assert tr is not None
        assert tr[0] == 102.0
        assert tr[1] == 104.0


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------


class TestHistoryBufferTimeRange:
    def test_empty(self):
        b = HistoryBuffer(max_frames=10)
        assert b.time_range is None

    def test_single_frame(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 42.0))
        assert b.time_range == (42.0, 42.0)

    def test_multiple_frames(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        b.record(_frame(3, 30.0))
        assert b.time_range == (10.0, 30.0)


# ---------------------------------------------------------------------------
# Time-based search
# ---------------------------------------------------------------------------


class TestHistoryBufferTimeSearch:
    def test_exact_match(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        b.record(_frame(3, 30.0))
        f = b.get_frame_at_time(20.0)
        assert f is not None
        assert f.frame_number == 2

    def test_between_frames_closer_to_first(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        f = b.get_frame_at_time(14.0)
        assert f is not None
        assert f.frame_number == 1

    def test_between_frames_closer_to_second(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        f = b.get_frame_at_time(16.0)
        assert f is not None
        assert f.frame_number == 2

    def test_before_first(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        f = b.get_frame_at_time(5.0)
        assert f is not None
        assert f.frame_number == 1

    def test_after_last(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 10.0))
        b.record(_frame(2, 20.0))
        f = b.get_frame_at_time(99.0)
        assert f is not None
        assert f.frame_number == 2

    def test_empty_buffer(self):
        b = HistoryBuffer(max_frames=10)
        assert b.get_frame_at_time(10.0) is None


# ---------------------------------------------------------------------------
# Range queries
# ---------------------------------------------------------------------------


class TestHistoryBufferRangeQueries:
    def test_get_range(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(5):
            b.record(_frame(i, float(i)))
        frames = b.get_range(1, 3)
        assert len(frames) == 2
        assert frames[0].frame_number == 1
        assert frames[1].frame_number == 2

    def test_get_range_clamped(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(1, 1.0))
        frames = b.get_range(-5, 100)
        assert len(frames) == 1

    def test_get_frames_in_time_range(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(5):
            b.record(_frame(i, 10.0 + i))
        frames = b.get_frames_in_time_range(11.0, 13.0)
        assert len(frames) == 3
        assert frames[0].frame_number == 1
        assert frames[-1].frame_number == 3

    def test_get_frames_in_time_range_empty(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(3):
            b.record(_frame(i, 10.0 + i))
        frames = b.get_frames_in_time_range(50.0, 60.0)
        assert frames == []

    def test_get_frames_in_time_range_empty_buffer(self):
        b = HistoryBuffer(max_frames=10)
        assert b.get_frames_in_time_range(0, 100) == []


# ---------------------------------------------------------------------------
# Clear / load
# ---------------------------------------------------------------------------


class TestHistoryBufferClearLoad:
    def test_clear(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(5):
            b.record(_frame(i, float(i)))
        b.clear()
        assert b.frame_count == 0
        assert b.time_range is None

    def test_load_frames(self):
        b = HistoryBuffer(max_frames=10)
        frames = [_frame(i, float(i)) for i in range(5)]
        b.load_frames(frames)
        assert b.frame_count == 5
        assert b.get_frame(0).frame_number == 0

    def test_load_frames_truncation(self):
        b = HistoryBuffer(max_frames=3)
        frames = [_frame(i, float(i)) for i in range(10)]
        b.load_frames(frames)
        assert b.frame_count == 3
        assert b.get_frame(0).frame_number == 7

    def test_load_replaces_existing(self):
        b = HistoryBuffer(max_frames=10)
        b.record(_frame(99, 99.0))
        b.load_frames([_frame(0, 0.0)])
        assert b.frame_count == 1
        assert b.get_frame(0).frame_number == 0

    def test_get_all_frames(self):
        b = HistoryBuffer(max_frames=10)
        for i in range(3):
            b.record(_frame(i, float(i)))
        all_frames = b.get_all_frames()
        assert len(all_frames) == 3
        assert all_frames[0].frame_number == 0


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------


class TestHistoryBufferMemory:
    def test_empty_memory(self):
        b = HistoryBuffer(max_frames=10)
        assert b.estimated_memory_bytes == 0

    def test_memory_grows_with_frames(self):
        b = HistoryBuffer(max_frames=100)
        b.record(_frame(1, 1.0))
        assert b.estimated_memory_bytes == 500  # 0 tracks + 500 base


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestHistoryBufferThreadSafety:
    def test_concurrent_record(self):
        b = HistoryBuffer(max_frames=1000)
        errors: list[Exception] = []

        def writer(start: int):
            try:
                for i in range(100):
                    b.record(_frame(start + i, float(start + i)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert b.frame_count == 500

    def test_concurrent_read_write(self):
        b = HistoryBuffer(max_frames=100)
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(100):
                    b.record(_frame(i, float(i)))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    b.get_frame(0)
                    b.frame_count
                    b.time_range
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
