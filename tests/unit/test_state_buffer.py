"""Tests for the thread-safe StateBuffer."""

from __future__ import annotations

import threading
import time

import pytest

from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot


# ---------------------------------------------------------------------------
# StateSnapshot
# ---------------------------------------------------------------------------
class TestStateSnapshot:
    def test_default_empty(self):
        snap = StateSnapshot()
        assert snap.timestamp == 0.0
        assert snap.system_status == {}
        assert snap.camera_tracks == []
        assert snap.radar_tracks == []
        assert snap.thermal_tracks == []
        assert snap.fused_tracks == []
        assert snap.enhanced_fused_tracks == []
        assert snap.hud_frame_jpeg is None

    def test_with_data(self):
        snap = StateSnapshot(
            timestamp=123.0,
            system_status={"fps": 30},
            camera_tracks=[{"track_id": "A"}],
            radar_tracks=[{"track_id": "B"}],
            thermal_tracks=[{"track_id": "C"}],
            fused_tracks=[{"fused_id": "F"}],
            enhanced_fused_tracks=[{"fused_id": "E"}],
            hud_frame_jpeg=b"\xff\xd8",
        )
        assert snap.timestamp == 123.0
        assert snap.system_status["fps"] == 30
        assert len(snap.camera_tracks) == 1
        assert len(snap.radar_tracks) == 1
        assert len(snap.thermal_tracks) == 1
        assert len(snap.fused_tracks) == 1
        assert len(snap.enhanced_fused_tracks) == 1
        assert snap.hud_frame_jpeg == b"\xff\xd8"


# ---------------------------------------------------------------------------
# StateBuffer — basic read / write
# ---------------------------------------------------------------------------
class TestStateBufferBasic:
    def test_initial_snapshot_empty(self):
        buf = StateBuffer()
        snap = buf.get_snapshot()
        assert snap.timestamp == 0.0
        assert snap.camera_tracks == []

    def test_update_replaces_snapshot(self):
        buf = StateBuffer()
        snap1 = StateSnapshot(timestamp=1.0, camera_tracks=[{"id": "X"}])
        buf.update(snap1)
        assert buf.get_snapshot().timestamp == 1.0
        assert buf.get_snapshot().camera_tracks == [{"id": "X"}]

        snap2 = StateSnapshot(timestamp=2.0, radar_tracks=[{"id": "R"}])
        buf.update(snap2)
        assert buf.get_snapshot().timestamp == 2.0
        assert buf.get_snapshot().camera_tracks == []  # replaced entirely
        assert buf.get_snapshot().radar_tracks == [{"id": "R"}]

    def test_get_snapshot_returns_same_object(self):
        buf = StateBuffer()
        snap = StateSnapshot(timestamp=5.0)
        buf.update(snap)
        assert buf.get_snapshot() is snap


# ---------------------------------------------------------------------------
# StateBuffer — frame event
# ---------------------------------------------------------------------------
class TestStateBufferFrameEvent:
    def test_wait_for_frame_returns_none_on_timeout(self):
        buf = StateBuffer()
        result = buf.wait_for_frame(timeout=0.05)
        assert result is None

    def test_wait_for_frame_returns_jpeg_after_update(self):
        buf = StateBuffer()
        jpeg = b"\xff\xd8\xff\xe0FRAME"
        buf.update(StateSnapshot(hud_frame_jpeg=jpeg))
        result = buf.wait_for_frame(timeout=1.0)
        assert result == jpeg

    def test_wait_for_frame_clears_event(self):
        buf = StateBuffer()
        buf.update(StateSnapshot(hud_frame_jpeg=b"IMG"))
        buf.wait_for_frame(timeout=1.0)
        # Second call should timeout because event was cleared
        result = buf.wait_for_frame(timeout=0.05)
        assert result is None

    def test_no_event_set_when_frame_is_none(self):
        buf = StateBuffer()
        buf.update(StateSnapshot(hud_frame_jpeg=None))
        result = buf.wait_for_frame(timeout=0.05)
        assert result is None

    def test_frame_event_from_background_thread(self):
        buf = StateBuffer()
        jpeg = b"THREADED_FRAME"

        def writer():
            time.sleep(0.05)
            buf.update(StateSnapshot(hud_frame_jpeg=jpeg))

        t = threading.Thread(target=writer)
        t.start()
        result = buf.wait_for_frame(timeout=2.0)
        t.join()
        assert result == jpeg


# ---------------------------------------------------------------------------
# StateBuffer — thread safety
# ---------------------------------------------------------------------------
class TestStateBufferThreadSafety:
    def test_concurrent_reads_and_writes(self):
        """Multiple writers and readers should not crash or corrupt data."""
        buf = StateBuffer()
        errors: list[str] = []
        stop = threading.Event()

        def writer(writer_id: int):
            i = 0
            while not stop.is_set():
                snap = StateSnapshot(
                    timestamp=float(writer_id * 1000 + i),
                    camera_tracks=[{"id": f"W{writer_id}-{i}"}],
                )
                buf.update(snap)
                i += 1

        def reader():
            while not stop.is_set():
                snap = buf.get_snapshot()
                if not isinstance(snap, StateSnapshot):
                    errors.append(f"Bad type: {type(snap)}")
                if not isinstance(snap.camera_tracks, list):
                    errors.append(f"Bad tracks type: {type(snap.camera_tracks)}")

        threads = []
        for wid in range(2):
            threads.append(threading.Thread(target=writer, args=(wid,)))
        for _ in range(3):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()

        time.sleep(0.3)
        stop.set()

        for t in threads:
            t.join(timeout=2.0)

        assert errors == [], f"Thread safety errors: {errors}"
