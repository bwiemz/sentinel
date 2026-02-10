"""Integration tests for the history & replay system.

Tests end-to-end workflows: record → export → import → replay, config loading
from OmegaConf, pipeline integration via system status, and web API surface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import RecordingState, PlaybackState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.config import HistoryConfig
from sentinel.history.frame import HistoryFrame
from sentinel.history.recorder import HistoryRecorder
from sentinel.history.replay import ReplayController
from sentinel.history.storage import HistoryStorage
from sentinel.ui.web.state_buffer import StateBuffer


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
    def __init__(self, clock: MockClock | None = None, n_cam: int = 2, n_radar: int = 1):
        self._clock = clock or MockClock()
        self._latest_tracks = [MockTrack(f"CAM-{i}") for i in range(n_cam)]
        self._latest_fused_tracks: list = []
        self._latest_enhanced_fused: list = []
        self._radar_track_manager = MockTrackManager(
            [MockTrack(f"RAD-{i}") for i in range(n_radar)]
        )
        self._thermal_track_manager = None

    def get_system_status(self) -> dict[str, Any]:
        return {"fps": 30, "track_count": 3}


# ---------------------------------------------------------------------------
# Record → export → import roundtrip
# ---------------------------------------------------------------------------


class TestRecordExportImportRoundtrip:
    def test_full_roundtrip_json(self, tmp_path):
        """Record frames, export JSON, import, verify data."""
        cfg = HistoryConfig(max_frames=100)
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock(1000.0)
        pipe = MockPipeline(clock, n_cam=3, n_radar=2)
        for _ in range(20):
            rec.record_frame(pipe)
            clock.advance(0.1)

        assert rec.recorded_count == 20

        # Export
        path = tmp_path / "recording.json"
        HistoryStorage.save(rec.buffer, path, fmt="json")

        # Import
        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 20

        # Verify data integrity (recorder uses 1-based frame numbers)
        for i in range(20):
            f = loaded.get_frame(i)
            assert f.frame_number == i + 1
            assert len(f.camera_tracks) == 3
            assert len(f.radar_tracks) == 2

    def test_full_roundtrip_msgpack(self, tmp_path):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock)
        for _ in range(10):
            rec.record_frame(pipe)
            clock.advance(0.5)

        path = tmp_path / "recording.msgpack"
        HistoryStorage.save(rec.buffer, path, fmt="msgpack")
        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 10

    def test_roundtrip_compressed(self, tmp_path):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        pipe = MockPipeline()
        for _ in range(15):
            rec.record_frame(pipe)

        path = tmp_path / "rec.json.gz"
        HistoryStorage.save(rec.buffer, path, fmt="json", compression=True)
        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 15

    def test_roundtrip_preserves_system_status(self, tmp_path):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()
        pipe = MockPipeline()
        rec.record_frame(pipe)

        path = tmp_path / "status.json"
        HistoryStorage.save(rec.buffer, path, fmt="json")
        loaded = HistoryStorage.load(path)
        f = loaded.get_frame(0)
        assert f.system_status["fps"] == 30

    def test_roundtrip_with_numpy_arrays(self, tmp_path):
        """Tracks with numpy arrays survive the full pipeline."""
        buf = HistoryBuffer(max_frames=10)
        buf.record(HistoryFrame(
            frame_number=0,
            timestamp=0.0,
            elapsed=0.0,
            camera_tracks=[{"position": np.array([1.0, 2.0, 3.0])}],
        ))

        path = tmp_path / "numpy.json"
        HistoryStorage.save(buf, path, fmt="json")
        loaded = HistoryStorage.load(path)
        track = loaded.get_frame(0).camera_tracks[0]
        np.testing.assert_array_almost_equal(track["position"], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Replay from recorded buffer
# ---------------------------------------------------------------------------


class TestReplayFromRecording:
    def test_replay_recorded_data(self):
        """Record frames, then replay them via ReplayController."""
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock(1000.0)
        pipe = MockPipeline(clock, n_cam=2)
        for _ in range(5):
            rec.record_frame(pipe)
            clock.advance(1.0)

        rec.stop()
        assert rec.buffer.frame_count == 5

        state_buf = StateBuffer()
        ctrl = ReplayController(state_buffer=state_buf)
        ctrl.load(rec.buffer)

        status = ctrl.get_status()
        assert status["total_frames"] == 5
        assert status["state"] == "stopped"

    def test_step_through_recorded_data(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock()
        pipe = MockPipeline(clock)
        for _ in range(5):
            rec.record_frame(pipe)
            clock.advance(0.5)
        rec.stop()

        state_buf = StateBuffer()
        ctrl = ReplayController(state_buffer=state_buf)
        ctrl.load(rec.buffer)

        ctrl.step_forward()
        assert ctrl.get_status()["current_index"] == 1
        ctrl.step_forward()
        assert ctrl.get_status()["current_index"] == 2
        ctrl.step_backward()
        assert ctrl.get_status()["current_index"] == 1

    def test_replay_publishes_to_state_buffer(self):
        cfg = HistoryConfig()
        rec = HistoryRecorder(cfg)
        rec.start()

        clock = MockClock(1000.0)
        pipe = MockPipeline(clock)
        for _ in range(3):
            rec.record_frame(pipe)
            clock.advance(1.0)
        rec.stop()

        state_buf = StateBuffer()
        ctrl = ReplayController(state_buffer=state_buf)
        ctrl.load(rec.buffer)

        ctrl.step_forward()
        snap = state_buf.get_snapshot()
        assert snap is not None
        assert snap.system_status.get("replay_mode") is True


# ---------------------------------------------------------------------------
# Config from OmegaConf
# ---------------------------------------------------------------------------


class TestHistoryOmegaconfIntegration:
    def test_from_default_yaml(self):
        """Load the real default.yaml and verify history config."""
        import os
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "default.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("default.yaml not found")
        raw = OmegaConf.load(yaml_path)
        hist_cfg = raw.sentinel.get("history", {})
        cfg = HistoryConfig.from_omegaconf(hist_cfg)
        assert cfg.enabled is False
        assert cfg.max_frames == 10000

    def test_custom_config(self):
        raw = OmegaConf.create({
            "enabled": True,
            "max_frames": 500,
            "capture_interval": 2,
            "storage": {"format": "json", "compression": True},
            "replay": {"default_speed": 2.0, "loop": True},
        })
        cfg = HistoryConfig.from_omegaconf(raw)
        assert cfg.enabled is True
        assert cfg.max_frames == 500
        assert cfg.capture_interval == 2
        assert cfg.storage_format == "json"
        assert cfg.compression is True
        assert cfg.default_speed == 2.0
        assert cfg.loop is True


# ---------------------------------------------------------------------------
# File import to replay workflow
# ---------------------------------------------------------------------------


class TestFileImportReplay:
    def test_import_and_replay(self, tmp_path):
        """Save to file, then load into ReplayController."""
        buf = HistoryBuffer(max_frames=50)
        for i in range(10):
            buf.record(HistoryFrame(
                frame_number=i,
                timestamp=1000.0 + i,
                elapsed=float(i),
                camera_tracks=[{"id": f"T{i}"}],
            ))

        path = tmp_path / "session.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        state_buf = StateBuffer()
        ctrl = ReplayController(state_buffer=state_buf)
        ctrl.load(loaded)

        assert ctrl.get_status()["total_frames"] == 10
        ctrl.step_forward()
        assert ctrl.get_status()["current_index"] == 1

    def test_metadata_has_correct_info(self, tmp_path):
        buf = HistoryBuffer(max_frames=50)
        for i in range(5):
            buf.record(HistoryFrame(
                frame_number=i,
                timestamp=100.0 + i * 0.5,
                elapsed=i * 0.5,
            ))

        path = tmp_path / "meta.json"
        HistoryStorage.save(buf, path, fmt="json", config_snapshot={"version": "22"})
        meta = HistoryStorage.get_metadata(path)
        assert meta["frame_count"] == 5
        assert meta["config_snapshot"]["version"] == "22"
        assert len(meta["time_range"]) == 2
