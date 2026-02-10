"""Tests for sentinel.history.storage — file export/import."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel.history.buffer import HistoryBuffer
from sentinel.history.frame import HistoryFrame
from sentinel.history.storage import HistoryStorage, HISTORY_FORMAT_VERSION


def _frame(n: int, ts: float, tracks: int = 0) -> HistoryFrame:
    return HistoryFrame(
        frame_number=n,
        timestamp=ts,
        elapsed=ts - 1000.0,
        camera_tracks=[{"track_id": f"C{i}", "score": 0.8} for i in range(tracks)],
        system_status={"fps": 30},
    )


def _populated_buffer(n: int = 5) -> HistoryBuffer:
    b = HistoryBuffer(max_frames=100)
    for i in range(n):
        b.record(_frame(i, 1000.0 + i * 0.1, tracks=2))
    return b


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


class TestHistoryStorageJSON:
    def test_save_load_roundtrip(self, tmp_path):
        buf = _populated_buffer(5)
        path = tmp_path / "test.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 5
        f = loaded.get_frame(0)
        assert f is not None
        assert f.frame_number == 0
        assert f.timestamp == 1000.0

    def test_frame_data_integrity(self, tmp_path):
        buf = _populated_buffer(3)
        path = tmp_path / "data.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        for i in range(3):
            orig = buf.get_frame(i)
            loaded_f = loaded.get_frame(i)
            assert loaded_f.frame_number == orig.frame_number
            assert loaded_f.timestamp == orig.timestamp
            assert loaded_f.camera_tracks == orig.camera_tracks
            assert loaded_f.system_status == orig.system_status

    def test_metadata(self, tmp_path):
        buf = _populated_buffer(5)
        path = tmp_path / "meta.json"
        HistoryStorage.save(buf, path, fmt="json", config_snapshot={"foo": "bar"})

        meta = HistoryStorage.get_metadata(path)
        assert meta["frame_count"] == 5
        assert meta["config_snapshot"]["foo"] == "bar"
        assert len(meta["time_range"]) == 2


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class TestHistoryStorageCompression:
    def test_gzip_roundtrip(self, tmp_path):
        buf = _populated_buffer(10)
        path = tmp_path / "compressed.json.gz"
        HistoryStorage.save(buf, path, fmt="json", compression=True)

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 10

    def test_compressed_smaller(self, tmp_path):
        buf = _populated_buffer(50)
        path_raw = tmp_path / "raw.json"
        path_gz = tmp_path / "compressed.json.gz"
        HistoryStorage.save(buf, path_raw, fmt="json", compression=False)
        HistoryStorage.save(buf, path_gz, fmt="json", compression=True)

        assert path_gz.stat().st_size < path_raw.stat().st_size


# ---------------------------------------------------------------------------
# Msgpack (if available)
# ---------------------------------------------------------------------------


class TestHistoryStorageMsgpack:
    def test_msgpack_roundtrip(self, tmp_path):
        buf = _populated_buffer(5)
        path = tmp_path / "test.msgpack"
        HistoryStorage.save(buf, path, fmt="msgpack")

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 5

    def test_msgpack_compressed(self, tmp_path):
        buf = _populated_buffer(5)
        path = tmp_path / "test.msgpack.gz"
        HistoryStorage.save(buf, path, fmt="msgpack", compression=True)

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestHistoryStorageEdgeCases:
    def test_empty_buffer(self, tmp_path):
        buf = HistoryBuffer(max_frames=10)
        path = tmp_path / "empty.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 0

    def test_single_frame(self, tmp_path):
        buf = HistoryBuffer(max_frames=10)
        buf.record(_frame(1, 1000.0))
        path = tmp_path / "single.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        assert loaded.frame_count == 1

    def test_max_frames_override(self, tmp_path):
        buf = _populated_buffer(10)
        path = tmp_path / "test.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path, max_frames=5)
        assert loaded.max_frames == 5
        # Should keep last 5 frames
        assert loaded.frame_count == 5

    def test_numpy_array_in_track_data(self, tmp_path):
        """Tracks containing numpy arrays survive serialization."""
        f = HistoryFrame(
            frame_number=0,
            timestamp=0.0,
            elapsed=0.0,
            camera_tracks=[{"position": np.array([1.0, 2.0, 3.0])}],
        )
        buf = HistoryBuffer(max_frames=10)
        buf.record(f)
        path = tmp_path / "numpy.json"
        HistoryStorage.save(buf, path, fmt="json")

        loaded = HistoryStorage.load(path)
        track = loaded.get_frame(0).camera_tracks[0]
        pos = track["position"]
        assert isinstance(pos, np.ndarray)
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])

    def test_format_version_in_file(self, tmp_path):
        buf = _populated_buffer(1)
        path = tmp_path / "ver.json"
        HistoryStorage.save(buf, path, fmt="json")

        raw = json.loads(path.read_text())
        assert raw["version"] == HISTORY_FORMAT_VERSION
