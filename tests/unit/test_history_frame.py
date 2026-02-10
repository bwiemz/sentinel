"""Tests for sentinel.history.frame — HistoryFrame dataclass."""

from __future__ import annotations

import pytest

from sentinel.history.frame import HistoryFrame


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHistoryFrameDefaults:
    """Default construction produces valid empty frame."""

    def test_minimal_construction(self):
        f = HistoryFrame(frame_number=0, timestamp=1000.0, elapsed=0.0)
        assert f.frame_number == 0
        assert f.timestamp == 1000.0
        assert f.elapsed == 0.0

    def test_default_lists_empty(self):
        f = HistoryFrame(frame_number=0, timestamp=0.0, elapsed=0.0)
        assert f.camera_tracks == []
        assert f.radar_tracks == []
        assert f.thermal_tracks == []
        assert f.fused_tracks == []
        assert f.enhanced_fused_tracks == []
        assert f.system_status == {}

    def test_with_data(self):
        f = HistoryFrame(
            frame_number=42,
            timestamp=1000.5,
            elapsed=0.5,
            camera_tracks=[{"track_id": "A"}],
            radar_tracks=[{"track_id": "B"}, {"track_id": "C"}],
            system_status={"fps": 30},
        )
        assert f.frame_number == 42
        assert len(f.camera_tracks) == 1
        assert len(f.radar_tracks) == 2
        assert f.system_status["fps"] == 30


class TestHistoryFrameFrozen:
    """Frozen dataclass prevents accidental mutation."""

    def test_cannot_set_frame_number(self):
        f = HistoryFrame(frame_number=1, timestamp=0.0, elapsed=0.0)
        with pytest.raises(AttributeError):
            f.frame_number = 99  # type: ignore[misc]

    def test_cannot_set_timestamp(self):
        f = HistoryFrame(frame_number=1, timestamp=0.0, elapsed=0.0)
        with pytest.raises(AttributeError):
            f.timestamp = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestHistoryFrameToDict:
    """to_dict produces a plain dict with all fields."""

    def test_roundtrip(self):
        f = HistoryFrame(
            frame_number=10,
            timestamp=1000.0,
            elapsed=5.0,
            camera_tracks=[{"id": "c1"}],
            radar_tracks=[{"id": "r1"}],
            thermal_tracks=[{"id": "t1"}],
            fused_tracks=[{"id": "f1"}],
            enhanced_fused_tracks=[{"id": "e1"}],
            system_status={"fps": 30, "uptime": 5.0},
        )
        d = f.to_dict()
        f2 = HistoryFrame.from_dict(d)
        assert f2.frame_number == f.frame_number
        assert f2.timestamp == f.timestamp
        assert f2.elapsed == f.elapsed
        assert f2.camera_tracks == f.camera_tracks
        assert f2.radar_tracks == f.radar_tracks
        assert f2.thermal_tracks == f.thermal_tracks
        assert f2.fused_tracks == f.fused_tracks
        assert f2.enhanced_fused_tracks == f.enhanced_fused_tracks
        assert f2.system_status == f.system_status

    def test_to_dict_keys(self):
        f = HistoryFrame(frame_number=0, timestamp=0.0, elapsed=0.0)
        d = f.to_dict()
        expected_keys = {
            "frame_number",
            "timestamp",
            "elapsed",
            "camera_tracks",
            "radar_tracks",
            "thermal_tracks",
            "fused_tracks",
            "enhanced_fused_tracks",
            "system_status",
        }
        assert set(d.keys()) == expected_keys


class TestHistoryFrameFromDict:
    """from_dict handles missing fields gracefully."""

    def test_missing_elapsed(self):
        d = {"frame_number": 5, "timestamp": 100.0}
        f = HistoryFrame.from_dict(d)
        assert f.elapsed == 0.0

    def test_missing_track_lists(self):
        d = {"frame_number": 1, "timestamp": 1.0}
        f = HistoryFrame.from_dict(d)
        assert f.camera_tracks == []
        assert f.radar_tracks == []
        assert f.thermal_tracks == []
        assert f.fused_tracks == []
        assert f.enhanced_fused_tracks == []
        assert f.system_status == {}

    def test_extra_keys_ignored(self):
        d = {
            "frame_number": 1,
            "timestamp": 1.0,
            "elapsed": 0.0,
            "unknown_field": "foo",
        }
        f = HistoryFrame.from_dict(d)
        assert f.frame_number == 1


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestHistoryFrameProperties:
    """track_count and estimated_size_bytes."""

    def test_track_count_empty(self):
        f = HistoryFrame(frame_number=0, timestamp=0.0, elapsed=0.0)
        assert f.track_count == 0

    def test_track_count_with_data(self):
        f = HistoryFrame(
            frame_number=0,
            timestamp=0.0,
            elapsed=0.0,
            camera_tracks=[{"id": "a"}],
            radar_tracks=[{"id": "b"}, {"id": "c"}],
            fused_tracks=[{"id": "d"}],
        )
        assert f.track_count == 4

    def test_estimated_size_empty(self):
        f = HistoryFrame(frame_number=0, timestamp=0.0, elapsed=0.0)
        assert f.estimated_size_bytes == 500  # base overhead

    def test_estimated_size_with_tracks(self):
        f = HistoryFrame(
            frame_number=0,
            timestamp=0.0,
            elapsed=0.0,
            camera_tracks=[{"id": "a"}] * 10,
        )
        assert f.estimated_size_bytes == 10 * 200 + 500

    def test_all_track_types_counted(self):
        f = HistoryFrame(
            frame_number=0,
            timestamp=0.0,
            elapsed=0.0,
            camera_tracks=[{}],
            radar_tracks=[{}],
            thermal_tracks=[{}],
            fused_tracks=[{}],
            enhanced_fused_tracks=[{}],
        )
        assert f.track_count == 5
