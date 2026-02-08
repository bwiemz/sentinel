"""Tests for the WebDashboard bridge."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel.ui.web.bridge import WebDashboard


def _make_config(**overrides):
    defaults = {
        "host": "127.0.0.1",
        "port": 9999,
        "track_update_hz": 10,
        "video_stream_fps": 15,
        "jpeg_quality": 50,
    }
    defaults.update(overrides)
    return defaults


def _mock_track(track_id="T1", state="confirmed"):
    t = MagicMock()
    t.to_dict.return_value = {"track_id": track_id, "state": state}
    t.is_alive = True
    return t


def _mock_fused(fused_id="F1"):
    ft = MagicMock()
    ft.to_dict.return_value = {"fused_id": fused_id}
    return ft


def _mock_pipeline(
    camera_tracks=None,
    radar_tracks=None,
    thermal_tracks=None,
    fused_tracks=None,
    enhanced_fused=None,
    hud_frame=None,
):
    p = MagicMock()
    p.get_system_status.return_value = {"fps": 30.0, "track_count": 1}
    p.get_track_snapshot.return_value = camera_tracks or []
    p.get_latest_hud_frame.return_value = hud_frame

    if radar_tracks is not None:
        p._radar_track_manager = SimpleNamespace(active_tracks=radar_tracks)
    else:
        p._radar_track_manager = None

    if thermal_tracks is not None:
        p._thermal_track_manager = SimpleNamespace(active_tracks=thermal_tracks)
    else:
        p._thermal_track_manager = None

    p._latest_fused_tracks = fused_tracks or []
    p._latest_enhanced_fused = enhanced_fused or []
    return p


class TestWebDashboardInit:
    def test_reads_config(self):
        d = WebDashboard(_make_config(port=1234, track_update_hz=20))
        assert d._port == 1234
        assert d._update_hz == 20

    def test_default_values(self):
        d = WebDashboard({})
        assert d._host == "0.0.0.0"
        assert d._port == 8080
        assert d._update_hz == 10


class TestWebDashboardPublish:
    def test_serializes_camera_tracks(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(camera_tracks=[_mock_track("C1")])
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.camera_tracks == [{"track_id": "C1", "state": "confirmed"}]

    def test_serializes_radar_tracks(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(radar_tracks=[_mock_track("R1")])
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.radar_tracks == [{"track_id": "R1", "state": "confirmed"}]

    def test_serializes_thermal_tracks(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(thermal_tracks=[_mock_track("TH1")])
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.thermal_tracks == [{"track_id": "TH1", "state": "confirmed"}]

    def test_serializes_fused_tracks(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(fused_tracks=[_mock_fused("F1")])
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.fused_tracks == [{"fused_id": "F1"}]

    def test_serializes_enhanced_fused(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(enhanced_fused=[_mock_fused("E1")])
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.enhanced_fused_tracks == [{"fused_id": "E1"}]

    def test_encodes_hud_frame(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline = _mock_pipeline(hud_frame=frame)
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.hud_frame_jpeg is not None
        assert snap.hud_frame_jpeg[:2] == b"\xff\xd8"

    def test_handles_none_hud_frame(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline(hud_frame=None)
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.hud_frame_jpeg is None

    def test_handles_no_radar_manager(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline()
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.radar_tracks == []

    def test_handles_no_thermal_manager(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline()
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.thermal_tracks == []

    def test_rate_limiting(self):
        d = WebDashboard(_make_config(track_update_hz=2))
        pipeline = _mock_pipeline()
        d.publish(pipeline)
        pipeline.get_system_status.reset_mock()
        d.publish(pipeline)  # should be skipped (too fast)
        pipeline.get_system_status.assert_not_called()

    def test_sets_timestamp(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline()
        before = time.time()
        d.publish(pipeline)
        after = time.time()
        snap = d._buffer.get_snapshot()
        assert before <= snap.timestamp <= after

    def test_sets_system_status(self):
        d = WebDashboard(_make_config(track_update_hz=1000))
        pipeline = _mock_pipeline()
        d.publish(pipeline)
        snap = d._buffer.get_snapshot()
        assert snap.system_status["fps"] == 30.0


class TestWebDashboardLifecycle:
    def test_start_creates_daemon_thread(self):
        d = WebDashboard(_make_config(port=0))
        mock_server = MagicMock()
        mock_server.run = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": MagicMock()}) as _:
            import uvicorn as uvi_mock
            uvi_mock.Config.return_value = MagicMock()
            uvi_mock.Server.return_value = mock_server
            d.start()
            assert d._server_thread is not None
            assert d._server_thread.daemon is True
            assert d._server_thread.name == "sentinel-web"
            d.stop()

    def test_stop_sets_should_exit(self):
        d = WebDashboard(_make_config())
        d._server = MagicMock()
        d.stop()
        assert d._server.should_exit is True

    def test_stop_without_start_is_noop(self):
        d = WebDashboard(_make_config())
        d.stop()  # should not raise
