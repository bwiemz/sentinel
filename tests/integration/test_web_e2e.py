"""End-to-end integration tests for the web dashboard."""

from __future__ import annotations

import time

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from sentinel.ui.web.bridge import WebDashboard
from sentinel.ui.web.server import create_app
from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _populated_buffer() -> StateBuffer:
    buf = StateBuffer()
    buf.update(
        StateSnapshot(
            timestamp=time.time(),
            system_status={
                "fps": 29.5,
                "track_count": 3,
                "confirmed_count": 2,
                "detection_count": 5,
                "uptime": 120.0,
                "camera_connected": True,
                "radar_track_count": 2,
                "thermal_track_count": 1,
                "fused_track_count": 2,
                "sensor_health": {
                    "camera": {"enabled": True, "error_count": 0},
                    "radar": {"enabled": True, "error_count": 0},
                    "thermal": {"enabled": True, "error_count": 0},
                },
                "detect_ms": 8.5,
                "track_ms": 1.2,
                "radar_ms": 0.9,
                "fusion_ms": 0.4,
                "render_ms": 2.7,
                "threat_counts": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 1},
            },
            camera_tracks=[
                {"track_id": "CAM-001", "state": "confirmed", "position": [640, 360], "score": 0.95},
                {"track_id": "CAM-002", "state": "tentative", "position": [200, 100], "score": 0.40},
            ],
            radar_tracks=[
                {"track_id": "RDR-001", "state": "confirmed", "range_m": 8000, "azimuth_deg": 15.0, "score": 0.88},
                {"track_id": "RDR-002", "state": "confirmed", "range_m": 5000, "azimuth_deg": -10.0, "score": 0.72},
            ],
            thermal_tracks=[
                {"track_id": "THM-001", "state": "confirmed", "azimuth_deg": 14.5, "temperature_k": 1800.0},
            ],
            fused_tracks=[],
            enhanced_fused_tracks=[
                {
                    "fused_id": "EFT-001",
                    "sensor_count": 3,
                    "range_m": 8000,
                    "azimuth_deg": 15.0,
                    "velocity_mps": 150.0,
                    "threat_level": "CRITICAL",
                    "is_stealth_candidate": False,
                    "is_hypersonic_candidate": True,
                    "fusion_quality": 0.9,
                },
                {
                    "fused_id": "EFT-002",
                    "sensor_count": 2,
                    "range_m": 5000,
                    "azimuth_deg": -10.0,
                    "velocity_mps": 50.0,
                    "threat_level": "LOW",
                    "is_stealth_candidate": False,
                    "is_hypersonic_candidate": False,
                    "fusion_quality": 0.7,
                },
            ],
            hud_frame_jpeg=b"\xff\xd8\xff\xe0FAKE_JPEG_DATA",
        )
    )
    return buf


def _app_and_buffer():
    buf = _populated_buffer()
    app = create_app(buf, update_hz=10, video_fps=15)
    return app, buf


# ---------------------------------------------------------------------------
# WebSocket E2E
# ---------------------------------------------------------------------------
class TestWebSocketE2E:
    def test_full_websocket_flow(self):
        """Connect, receive a message, verify structure, disconnect."""
        app, buf = _app_and_buffer()
        client = TestClient(app)
        with client.websocket_connect("/ws/tracks") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "update"
            assert msg["timestamp"] > 0
            assert msg["status"]["fps"] == 29.5
            assert len(msg["tracks"]["camera"]) == 2
            assert len(msg["tracks"]["radar"]) == 2
            assert len(msg["tracks"]["thermal"]) == 1
            assert len(msg["tracks"]["enhanced_fused"]) == 2

    def test_multiple_messages(self):
        """Client receives multiple sequential updates."""
        app, buf = _app_and_buffer()
        client = TestClient(app)
        with client.websocket_connect("/ws/tracks") as ws:
            msg1 = ws.receive_json()
            msg2 = ws.receive_json()
            assert msg1["type"] == "update"
            assert msg2["type"] == "update"

    def test_buffer_update_reflected(self):
        """Updating the buffer changes what WebSocket sends."""
        app, buf = _app_and_buffer()
        client = TestClient(app)
        with client.websocket_connect("/ws/tracks") as ws:
            ws.receive_json()  # initial
            # Update buffer
            buf.update(StateSnapshot(
                timestamp=time.time(),
                system_status={"fps": 60.0},
                camera_tracks=[{"track_id": "NEW"}],
            ))
            msg = ws.receive_json()
            assert msg["status"]["fps"] == 60.0
            assert msg["tracks"]["camera"][0]["track_id"] == "NEW"


# ---------------------------------------------------------------------------
# REST E2E
# ---------------------------------------------------------------------------
class TestRestE2E:
    @pytest.mark.asyncio
    async def test_full_data_flow(self):
        """All REST endpoints return consistent data from the same buffer."""
        app, buf = _app_and_buffer()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Status
            status_resp = await client.get("/api/status")
            assert status_resp.json()["fps"] == 29.5

            # Tracks
            tracks_resp = await client.get("/api/tracks")
            data = tracks_resp.json()
            assert len(data["camera"]) == 2
            assert data["enhanced_fused"][0]["threat_level"] == "CRITICAL"

            # Config
            cfg_resp = await client.get("/api/config")
            assert cfg_resp.json()["track_update_hz"] == 10

            # Root HTML
            root_resp = await client.get("/")
            assert "SENTINEL" in root_resp.text

    @pytest.mark.asyncio
    async def test_threat_data_in_tracks(self):
        """Enhanced fused tracks include threat classification fields."""
        app, buf = _app_and_buffer()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/tracks")
            eft = resp.json()["enhanced_fused"]
            critical = [t for t in eft if t["threat_level"] == "CRITICAL"]
            assert len(critical) == 1
            assert critical[0]["is_hypersonic_candidate"] is True
            assert critical[0]["sensor_count"] == 3


# ---------------------------------------------------------------------------
# MJPEG E2E
# ---------------------------------------------------------------------------
class TestMjpegE2E:
    def test_mjpeg_generator_yields_frame(self):
        """The MJPEG generator yields properly formatted multipart frames."""
        import asyncio
        from sentinel.ui.web.server import _mjpeg_generator

        buf = StateBuffer()
        buf.update(StateSnapshot(hud_frame_jpeg=b"\xff\xd8FAKE_JPEG"))

        async def get_first_frame():
            gen = _mjpeg_generator(buf, fps=15)
            return await gen.__anext__()

        frame = asyncio.run(get_first_frame())
        assert b"--frame" in frame
        assert b"Content-Type: image/jpeg" in frame
        assert b"\xff\xd8FAKE_JPEG" in frame


# ---------------------------------------------------------------------------
# Dashboard disabled by default
# ---------------------------------------------------------------------------
class TestDashboardDisabledByDefault:
    def test_web_dashboard_not_created_when_disabled(self):
        """With enabled: false (default), no dashboard is created."""
        from tests.integration.conftest import _minimal_pipeline_config
        from sentinel.core.pipeline import SentinelPipeline

        cfg = _minimal_pipeline_config()
        # Ensure web is disabled (default)
        assert cfg.sentinel.ui.get("web", {}).get("enabled", False) is False
        pipeline = SentinelPipeline(cfg)
        assert pipeline._web_dashboard is None


# ---------------------------------------------------------------------------
# Bridge publish E2E
# ---------------------------------------------------------------------------
class TestBridgePublishE2E:
    def test_publish_populates_buffer_for_server(self):
        """publish() on a mock pipeline → server returns that data."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        dashboard = WebDashboard({
            "host": "127.0.0.1",
            "port": 0,
            "track_update_hz": 1000,
            "video_stream_fps": 15,
        })

        # Mock pipeline
        pipeline = MagicMock()
        pipeline.get_system_status.return_value = {"fps": 25.0}
        pipeline.get_track_snapshot.return_value = []
        pipeline._radar_track_manager = None
        pipeline._thermal_track_manager = None
        pipeline._latest_fused_tracks = []
        pipeline._latest_enhanced_fused = []
        pipeline.get_latest_hud_frame.return_value = None

        dashboard.publish(pipeline)

        # Now query the server
        client = TestClient(dashboard._app)
        resp = client.get("/api/status")
        assert resp.json()["fps"] == 25.0
