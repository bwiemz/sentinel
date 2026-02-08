"""Tests for the FastAPI web server."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.ui.web.server import create_app
from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot


@pytest.fixture
def state_buffer():
    buf = StateBuffer()
    buf.update(
        StateSnapshot(
            timestamp=1000.0,
            system_status={"fps": 30.0, "track_count": 2, "uptime": 60.0},
            camera_tracks=[{"track_id": "CAM01", "state": "confirmed"}],
            radar_tracks=[{"track_id": "RDR01", "range_m": 5000}],
            thermal_tracks=[{"track_id": "THM01", "azimuth_deg": 15.0}],
            fused_tracks=[{"fused_id": "FUS01"}],
            enhanced_fused_tracks=[{"fused_id": "EFT01", "threat_level": "HIGH"}],
        )
    )
    return buf


@pytest.fixture
def app(state_buffer):
    return create_app(state_buffer, update_hz=10, video_fps=15)


@pytest.fixture
def transport(app):
    return ASGITransport(app=app)


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------
class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_returns_html(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "SENTINEL" in resp.text


class TestApiStatus:
    @pytest.mark.asyncio
    async def test_returns_json(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fps"] == 30.0
        assert data["track_count"] == 2

    @pytest.mark.asyncio
    async def test_reflects_buffer_update(self, transport, state_buffer):
        state_buffer.update(StateSnapshot(system_status={"fps": 60.0}))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")
        assert resp.json()["fps"] == 60.0


class TestApiTracks:
    @pytest.mark.asyncio
    async def test_returns_all_sensor_tracks(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/tracks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestamp"] == 1000.0
        assert len(data["camera"]) == 1
        assert len(data["radar"]) == 1
        assert len(data["thermal"]) == 1
        assert len(data["fused"]) == 1
        assert len(data["enhanced_fused"]) == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_tracks(self, transport, state_buffer):
        state_buffer.update(StateSnapshot(timestamp=2.0))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/tracks")
        data = resp.json()
        assert data["camera"] == []
        assert data["radar"] == []


class TestApiConfig:
    @pytest.mark.asyncio
    async def test_returns_rates(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["track_update_hz"] == 10
        assert data["video_stream_fps"] == 15


class TestStaticFiles:
    @pytest.mark.asyncio
    async def test_css_served(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/static/css/dashboard.css")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_js_served(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/static/js/main.js")
        assert resp.status_code == 200


# ------------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------------
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_connect_and_receive(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("GET", "/ws/tracks") as _:
                pass  # Just verify WebSocket endpoint exists

    @pytest.mark.asyncio
    async def test_websocket_message_structure(self, app):
        """Connect via the ASGI interface and verify message shape."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/ws/tracks") as ws:
            data = ws.receive_json()
            assert data["type"] == "update"
            assert "timestamp" in data
            assert "status" in data
            assert "tracks" in data
            tracks = data["tracks"]
            assert "camera" in tracks
            assert "radar" in tracks
            assert "thermal" in tracks
            assert "fused" in tracks
            assert "enhanced_fused" in tracks

    @pytest.mark.asyncio
    async def test_websocket_tracks_contain_data(self, app, state_buffer):
        from starlette.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/ws/tracks") as ws:
            data = ws.receive_json()
            assert data["tracks"]["camera"][0]["track_id"] == "CAM01"
            assert data["tracks"]["radar"][0]["track_id"] == "RDR01"
            assert data["status"]["fps"] == 30.0
