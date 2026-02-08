"""FastAPI application for the SENTINEL web dashboard."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sentinel.ui.web.state_buffer import StateBuffer

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    state_buffer: StateBuffer,
    update_hz: int = 10,
    video_fps: int = 15,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        state_buffer: Shared buffer written by the pipeline, read here.
        update_hz: WebSocket push rate in Hz.
        video_fps: MJPEG stream target FPS.
    """
    app = FastAPI(title="SENTINEL Dashboard", version="0.1.0")

    # Static files (CSS / JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # REST endpoints
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main dashboard page."""
        index = STATIC_DIR / "index.html"
        return HTMLResponse(content=index.read_text(encoding="utf-8"))

    @app.get("/api/status")
    async def get_status():
        snapshot = state_buffer.get_snapshot()
        return JSONResponse(content=snapshot.system_status)

    @app.get("/api/tracks")
    async def get_tracks():
        snapshot = state_buffer.get_snapshot()
        return JSONResponse(content={
            "timestamp": snapshot.timestamp,
            "camera": snapshot.camera_tracks,
            "radar": snapshot.radar_tracks,
            "thermal": snapshot.thermal_tracks,
            "fused": snapshot.fused_tracks,
            "enhanced_fused": snapshot.enhanced_fused_tracks,
        })

    @app.get("/api/config")
    async def get_config():
        return JSONResponse(content={
            "track_update_hz": update_hz,
            "video_stream_fps": video_fps,
        })

    # ------------------------------------------------------------------
    # MJPEG stream
    # ------------------------------------------------------------------

    @app.get("/api/video/hud")
    async def hud_stream():
        return StreamingResponse(
            _mjpeg_generator(state_buffer, video_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws/tracks")
    async def websocket_tracks(ws: WebSocket):
        await ws.accept()
        interval = 1.0 / max(update_hz, 1)
        logger.info("WebSocket client connected from %s", ws.client)
        try:
            while True:
                snapshot = state_buffer.get_snapshot()
                payload = {
                    "type": "update",
                    "timestamp": snapshot.timestamp,
                    "status": snapshot.system_status,
                    "tracks": {
                        "camera": snapshot.camera_tracks,
                        "radar": snapshot.radar_tracks,
                        "thermal": snapshot.thermal_tracks,
                        "fused": snapshot.fused_tracks,
                        "enhanced_fused": snapshot.enhanced_fused_tracks,
                    },
                }
                await ws.send_json(payload)
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception:
            logger.debug("WebSocket error", exc_info=True)

    return app


async def _mjpeg_generator(buffer: StateBuffer, fps: int):
    """Yield MJPEG frames from the state buffer."""
    interval = 1.0 / max(fps, 1)
    while True:
        # Run the blocking wait in a thread so we don't block the event loop
        frame_bytes = await asyncio.to_thread(buffer.wait_for_frame, 2.0)
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        await asyncio.sleep(interval)
