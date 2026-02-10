"""FastAPI application for the SENTINEL web dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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

    # ------------------------------------------------------------------
    # History & Replay endpoints (Phase 22)
    # ------------------------------------------------------------------

    @app.get("/api/history/status")
    async def history_status():
        recorder = getattr(app.state, "history_recorder", None)
        if recorder is None:
            return JSONResponse(content={"enabled": False})
        return JSONResponse(content={"enabled": True, **recorder.get_status()})

    @app.post("/api/history/start")
    async def history_start():
        recorder = getattr(app.state, "history_recorder", None)
        if recorder is None:
            return JSONResponse(content={"error": "History not enabled"}, status_code=400)
        recorder.start()
        return JSONResponse(content={"state": "recording"})

    @app.post("/api/history/stop")
    async def history_stop():
        recorder = getattr(app.state, "history_recorder", None)
        if recorder is None:
            return JSONResponse(content={"error": "History not enabled"}, status_code=400)
        recorder.stop()
        return JSONResponse(content={"state": "idle"})

    @app.post("/api/history/pause")
    async def history_pause():
        recorder = getattr(app.state, "history_recorder", None)
        if recorder is None:
            return JSONResponse(content={"error": "History not enabled"}, status_code=400)
        recorder.pause()
        return JSONResponse(content={"state": "paused"})

    @app.post("/api/history/export")
    async def history_export(request: Request):
        recorder = getattr(app.state, "history_recorder", None)
        if recorder is None:
            return JSONResponse(content={"error": "History not enabled"}, status_code=400)
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        filename = body.get("filename", f"recording_{int(time.time())}")

        from sentinel.history.storage import HistoryStorage

        storage_dir = Path("data/recordings")
        filepath = storage_dir / f"{filename}.json"
        HistoryStorage.save(recorder.buffer, filepath, fmt="json")
        return JSONResponse(content={
            "filepath": str(filepath),
            "frames": recorder.buffer.frame_count,
        })

    @app.post("/api/history/import")
    async def history_import(request: Request):
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        body = await request.json()
        filepath = body.get("filepath")
        if not filepath:
            return JSONResponse(content={"error": "filepath required"}, status_code=400)

        from sentinel.history.storage import HistoryStorage

        buf = HistoryStorage.load(filepath)
        controller.load(buf)
        return JSONResponse(content={
            "frames": buf.frame_count,
            "time_range": list(buf.time_range) if buf.time_range else None,
        })

    @app.post("/api/replay/play")
    async def replay_play():
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        controller.play()
        return JSONResponse(content=controller.get_status())

    @app.post("/api/replay/pause")
    async def replay_pause():
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        controller.pause()
        return JSONResponse(content=controller.get_status())

    @app.post("/api/replay/stop")
    async def replay_stop():
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        controller.stop()
        return JSONResponse(content=controller.get_status())

    @app.post("/api/replay/seek")
    async def replay_seek(request: Request):
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        body = await request.json()
        if "frame" in body:
            val = body["frame"]
            if val == "__step_back__":
                controller.step_backward()
            elif val == "__step_fwd__":
                controller.step_forward()
            else:
                controller.seek_to_frame(int(val))
        elif "time" in body:
            controller.seek_to_time(float(body["time"]))
        return JSONResponse(content=controller.get_status())

    @app.post("/api/replay/speed")
    async def replay_speed(request: Request):
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"error": "Replay not available"}, status_code=400)
        body = await request.json()
        controller.set_speed(float(body.get("speed", 1.0)))
        return JSONResponse(content=controller.get_status())

    @app.get("/api/replay/status")
    async def replay_status():
        controller = getattr(app.state, "replay_controller", None)
        if controller is None:
            return JSONResponse(content={"enabled": False})
        return JSONResponse(content={"enabled": True, **controller.get_status()})

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
