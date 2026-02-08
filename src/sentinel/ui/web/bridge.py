"""WebDashboard lifecycle manager — wires pipeline state to the web server."""

from __future__ import annotations

import logging
import threading
import time

from omegaconf import DictConfig

from sentinel.ui.web.mjpeg import encode_frame_jpeg
from sentinel.ui.web.server import create_app
from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot

logger = logging.getLogger(__name__)


class WebDashboard:
    """Manages the web dashboard alongside the pipeline.

    Usage::

        dashboard = WebDashboard(config)
        dashboard.start()                 # uvicorn in a daemon thread
        ...
        dashboard.publish(pipeline)       # each pipeline frame
        ...
        dashboard.stop()

    Args:
        config: ``sentinel.ui.web`` config section (DictConfig or dict).
    """

    def __init__(self, config: DictConfig | dict) -> None:
        cfg = config if isinstance(config, dict) else dict(config)
        self._host: str = cfg.get("host", "0.0.0.0")
        self._port: int = int(cfg.get("port", 8080))
        self._update_hz: int = int(cfg.get("track_update_hz", 10))
        self._video_fps: int = int(cfg.get("video_stream_fps", 15))
        self._jpeg_quality: int = int(cfg.get("jpeg_quality", 80))

        self._buffer = StateBuffer()
        self._app = create_app(self._buffer, self._update_hz, self._video_fps)
        self._server_thread: threading.Thread | None = None
        self._server = None  # uvicorn.Server

        # Rate limit: don't publish faster than update_hz
        self._min_interval = 1.0 / max(self._update_hz, 1)
        self._last_publish: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the web server in a daemon thread."""
        import uvicorn

        uvi_config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(uvi_config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="sentinel-web",
            daemon=True,
        )
        self._server_thread.start()
        logger.info("Web dashboard started at http://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Signal the server to shut down."""
        if self._server is not None:
            self._server.should_exit = True
            logger.info("Web dashboard shutdown requested")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, pipeline) -> None:
        """Capture pipeline state and push to the web buffer.

        Called from the pipeline main thread.  Rate-limited to
        ``track_update_hz``.  All data is serialized to plain dicts/bytes
        before entering the buffer.
        """
        now = time.monotonic()
        if (now - self._last_publish) < self._min_interval:
            return
        self._last_publish = now

        snapshot = StateSnapshot(
            timestamp=time.time(),
            system_status=pipeline.get_system_status(),
            camera_tracks=self._serialize_camera(pipeline),
            radar_tracks=self._serialize_radar(pipeline),
            thermal_tracks=self._serialize_thermal(pipeline),
            fused_tracks=self._serialize_fused(pipeline),
            enhanced_fused_tracks=self._serialize_enhanced_fused(pipeline),
            hud_frame_jpeg=self._encode_hud(pipeline),
        )
        self._buffer.update(snapshot)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_camera(pipeline) -> list[dict]:
        return [t.to_dict() for t in pipeline.get_track_snapshot()]

    @staticmethod
    def _serialize_radar(pipeline) -> list[dict]:
        mgr = getattr(pipeline, "_radar_track_manager", None)
        if mgr is None:
            return []
        return [t.to_dict() for t in mgr.active_tracks if t.is_alive]

    @staticmethod
    def _serialize_thermal(pipeline) -> list[dict]:
        mgr = getattr(pipeline, "_thermal_track_manager", None)
        if mgr is None:
            return []
        return [t.to_dict() for t in mgr.active_tracks if t.is_alive]

    @staticmethod
    def _serialize_fused(pipeline) -> list[dict]:
        fused = getattr(pipeline, "_latest_fused_tracks", None)
        if not fused:
            return []
        return [ft.to_dict() for ft in fused]

    @staticmethod
    def _serialize_enhanced_fused(pipeline) -> list[dict]:
        enhanced = getattr(pipeline, "_latest_enhanced_fused", None)
        if not enhanced:
            return []
        return [eft.to_dict() for eft in enhanced]

    def _encode_hud(self, pipeline) -> bytes | None:
        frame = pipeline.get_latest_hud_frame()
        if frame is None:
            return None
        try:
            return encode_frame_jpeg(frame, quality=self._jpeg_quality)
        except Exception:
            logger.debug("HUD frame encoding failed", exc_info=True)
            return None
