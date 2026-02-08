"""Main pipeline orchestrator -- connects all SENTINEL subsystems."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from sentinel.core.bus import EventBus
from sentinel.core.clock import FrameTimer, SystemClock
from sentinel.core.types import Detection
from sentinel.sensors.camera import CameraAdapter
from sentinel.detection.yolo import YOLODetector
from sentinel.tracking.track import Track
from sentinel.tracking.track_manager import TrackManager
from sentinel.ui.hud.renderer import HUDRenderer

logger = logging.getLogger(__name__)


class SentinelPipeline:
    """Central pipeline: sensor read -> detect -> track -> render.

    Full pipeline: Camera -> YOLO Detection -> Kalman Tracking -> HUD Overlay.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self._clock = SystemClock()
        self._timer = FrameTimer(window_size=60)
        self._bus = EventBus()
        self._running = False

        # Initialize sensors
        cam_cfg = config.sentinel.sensors.camera
        self._camera = CameraAdapter(
            source=cam_cfg.source,
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
            buffer_size=cam_cfg.get("buffer_size", 1),
            backend=cam_cfg.get("backend", "auto"),
        )

        # Initialize detector
        det_cfg = config.sentinel.detection
        self._detector = YOLODetector(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            confidence=det_cfg.get("confidence", 0.25),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=det_cfg.get("device", "auto"),
            classes=det_cfg.get("classes", None),
            max_detections=det_cfg.get("max_detections", 100),
            image_size=det_cfg.get("image_size", 640),
        )

        # Initialize tracker
        self._track_manager = TrackManager(config.sentinel.tracking)

        # Initialize HUD
        hud_cfg = config.sentinel.ui.hud
        self._hud_enabled = hud_cfg.get("enabled", True)
        self._hud = HUDRenderer(hud_cfg) if self._hud_enabled else None

        # State
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: list[Detection] = []
        self._latest_tracks: list[Track] = []

    def run(self) -> None:
        """Main loop: read -> detect -> track -> render -> display."""
        logger.info("=" * 60)
        logger.info("  SENTINEL Tracking System v%s", self._config.sentinel.system.version)
        logger.info("  Initializing subsystems...")
        logger.info("=" * 60)

        if not self._camera.connect():
            logger.error("Failed to connect camera. Aborting.")
            return

        logger.info("Warming up detector...")
        self._detector.warmup()
        logger.info("Tracker ready: max %d tracks", self._config.sentinel.tracking.track_management.max_tracks)

        self._running = True
        logger.info("Pipeline running. Press 'q' to quit.")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        while self._running:
            # 1. Read sensor frame
            frame_data = self._camera.read_frame()
            if frame_data is None:
                if not self._camera.is_connected:
                    logger.warning("Camera disconnected.")
                    break
                continue

            self._timer.tick()
            frame = frame_data.data
            timestamp = frame_data.timestamp

            # 2. Detect
            detections = self._detector.detect(frame, timestamp)
            self._latest_detections = detections

            # 3. Track
            tracks = self._track_manager.step(detections)
            self._latest_tracks = tracks

            # 4. Render HUD
            if self._hud is not None:
                status = self.get_system_status()
                display = self._hud.render(frame, tracks, detections, status)
            else:
                display = frame

            self._latest_frame = display

            # 5. Display
            if self._config.sentinel.ui.hud.get("display", True):
                cv2.imshow("SENTINEL", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested.")
                    self._running = False

    def _shutdown(self) -> None:
        logger.info("Shutting down pipeline...")
        self._running = False
        self._camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Final stats:")
        logger.info("  Total tracks created: %d", len(self._track_manager._tracks) + self._track_manager.track_count)
        logger.info("  Active tracks at shutdown: %d", self._track_manager.track_count)

    def get_system_status(self) -> dict:
        return {
            "fps": self._timer.fps,
            "detection_count": len(self._latest_detections),
            "track_count": self._track_manager.track_count,
            "confirmed_count": len(self._track_manager.confirmed_tracks),
            "camera_connected": self._camera.is_connected,
            "uptime": self._clock.elapsed(),
        }

    def get_track_snapshot(self) -> list[Track]:
        return self._latest_tracks

    def get_latest_hud_frame(self) -> Optional[np.ndarray]:
        return self._latest_frame
