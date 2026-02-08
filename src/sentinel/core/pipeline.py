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

logger = logging.getLogger(__name__)


class SentinelPipeline:
    """Central pipeline: sensor read -> detect -> (track -> fuse ->) render.

    Phase 1: Camera + YOLO detection with basic annotated output.
    Tracking and fusion hooks are stubbed for Phase 2+.
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

        # State
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: list[Detection] = []

    def run(self) -> None:
        """Main loop: read -> detect -> annotate -> display."""
        logger.info("=" * 60)
        logger.info("  SENTINEL Tracking System v%s", self._config.sentinel.system.version)
        logger.info("  Initializing subsystems...")
        logger.info("=" * 60)

        if not self._camera.connect():
            logger.error("Failed to connect camera. Aborting.")
            return

        logger.info("Warming up detector...")
        self._detector.warmup()

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

            # 3. Annotate frame (Phase 1 simple rendering)
            annotated = self._annotate_frame(frame, detections)
            self._latest_frame = annotated

            # 4. Display
            cv2.imshow("SENTINEL", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                self._running = False

    def _annotate_frame(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detection results on frame. Simple Phase 1 rendering."""
        display = frame.copy()

        for det in detections:
            if det.bbox is None:
                continue

            x1, y1, x2, y2 = det.bbox.astype(int)
            color = (0, 255, 0)  # Green

            # Bounding box with corner brackets (mil-spec style)
            bracket_len = max(15, int(min(x2 - x1, y2 - y1) * 0.2))
            thickness = 1

            # Top-left corner
            cv2.line(display, (x1, y1), (x1 + bracket_len, y1), color, thickness, cv2.LINE_AA)
            cv2.line(display, (x1, y1), (x1, y1 + bracket_len), color, thickness, cv2.LINE_AA)
            # Top-right corner
            cv2.line(display, (x2, y1), (x2 - bracket_len, y1), color, thickness, cv2.LINE_AA)
            cv2.line(display, (x2, y1), (x2, y1 + bracket_len), color, thickness, cv2.LINE_AA)
            # Bottom-left corner
            cv2.line(display, (x1, y2), (x1 + bracket_len, y2), color, thickness, cv2.LINE_AA)
            cv2.line(display, (x1, y2), (x1, y2 - bracket_len), color, thickness, cv2.LINE_AA)
            # Bottom-right corner
            cv2.line(display, (x2, y2), (x2 - bracket_len, y2), color, thickness, cv2.LINE_AA)
            cv2.line(display, (x2, y2), (x2, y2 - bracket_len), color, thickness, cv2.LINE_AA)

            # Label
            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 40, 0), -1)
            cv2.putText(
                display, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        # Status bar
        fps_text = f"FPS: {self._timer.fps:.1f}"
        det_text = f"DETECTIONS: {len(detections)}"
        h, w = display.shape[:2]

        # Top bar background
        cv2.rectangle(display, (0, 0), (w, 28), (0, 20, 0), -1)
        cv2.putText(
            display, "SENTINEL v0.1", (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display, fps_text, (w - 120, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display, det_text, (w // 2 - 60, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # Border frame
        cv2.rectangle(display, (1, 1), (w - 2, h - 2), (0, 100, 0), 1)

        return display

    def _shutdown(self) -> None:
        logger.info("Shutting down pipeline...")
        self._running = False
        self._camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def get_system_status(self) -> dict:
        return {
            "fps": self._timer.fps,
            "detection_count": len(self._latest_detections),
            "camera_connected": self._camera.is_connected,
            "uptime": self._clock.elapsed(),
        }
