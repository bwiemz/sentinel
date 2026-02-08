"""Main pipeline orchestrator -- connects all SENTINEL subsystems.

Supports dual-sensor operation (camera + radar) with track-level fusion
when radar is enabled, or camera-only mode when disabled.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

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
    """Central pipeline: sensor read -> detect -> track -> fuse -> render.

    Camera + Radar -> YOLO + Radar Detection -> KF + EKF Tracking -> Fusion -> HUD.
    Radar components are only initialized when sentinel.sensors.radar.enabled is true.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self._clock = SystemClock()
        self._timer = FrameTimer(window_size=60)
        self._bus = EventBus()
        self._running = False

        # Initialize camera sensor
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

        # Initialize camera tracker
        self._track_manager = TrackManager(config.sentinel.tracking)

        # Initialize HUD
        hud_cfg = config.sentinel.ui.hud
        self._hud_enabled = hud_cfg.get("enabled", True)
        self._hud = HUDRenderer(hud_cfg) if self._hud_enabled else None

        # State
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: list[Detection] = []
        self._latest_tracks: list[Track] = []

        # --- Radar (Phase 4) ---
        self._radar_enabled = config.sentinel.sensors.radar.get("enabled", False)
        self._radar = None
        self._radar_track_manager = None
        self._track_fusion = None
        self._latest_radar_detections: list[Detection] = []
        self._latest_radar_tracks: list = []
        self._latest_fused_tracks: list = []

        if self._radar_enabled:
            self._init_radar(config)

    def _init_radar(self, config: DictConfig) -> None:
        """Initialize radar simulator, radar tracker, and fusion module."""
        from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator
        from sentinel.tracking.radar_track_manager import RadarTrackManager
        from sentinel.fusion.track_fusion import TrackFusion

        radar_cfg = config.sentinel.sensors.radar
        sim_config = RadarSimConfig.from_omegaconf(radar_cfg)
        self._radar = RadarSimulator(sim_config)

        # Build radar tracking config
        radar_tracking = config.sentinel.get("tracking", {}).get("radar", {})
        radar_track_cfg = OmegaConf.create({
            "filter": {
                "dt": radar_tracking.get("filter", {}).get("dt", 1.0 / radar_cfg.get("scan_rate_hz", 10)),
                "type": "ekf",
            },
            "association": {
                "gate_threshold": radar_tracking.get("association", {}).get("gate_threshold", 9.21),
            },
            "track_management": {
                "confirm_hits": radar_tracking.get("track_management", {}).get("confirm_hits", 3),
                "max_coast_frames": radar_tracking.get("track_management", {}).get("max_coast_frames", 5),
                "max_tracks": radar_tracking.get("track_management", {}).get("max_tracks", 50),
            },
        })
        self._radar_track_manager = RadarTrackManager(radar_track_cfg)

        # Fusion
        fusion_cfg = config.sentinel.get("fusion", {})
        cam_cfg = config.sentinel.sensors.camera
        self._track_fusion = TrackFusion(
            camera_hfov_deg=fusion_cfg.get("camera_hfov_deg", 60.0),
            image_width_px=cam_cfg.get("width", 1280),
            azimuth_gate_deg=fusion_cfg.get("azimuth_gate_deg", 5.0),
        )

        logger.info("Radar subsystem initialized (simulator mode, %.0f Hz)", radar_cfg.get("scan_rate_hz", 10))

    def run(self) -> None:
        """Main loop: read -> detect -> track -> fuse -> render -> display."""
        logger.info("=" * 60)
        logger.info("  SENTINEL Tracking System v%s", self._config.sentinel.system.version)
        logger.info("  Initializing subsystems...")
        logger.info("=" * 60)

        if not self._camera.connect():
            logger.error("Failed to connect camera. Aborting.")
            return

        if self._radar_enabled and self._radar is not None:
            self._radar.connect()
            logger.info("Radar simulator connected.")

        logger.info("Warming up detector...")
        self._detector.warmup()
        logger.info("Tracker ready: max %d tracks", self._config.sentinel.tracking.track_management.max_tracks)
        if self._radar_enabled:
            logger.info("Radar tracking + fusion enabled.")

        self._running = True
        logger.info("Pipeline running. Press 'q' to quit.")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        radar_interval = 1.0 / self._config.sentinel.sensors.radar.get("scan_rate_hz", 10)
        last_radar_time = 0.0

        while self._running:
            # 1. Read camera frame
            frame_data = self._camera.read_frame()
            if frame_data is None:
                if not self._camera.is_connected:
                    logger.warning("Camera disconnected.")
                    break
                continue

            self._timer.tick()
            frame = frame_data.data
            timestamp = frame_data.timestamp

            # 2. Camera detection + tracking
            detections = self._detector.detect(frame, timestamp)
            self._latest_detections = detections
            tracks = self._track_manager.step(detections)
            self._latest_tracks = tracks

            # 3. Radar scan (at radar rate, if enabled)
            if self._radar_enabled and self._radar is not None:
                if (timestamp - last_radar_time) >= radar_interval:
                    self._process_radar_scan()
                    last_radar_time = timestamp

                # 4. Fusion (every camera frame, using latest radar tracks)
                if self._track_fusion is not None:
                    self._latest_fused_tracks = self._track_fusion.fuse(
                        self._latest_tracks,
                        self._latest_radar_tracks,
                    )

            # 5. Render HUD
            if self._hud is not None:
                status = self.get_system_status()
                display = self._hud.render(
                    frame, tracks, detections, status,
                    radar_tracks=self._latest_radar_tracks if self._radar_enabled else None,
                    fused_tracks=self._latest_fused_tracks if self._radar_enabled else None,
                )
            else:
                display = frame

            self._latest_frame = display

            # 6. Display
            if self._config.sentinel.ui.hud.get("display", True):
                cv2.imshow("SENTINEL", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested.")
                    self._running = False

    def _process_radar_scan(self) -> None:
        """Read and process one radar scan."""
        from sentinel.sensors.radar_sim import radar_frame_to_detections

        radar_frame = self._radar.read_frame()
        if radar_frame is not None:
            radar_dets = radar_frame_to_detections(radar_frame)
            self._latest_radar_detections = radar_dets
            self._latest_radar_tracks = self._radar_track_manager.step(radar_dets)

    def _shutdown(self) -> None:
        logger.info("Shutting down pipeline...")
        self._running = False
        self._camera.disconnect()
        if self._radar is not None:
            self._radar.disconnect()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Final stats:")
        logger.info("  Camera tracks created: %d", len(self._track_manager._tracks) + self._track_manager.track_count)
        logger.info("  Active camera tracks: %d", self._track_manager.track_count)
        if self._radar_enabled and self._radar_track_manager is not None:
            logger.info("  Active radar tracks: %d", self._radar_track_manager.track_count)
            logger.info("  Fused tracks: %d", len(self._latest_fused_tracks))

    def get_system_status(self) -> dict:
        status = {
            "fps": self._timer.fps,
            "detection_count": len(self._latest_detections),
            "track_count": self._track_manager.track_count,
            "confirmed_count": len(self._track_manager.confirmed_tracks),
            "camera_connected": self._camera.is_connected,
            "uptime": self._clock.elapsed(),
        }
        if self._radar_enabled:
            status["radar_connected"] = self._radar.is_connected if self._radar else False
            status["radar_detection_count"] = len(self._latest_radar_detections)
            status["radar_track_count"] = self._radar_track_manager.track_count if self._radar_track_manager else 0
            status["fused_track_count"] = len(self._latest_fused_tracks)
        return status

    def get_track_snapshot(self) -> list[Track]:
        return self._latest_tracks

    def get_latest_hud_frame(self) -> Optional[np.ndarray]:
        return self._latest_frame
