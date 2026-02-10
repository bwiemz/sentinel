"""Main pipeline orchestrator -- connects all SENTINEL subsystems.

Supports dual-sensor operation (camera + radar) with track-level fusion
when radar is enabled, or camera-only mode when disabled.
Includes graceful degradation: individual sensor failures are isolated
and auto-disabled after consecutive errors without crashing the pipeline.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sentinel.core.bus import EventBus
from sentinel.core.clock import Clock, FrameTimer, SystemClock
from sentinel.core.types import Detection
from sentinel.detection.yolo import YOLODetector
from sentinel.sensors.camera import CameraAdapter
from sentinel.tracking.track import Track
from sentinel.tracking.track_manager import TrackManager
from sentinel.ui.hud.renderer import HUDRenderer
from sentinel.utils.geo_context import GeoContext

logger = logging.getLogger(__name__)


class _SensorHealth:
    """Tracks consecutive error counts for a sensor subsystem."""

    __slots__ = ("name", "error_count", "max_errors", "enabled")

    def __init__(self, name: str, max_errors: int, enabled: bool = True):
        self.name = name
        self.error_count = 0
        self.max_errors = max_errors
        self.enabled = enabled

    def record_success(self) -> None:
        self.error_count = 0

    def record_error(self) -> bool:
        """Record an error. Returns True if sensor was just disabled."""
        self.error_count += 1
        if self.error_count >= self.max_errors and self.enabled:
            self.enabled = False
            return True
        return False


class SentinelPipeline:
    """Central pipeline: sensor read -> detect -> track -> fuse -> render.

    Camera + Radar -> YOLO + Radar Detection -> KF + EKF Tracking -> Fusion -> HUD.
    Radar components are only initialized when sentinel.sensors.radar.enabled is true.
    """

    def __init__(self, config: DictConfig, clock: Clock | None = None):
        self._config = config
        self._clock = clock if clock is not None else SystemClock()
        self._timer = FrameTimer(window_size=60)
        self._bus = EventBus()
        self._running = False

        # Graceful degradation settings
        sys_cfg = config.sentinel.get("system", {})
        self._graceful_degradation = sys_cfg.get("graceful_degradation", True)
        max_errors = sys_cfg.get("max_sensor_errors", 10)

        # Timing instrumentation (ms per stage, exponential moving average)
        self._timing: dict[str, float] = {
            "detect_ms": 0.0,
            "track_ms": 0.0,
            "radar_ms": 0.0,
            "fusion_ms": 0.0,
            "render_ms": 0.0,
        }
        self._timing_alpha = 0.1  # EMA smoothing factor

        # Geodetic reference context (Phase 16)
        geo_cfg = config.sentinel.get("geo_reference", {})
        self._geo_context: GeoContext | None = GeoContext.from_config(geo_cfg)
        if self._geo_context is not None:
            logger.info(
                "Geodetic reference enabled: %s (%.6f, %.6f, alt=%.1f m)",
                self._geo_context.name or "unnamed",
                self._geo_context.lat0_deg,
                self._geo_context.lon0_deg,
                self._geo_context.alt0_m,
            )

        # Per-sensor health trackers
        self._sensor_health: dict[str, _SensorHealth] = {
            "camera": _SensorHealth("camera", max_errors),
            "detector": _SensorHealth("detector", max_errors),
            "radar": _SensorHealth("radar", max_errors),
            "multifreq_radar": _SensorHealth("multifreq_radar", max_errors),
            "thermal": _SensorHealth("thermal", max_errors),
            "quantum_radar": _SensorHealth("quantum_radar", max_errors),
            "fusion": _SensorHealth("fusion", max_errors),
        }

        # Initialize camera sensor
        cam_cfg = config.sentinel.sensors.camera
        self._camera = CameraAdapter(
            source=cam_cfg.source,
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
            buffer_size=cam_cfg.get("buffer_size", 1),
            backend=cam_cfg.get("backend", "auto"),
            clock=self._clock,
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
        self._latest_frame: np.ndarray | None = None
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

        # --- Multi-Frequency Radar (Phase 5) ---
        self._multifreq_radar_enabled = config.sentinel.sensors.get("multifreq_radar", {}).get("enabled", False)
        self._multifreq_radar = None
        self._multifreq_correlator = None
        self._latest_correlated_detections: list = []

        if self._multifreq_radar_enabled:
            self._init_multifreq_radar(config)

        # --- Thermal (Phase 5) ---
        self._thermal_enabled = config.sentinel.sensors.get("thermal", {}).get("enabled", False)
        self._thermal = None
        self._thermal_track_manager = None
        self._latest_thermal_detections: list[Detection] = []
        self._latest_thermal_tracks: list = []

        if self._thermal_enabled:
            self._init_thermal(config)

        # --- Quantum Radar (Phase 6) ---
        self._quantum_radar_enabled = config.sentinel.sensors.get("quantum_radar", {}).get("enabled", False)
        self._quantum_radar = None
        self._quantum_radar_track_manager = None
        self._latest_quantum_radar_detections: list[Detection] = []
        self._latest_quantum_radar_tracks: list = []

        if self._quantum_radar_enabled:
            self._init_quantum_radar(config)

        # --- IFF / ROE defaults (Phase 19) ---
        self._iff_interrogator = None
        self._roe_engine = None
        self._controlled_airspace = False

        # --- Multi-Sensor Fusion (Phase 5+6) ---
        self._multi_sensor_fusion = None
        self._latest_enhanced_fused: list = []
        if self._multifreq_radar_enabled or self._thermal_enabled or self._quantum_radar_enabled:
            self._init_multi_sensor_fusion(config)

        # --- Engagement Zones (Phase 21) ---
        self._engagement_manager = None
        self._latest_engagement_plan = None
        eng_cfg = config.sentinel.get("engagement", {})
        if eng_cfg.get("enabled", False):
            try:
                from sentinel.engagement.manager import EngagementManager
                self._engagement_manager = EngagementManager.from_config(
                    eng_cfg,
                    geo_context=self._geo_context if hasattr(self, '_geo_context') else None,
                )
                if self._engagement_manager is not None:
                    logger.info("Engagement zone system enabled with %d zones, %d weapons",
                                len(self._engagement_manager.zone_manager.get_all_zones()),
                                len(self._engagement_manager.weapons))
            except Exception:
                logger.exception("Failed to initialize engagement system")

        # --- Track History & Replay (Phase 22) ---
        self._history_recorder = None
        hist_cfg = config.sentinel.get("history", {})
        if hist_cfg.get("enabled", False):
            try:
                from sentinel.history.config import HistoryConfig
                from sentinel.history.recorder import HistoryRecorder

                history_config = HistoryConfig.from_omegaconf(hist_cfg)
                self._history_recorder = HistoryRecorder(history_config)
                if history_config.auto_record:
                    self._history_recorder.start()
                logger.info(
                    "History recording enabled (max_frames=%d, interval=%d, auto=%s)",
                    history_config.max_frames,
                    history_config.capture_interval,
                    history_config.auto_record,
                )
            except Exception:
                logger.exception("Failed to initialize history recorder")

        # --- Data Link / STANAG 5516 (Phase 23) ---
        self._datalink_gateway = None
        dl_cfg = config.sentinel.get("datalink", {})
        if dl_cfg.get("enabled", False):
            try:
                from sentinel.datalink.gateway import DataLinkGateway

                self._datalink_gateway = DataLinkGateway.from_config(
                    dl_cfg,
                    geo_context=self._geo_context if hasattr(self, '_geo_context') else None,
                )
                if self._datalink_gateway is not None:
                    logger.info("Data link gateway enabled (source=%s, rate=%.1f Hz)",
                                dl_cfg.get("source_id", "SENTINEL-01"),
                                dl_cfg.get("publish_rate_hz", 1.0))
            except Exception:
                logger.exception("Failed to initialize data link gateway")

        # --- Web Dashboard (Phase 10) ---
        web_cfg = config.sentinel.ui.get("web", {})
        self._web_dashboard = None
        if web_cfg.get("enabled", False):
            try:
                from sentinel.ui.web import WebDashboard

                self._web_dashboard = WebDashboard(web_cfg)
            except ImportError:
                logger.warning("Web dependencies not installed. Install with: pip install sentinel[web]")

    def _init_radar(self, config: DictConfig) -> None:
        """Initialize radar simulator, radar tracker, and fusion module."""
        from sentinel.fusion.track_fusion import TrackFusion
        from sentinel.sensors.radar_sim import RadarSimConfig, RadarSimulator
        from sentinel.tracking.radar_track_manager import RadarTrackManager

        radar_cfg = config.sentinel.sensors.radar
        sim_config = RadarSimConfig.from_omegaconf(radar_cfg, geo_context=self._geo_context)
        self._radar = RadarSimulator(sim_config, clock=self._clock, geo_context=self._geo_context)

        # Build radar tracking config
        radar_tracking = config.sentinel.get("tracking", {}).get("radar", {})
        radar_track_cfg = OmegaConf.create(
            {
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
            }
        )
        self._radar_track_manager = RadarTrackManager(radar_track_cfg)

        # Fusion
        fusion_cfg = config.sentinel.get("fusion", {})
        cam_cfg = config.sentinel.sensors.camera
        self._track_fusion = TrackFusion(
            camera_hfov_deg=fusion_cfg.get("camera_hfov_deg", 60.0),
            image_width_px=cam_cfg.get("width", 1280),
            azimuth_gate_deg=fusion_cfg.get("azimuth_gate_deg", 5.0),
            use_temporal_alignment=fusion_cfg.get("temporal_alignment", False),
            use_statistical_distance=fusion_cfg.get("use_statistical_distance", False),
            statistical_distance_gate=fusion_cfg.get("statistical_distance_gate", 9.21),
        )

        logger.info("Radar subsystem initialized (simulator mode, %.0f Hz)", radar_cfg.get("scan_rate_hz", 10))

    def _init_multifreq_radar(self, config: DictConfig) -> None:
        """Initialize multi-frequency radar simulator and correlator."""
        from sentinel.fusion.multifreq_correlator import MultiFreqCorrelator
        from sentinel.sensors.multifreq_radar_sim import MultiFreqRadarConfig, MultiFreqRadarSimulator

        mfr_cfg = config.sentinel.sensors.multifreq_radar
        mfr_config = MultiFreqRadarConfig.from_omegaconf(mfr_cfg, geo_context=self._geo_context)
        self._multifreq_radar = MultiFreqRadarSimulator(mfr_config, clock=self._clock, geo_context=self._geo_context)

        corr_cfg = config.sentinel.get("fusion", {}).get("multifreq_correlation", {})
        threat_cfg = config.sentinel.get("fusion", {}).get("threat_classification", {})
        self._multifreq_correlator = MultiFreqCorrelator(
            range_gate_m=corr_cfg.get("range_gate_m", 500.0),
            azimuth_gate_deg=corr_cfg.get("azimuth_gate_deg", 3.0),
            stealth_rcs_variation_db=threat_cfg.get("stealth_rcs_variation_db", 15.0),
        )

        # Also set up radar tracking if not already enabled via single radar
        if not self._radar_enabled:
            from sentinel.tracking.radar_track_manager import RadarTrackManager

            radar_tracking = config.sentinel.get("tracking", {}).get("radar", {})
            radar_track_cfg = OmegaConf.create(
                {
                    "filter": {"dt": 1.0 / mfr_cfg.get("scan_rate_hz", 10), "type": "ekf"},
                    "association": {
                        "gate_threshold": radar_tracking.get("association", {}).get("gate_threshold", 9.21)
                    },
                    "track_management": {
                        "confirm_hits": radar_tracking.get("track_management", {}).get("confirm_hits", 3),
                        "max_coast_frames": radar_tracking.get("track_management", {}).get("max_coast_frames", 5),
                        "max_tracks": radar_tracking.get("track_management", {}).get("max_tracks", 50),
                    },
                }
            )
            self._radar_track_manager = RadarTrackManager(radar_track_cfg)

        logger.info("Multi-frequency radar subsystem initialized (bands: %s)", mfr_cfg.get("bands", []))

    def _init_thermal(self, config: DictConfig) -> None:
        """Initialize thermal simulator and thermal track manager."""
        from sentinel.sensors.thermal_sim import ThermalSimConfig, ThermalSimulator
        from sentinel.tracking.thermal_track_manager import ThermalTrackManager

        thm_cfg = config.sentinel.sensors.thermal
        thm_config = ThermalSimConfig.from_omegaconf(thm_cfg, geo_context=self._geo_context)
        self._thermal = ThermalSimulator(thm_config, clock=self._clock, geo_context=self._geo_context)

        thermal_tracking = config.sentinel.get("tracking", {}).get("thermal", {})
        thermal_track_cfg = OmegaConf.create(
            {
                "filter": {
                    "dt": thermal_tracking.get("filter", {}).get("dt", 0.033),
                    "assumed_initial_range_m": thermal_tracking.get("filter", {}).get(
                        "assumed_initial_range_m", 10000.0
                    ),
                },
                "association": {
                    "gate_threshold": thermal_tracking.get("association", {}).get("gate_threshold", 9.21),
                },
                "track_management": {
                    "confirm_hits": thermal_tracking.get("track_management", {}).get("confirm_hits", 3),
                    "max_coast_frames": thermal_tracking.get("track_management", {}).get("max_coast_frames", 10),
                    "max_tracks": thermal_tracking.get("track_management", {}).get("max_tracks", 50),
                },
            }
        )
        self._thermal_track_manager = ThermalTrackManager(thermal_track_cfg)

        logger.info("Thermal subsystem initialized")

    def _init_quantum_radar(self, config: DictConfig) -> None:
        """Initialize quantum illumination radar simulator and tracker."""
        from sentinel.sensors.quantum_radar_sim import QuantumRadarConfig, QuantumRadarSimulator
        from sentinel.tracking.radar_track_manager import RadarTrackManager

        qr_cfg = config.sentinel.sensors.quantum_radar
        qr_config = QuantumRadarConfig.from_omegaconf(qr_cfg, geo_context=self._geo_context)
        self._quantum_radar = QuantumRadarSimulator(qr_config, clock=self._clock, geo_context=self._geo_context)

        # Reuse RadarTrackManager with quantum radar tracking params
        qr_tracking = config.sentinel.get("tracking", {}).get("quantum_radar", {})
        qr_track_cfg = OmegaConf.create(
            {
                "filter": {
                    "dt": qr_tracking.get("filter", {}).get("dt", 1.0 / qr_cfg.get("scan_rate_hz", 10)),
                    "type": "ekf",
                },
                "association": {
                    "gate_threshold": qr_tracking.get("association", {}).get("gate_threshold", 9.21),
                },
                "track_management": {
                    "confirm_hits": qr_tracking.get("track_management", {}).get("confirm_hits", 3),
                    "max_coast_frames": qr_tracking.get("track_management", {}).get("max_coast_frames", 5),
                    "max_tracks": qr_tracking.get("track_management", {}).get("max_tracks", 50),
                },
            }
        )
        self._quantum_radar_track_manager = RadarTrackManager(qr_track_cfg)

        logger.info("Quantum radar subsystem initialized (receiver: %s)", qr_cfg.get("receiver_type", "opa"))

    def _init_multi_sensor_fusion(self, config: DictConfig) -> None:
        """Initialize enhanced multi-sensor fusion."""
        from sentinel.fusion.multi_sensor_fusion import MultiSensorFusion

        fusion_cfg = config.sentinel.get("fusion", {})
        cam_cfg = config.sentinel.sensors.camera
        threat_cfg = fusion_cfg.get("threat_classification", {})

        # IFF interrogator (Phase 19)
        iff_interrogator = None
        iff_cfg = config.sentinel.get("iff", {})
        if iff_cfg.get("enabled", False):
            try:
                from sentinel.sensors.iff import IFFConfig, IFFInterrogator

                iff_config = IFFConfig.from_omegaconf(iff_cfg)
                iff_interrogator = IFFInterrogator(iff_config)
                logger.info("IFF interrogator enabled (modes: %s)", iff_cfg.get("modes", []))
            except Exception:
                logger.warning("IFF interrogator init failed, continuing without IFF")

        # ROE engine (Phase 19)
        roe_engine = None
        roe_cfg = config.sentinel.get("roe", {})
        if roe_cfg.get("enabled", False):
            try:
                from sentinel.classification.roe import ROEConfig, ROEEngine

                roe_config = ROEConfig.from_omegaconf(roe_cfg)
                roe_engine = ROEEngine(roe_config)
                logger.info("ROE engine enabled (default posture: %s)", roe_cfg.get("default_posture", "weapons_hold"))
            except Exception:
                logger.warning("ROE engine init failed, continuing without ROE")

        self._iff_interrogator = iff_interrogator
        self._roe_engine = roe_engine
        self._controlled_airspace = iff_cfg.get("controlled_airspace", False)

        self._multi_sensor_fusion = MultiSensorFusion(
            camera_hfov_deg=fusion_cfg.get("camera_hfov_deg", 60.0),
            image_width_px=cam_cfg.get("width", 1280),
            azimuth_gate_deg=fusion_cfg.get("azimuth_gate_deg", 5.0),
            thermal_azimuth_gate_deg=fusion_cfg.get("thermal_azimuth_gate_deg", 3.0),
            min_fusion_quality=fusion_cfg.get("min_fusion_quality", 0.0),
            hypersonic_temp_threshold_k=threat_cfg.get("hypersonic_temp_threshold_k", 1500.0),
            stealth_rcs_variation_db=threat_cfg.get("stealth_rcs_variation_db", 15.0),
            use_temporal_alignment=fusion_cfg.get("temporal_alignment", False),
            use_statistical_distance=fusion_cfg.get("use_statistical_distance", False),
            statistical_distance_gate=fusion_cfg.get("statistical_distance_gate", 9.21),
            threat_classification_method=threat_cfg.get("method", "rule_based"),
            threat_model_path=threat_cfg.get("model_path", None),
            threat_confidence_threshold=threat_cfg.get("confidence_threshold", 0.6),
            intent_estimation_enabled=threat_cfg.get("intent_estimation", False),
            iff_interrogator=iff_interrogator,
            roe_engine=roe_engine,
            controlled_airspace=self._controlled_airspace,
        )

        logger.info("Multi-sensor fusion initialized (camera + radar + thermal)")

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

        if self._multifreq_radar_enabled and self._multifreq_radar is not None:
            self._multifreq_radar.connect()
            logger.info("Multi-frequency radar simulator connected.")

        if self._thermal_enabled and self._thermal is not None:
            self._thermal.connect()
            logger.info("Thermal simulator connected.")

        if self._quantum_radar_enabled and self._quantum_radar is not None:
            self._quantum_radar.connect()
            logger.info("Quantum radar simulator connected.")

        logger.info("Warming up detector...")
        self._detector.warmup()
        logger.info("Tracker ready: max %d tracks", self._config.sentinel.tracking.track_management.max_tracks)
        if self._radar_enabled:
            logger.info("Radar tracking + fusion enabled.")
        if self._multifreq_radar_enabled:
            logger.info("Multi-frequency radar + correlation enabled.")
        if self._thermal_enabled:
            logger.info("Thermal tracking enabled.")
        if self._quantum_radar_enabled:
            logger.info("Quantum illumination radar enabled.")

        if self._web_dashboard is not None:
            if self._history_recorder is not None:
                self._web_dashboard.set_history_recorder(self._history_recorder)
            self._web_dashboard.start()

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
        mfr_interval = 1.0 / self._config.sentinel.sensors.get("multifreq_radar", {}).get("scan_rate_hz", 10)
        thermal_interval = 1.0 / self._config.sentinel.sensors.get("thermal", {}).get("frame_rate_hz", 30)
        qr_interval = 1.0 / self._config.sentinel.sensors.get("quantum_radar", {}).get("scan_rate_hz", 10)
        last_radar_time = 0.0
        last_mfr_time = 0.0
        last_thermal_time = 0.0
        last_qr_time = 0.0

        while self._running:
            # 1. Read camera frame
            try:
                frame_data = self._camera.read_frame()
            except Exception:
                logger.exception("Camera read failed")
                health = self._sensor_health["camera"]
                disabled = health.record_error()
                if disabled:
                    logger.warning("Camera disabled after %d consecutive errors", health.max_errors)
                if not self._graceful_degradation or not health.enabled:
                    break
                continue

            if frame_data is None:
                if not self._camera.is_connected:
                    logger.warning("Camera disconnected.")
                    break
                continue

            self._sensor_health["camera"].record_success()
            self._timer.tick()
            frame = frame_data.data
            timestamp = frame_data.timestamp

            # 2. Camera detection + tracking
            try:
                t0 = time.perf_counter()
                detections = self._detector.detect(frame, timestamp)
                t1 = time.perf_counter()
                self._latest_detections = detections
                tracks = self._track_manager.step(detections)
                t2 = time.perf_counter()
                self._latest_tracks = tracks
                self._sensor_health["detector"].record_success()
                self._update_timing("detect_ms", (t1 - t0) * 1000)
                self._update_timing("track_ms", (t2 - t1) * 1000)
            except Exception:
                logger.exception("Detection/tracking failed")
                health = self._sensor_health["detector"]
                disabled = health.record_error()
                if disabled:
                    logger.warning("Detector disabled after %d consecutive errors", health.max_errors)
                if not self._graceful_degradation or not health.enabled:
                    break
                detections = []
                tracks = self._latest_tracks

            # 3. Radar scan (at radar rate, if enabled)
            if (
                self._radar_enabled
                and self._radar is not None
                and self._sensor_health["radar"].enabled
                and (timestamp - last_radar_time) >= radar_interval
            ):
                try:
                    t_radar = time.perf_counter()
                    self._process_radar_scan()
                    self._update_timing("radar_ms", (time.perf_counter() - t_radar) * 1000)
                    self._sensor_health["radar"].record_success()
                    last_radar_time = timestamp
                except Exception:
                    logger.exception("Radar scan failed")
                    if self._sensor_health["radar"].record_error():
                        logger.warning(
                            "Radar disabled after %d consecutive errors", self._sensor_health["radar"].max_errors
                        )

            # 3b. Multi-frequency radar scan (if enabled)
            if (
                self._multifreq_radar_enabled
                and self._multifreq_radar is not None
                and self._sensor_health["multifreq_radar"].enabled
                and (timestamp - last_mfr_time) >= mfr_interval
            ):
                try:
                    self._process_multifreq_radar_scan()
                    self._sensor_health["multifreq_radar"].record_success()
                    last_mfr_time = timestamp
                except Exception:
                    logger.exception("Multi-freq radar scan failed")
                    if self._sensor_health["multifreq_radar"].record_error():
                        logger.warning(
                            "Multi-freq radar disabled after %d consecutive errors",
                            self._sensor_health["multifreq_radar"].max_errors,
                        )

            # 3c. Thermal scan (if enabled)
            if (
                self._thermal_enabled
                and self._thermal is not None
                and self._sensor_health["thermal"].enabled
                and (timestamp - last_thermal_time) >= thermal_interval
            ):
                try:
                    self._process_thermal_scan()
                    self._sensor_health["thermal"].record_success()
                    last_thermal_time = timestamp
                except Exception:
                    logger.exception("Thermal scan failed")
                    if self._sensor_health["thermal"].record_error():
                        logger.warning(
                            "Thermal disabled after %d consecutive errors", self._sensor_health["thermal"].max_errors
                        )

            # 3d. Quantum radar scan (if enabled)
            if (
                self._quantum_radar_enabled
                and self._quantum_radar is not None
                and self._sensor_health["quantum_radar"].enabled
                and (timestamp - last_qr_time) >= qr_interval
            ):
                try:
                    self._process_quantum_radar_scan()
                    self._sensor_health["quantum_radar"].record_success()
                    last_qr_time = timestamp
                except Exception:
                    logger.exception("Quantum radar scan failed")
                    if self._sensor_health["quantum_radar"].record_error():
                        logger.warning(
                            "Quantum radar disabled after %d consecutive errors",
                            self._sensor_health["quantum_radar"].max_errors,
                        )

            # 4. Fusion (every camera frame, using latest tracks)
            try:
                t_fuse = time.perf_counter()
                if self._multi_sensor_fusion is not None:
                    self._latest_enhanced_fused = self._multi_sensor_fusion.fuse(
                        self._latest_tracks,
                        self._latest_radar_tracks,
                        self._latest_thermal_tracks if self._thermal_enabled else None,
                        self._latest_correlated_detections if self._multifreq_radar_enabled else None,
                        quantum_radar_tracks=self._latest_quantum_radar_tracks if self._quantum_radar_enabled else None,
                    )
                elif self._track_fusion is not None:
                    self._latest_fused_tracks = self._track_fusion.fuse(
                        self._latest_tracks,
                        self._latest_radar_tracks,
                    )
                self._update_timing("fusion_ms", (time.perf_counter() - t_fuse) * 1000)
                self._sensor_health["fusion"].record_success()
            except Exception:
                logger.exception("Fusion failed")
                self._sensor_health["fusion"].record_error()

            # 4b. Engagement evaluation (if enabled)
            if self._engagement_manager is not None and self._latest_enhanced_fused:
                try:
                    self._latest_engagement_plan = self._engagement_manager.evaluate(
                        self._latest_enhanced_fused,
                        current_time=timestamp,
                    )
                except Exception:
                    logger.exception("Engagement evaluation failed")

            # 4c. History recording (if enabled)
            if self._history_recorder is not None:
                try:
                    self._history_recorder.record_frame(self)
                except Exception:
                    logger.debug("History recording failed", exc_info=True)

            # 4d. Data link publishing (if enabled)
            if self._datalink_gateway is not None:
                try:
                    tracks_to_publish = self._latest_enhanced_fused or self._latest_fused_tracks or []
                    self._datalink_gateway.publish_tracks(tracks_to_publish, timestamp)
                    self._datalink_gateway.process_incoming()
                except Exception:
                    logger.debug("Data link gateway failed", exc_info=True)

            # 5. Render HUD
            if self._hud is not None:
                try:
                    t_render = time.perf_counter()
                    status = self.get_system_status()
                    display = self._hud.render(
                        frame,
                        tracks,
                        detections,
                        status,
                        radar_tracks=self._latest_radar_tracks
                        if self._radar_enabled or self._multifreq_radar_enabled
                        else None,
                        fused_tracks=self._latest_fused_tracks
                        if self._radar_enabled and not self._multi_sensor_fusion
                        else None,
                        thermal_tracks=self._latest_thermal_tracks if self._thermal_enabled else None,
                        enhanced_fused_tracks=self._latest_enhanced_fused if self._multi_sensor_fusion else None,
                    )
                    self._update_timing("render_ms", (time.perf_counter() - t_render) * 1000)
                except Exception:
                    logger.exception("HUD render failed")
                    display = frame
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

            # 7. Publish to web dashboard
            if self._web_dashboard is not None:
                try:
                    self._web_dashboard.publish(self)
                except Exception:
                    logger.debug("Web dashboard publish failed", exc_info=True)

    def _update_timing(self, key: str, value_ms: float) -> None:
        """Update an EMA timing measurement."""
        a = self._timing_alpha
        self._timing[key] = a * value_ms + (1 - a) * self._timing.get(key, value_ms)

    def _process_radar_scan(self) -> None:
        """Read and process one radar scan."""
        from sentinel.sensors.radar_sim import radar_frame_to_detections

        radar_frame = self._radar.read_frame()
        if radar_frame is not None:
            radar_dets = radar_frame_to_detections(radar_frame)
            self._latest_radar_detections = radar_dets
            self._latest_radar_tracks = self._radar_track_manager.step(radar_dets)

    def _process_multifreq_radar_scan(self) -> None:
        """Read multi-freq radar, correlate across bands, and track."""
        from sentinel.sensors.multifreq_radar_sim import multifreq_radar_frame_to_detections

        frame = self._multifreq_radar.read_frame()
        if frame is None:
            return

        dets = multifreq_radar_frame_to_detections(frame)
        self._latest_radar_detections = dets

        # Correlate across frequency bands
        if self._multifreq_correlator is not None:
            self._latest_correlated_detections, _ = self._multifreq_correlator.correlate(dets)

        # Feed primary detections into radar tracker
        if self._radar_track_manager is not None:
            # Use the primary detection from each correlated group
            primary_dets = []
            for cd in self._latest_correlated_detections:
                primary_dets.append(cd.primary_detection)
            self._latest_radar_tracks = self._radar_track_manager.step(primary_dets)

    def _process_thermal_scan(self) -> None:
        """Read thermal frame and track."""
        from sentinel.sensors.thermal_sim import thermal_frame_to_detections

        frame = self._thermal.read_frame()
        if frame is None:
            return

        dets = thermal_frame_to_detections(frame)
        self._latest_thermal_detections = dets
        self._latest_thermal_tracks = self._thermal_track_manager.step(dets)

    def _process_quantum_radar_scan(self) -> None:
        """Read quantum radar frame and track."""
        from sentinel.sensors.quantum_radar_sim import quantum_radar_frame_to_detections

        frame = self._quantum_radar.read_frame()
        if frame is None:
            return

        dets = quantum_radar_frame_to_detections(frame)
        self._latest_quantum_radar_detections = dets
        self._latest_quantum_radar_tracks = self._quantum_radar_track_manager.step(dets)

    def _shutdown(self) -> None:
        logger.info("Shutting down pipeline...")

        # Stop web dashboard first (non-blocking)
        if self._web_dashboard is not None:
            try:
                self._web_dashboard.stop()
            except Exception:
                logger.debug("Web dashboard stop failed", exc_info=True)
        self._running = False

        # Disconnect each sensor in isolation so one failure doesn't prevent others
        for name, sensor in [
            ("camera", self._camera),
            ("radar", self._radar),
            ("multifreq_radar", self._multifreq_radar),
            ("thermal", self._thermal),
            ("quantum_radar", self._quantum_radar),
        ]:
            if sensor is not None:
                try:
                    sensor.disconnect()
                except Exception:
                    logger.exception("Error disconnecting %s", name)

        try:
            cv2.destroyAllWindows()
        except Exception:
            logger.exception("Error destroying OpenCV windows")

        logger.info("Pipeline stopped. Final stats:")
        try:
            logger.info(
                "  Camera tracks created: %d", len(self._track_manager._tracks) + self._track_manager.track_count
            )
            logger.info("  Active camera tracks: %d", self._track_manager.track_count)
            if self._radar_enabled and self._radar_track_manager is not None:
                logger.info("  Active radar tracks: %d", self._radar_track_manager.track_count)
                logger.info("  Fused tracks: %d", len(self._latest_fused_tracks))
            if self._multifreq_radar_enabled:
                logger.info("  Correlated detections: %d", len(self._latest_correlated_detections))
            if self._thermal_enabled and self._thermal_track_manager is not None:
                logger.info("  Active thermal tracks: %d", self._thermal_track_manager.track_count)
            if self._quantum_radar_enabled and self._quantum_radar_track_manager is not None:
                logger.info("  Active quantum radar tracks: %d", self._quantum_radar_track_manager.track_count)
            if self._multi_sensor_fusion is not None:
                logger.info("  Enhanced fused tracks: %d", len(self._latest_enhanced_fused))
        except Exception:
            logger.exception("Error logging final stats")

    def get_system_status(self) -> dict:
        status = {
            "fps": self._timer.fps,
            "detection_count": len(self._latest_detections),
            "track_count": self._track_manager.track_count,
            "confirmed_count": len(self._track_manager.confirmed_tracks),
            "camera_connected": self._camera.is_connected,
            "uptime": self._clock.elapsed(),
            "sensor_health": {
                name: {"enabled": h.enabled, "error_count": h.error_count} for name, h in self._sensor_health.items()
            },
            "timing_ms": dict(self._timing),
        }
        if self._radar_enabled:
            status["radar_connected"] = self._radar.is_connected if self._radar else False
            status["radar_detection_count"] = len(self._latest_radar_detections)
            status["radar_track_count"] = self._radar_track_manager.track_count if self._radar_track_manager else 0
            status["fused_track_count"] = len(self._latest_fused_tracks)
        if self._multifreq_radar_enabled:
            status["radar_connected"] = self._multifreq_radar.is_connected if self._multifreq_radar else False
            status["radar_detection_count"] = len(self._latest_radar_detections)
            status["radar_track_count"] = self._radar_track_manager.track_count if self._radar_track_manager else 0
            status["correlated_count"] = len(self._latest_correlated_detections)
        if self._thermal_enabled:
            status["thermal_connected"] = self._thermal.is_connected if self._thermal else False
            status["thermal_track_count"] = (
                self._thermal_track_manager.track_count if self._thermal_track_manager else 0
            )
        if self._quantum_radar_enabled:
            status["quantum_radar_connected"] = self._quantum_radar.is_connected if self._quantum_radar else False
            status["quantum_radar_track_count"] = (
                self._quantum_radar_track_manager.track_count if self._quantum_radar_track_manager else 0
            )
        if self._multi_sensor_fusion is not None:
            status["fused_track_count"] = len(self._latest_enhanced_fused)
            # Threat counts
            threat_counts: dict[str, int] = {}
            for eft in self._latest_enhanced_fused:
                lvl = getattr(eft, "threat_level", "UNKNOWN")
                threat_counts[lvl] = threat_counts.get(lvl, 0) + 1
            status["threat_counts"] = threat_counts

        # Clock mode
        from sentinel.core.clock import SimClock

        status["clock_mode"] = "simulated" if isinstance(self._clock, SimClock) else "realtime"
        status["sim_time"] = self._clock.now() if isinstance(self._clock, SimClock) else None

        # C++ acceleration
        try:
            from sentinel.tracking._accel import has_cpp_acceleration, has_cpp_batch

            status["cpp_accel"] = {
                "core": has_cpp_acceleration(),
                "batch": has_cpp_batch(),
            }
        except ImportError:
            status["cpp_accel"] = {"core": False, "batch": False}

        # IFF/ROE status
        status["iff_enabled"] = self._iff_interrogator is not None
        status["roe_enabled"] = self._roe_engine is not None

        # Network status (Phase 20)
        net_cfg = self._config.sentinel.get("network", {})
        status["network_enabled"] = net_cfg.get("enabled", False)
        if status["network_enabled"]:
            status["network_node_id"] = net_cfg.get("node_id", "LOCAL")
            status["network_role"] = net_cfg.get("role", "sensor")

        # Engagement status (Phase 21)
        status["engagement_enabled"] = self._engagement_manager is not None
        if self._engagement_manager is not None and self._latest_engagement_plan is not None:
            plan = self._latest_engagement_plan
            status["engagement_assignments"] = len(plan.assignment.assignments)
            status["engagement_zones"] = len(self._engagement_manager.zone_manager.get_all_zones())
            status["engagement_weapons"] = len(self._engagement_manager.weapons)

        # EW status (Phase 13)
        ew_cfg = self._config.sentinel.get("environment", {}).get("ew", {})
        status["ew_enabled"] = ew_cfg.get("enabled", False)

        # History status (Phase 22)
        status["history_enabled"] = self._history_recorder is not None
        if self._history_recorder is not None:
            status["history"] = self._history_recorder.get_status()

        # Data link status (Phase 23)
        status["datalink_enabled"] = self._datalink_gateway is not None
        if self._datalink_gateway is not None:
            status["datalink"] = self._datalink_gateway.get_stats()

        return status

    @property
    def history_recorder(self):
        return self._history_recorder

    @property
    def datalink_gateway(self):
        return self._datalink_gateway

    def get_track_snapshot(self) -> list[Track]:
        return self._latest_tracks

    def get_latest_hud_frame(self) -> np.ndarray | None:
        return self._latest_frame
