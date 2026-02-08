"""Tests for structured logging and timing instrumentation."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

from sentinel.utils.logging import setup_logging


class TestSetupLogging:
    def test_console_output(self, capfd):
        """setup_logging() should produce output on stdout."""
        setup_logging("DEBUG")
        logger = logging.getLogger("sentinel.test_console")
        logger.info("hello console")
        out = capfd.readouterr().out
        assert "hello console" in out

    def test_log_level_propagation(self):
        """Setting level=WARNING should suppress INFO messages."""
        setup_logging("WARNING")
        root = logging.getLogger("sentinel")
        assert root.level == logging.WARNING

    def test_json_mode(self, capfd):
        """log_json=True should output valid JSON."""
        setup_logging("INFO", log_json=True)
        logger = logging.getLogger("sentinel.test_json")
        logger.info("json test")
        out = capfd.readouterr().out
        # Should contain a JSON object with the message
        assert "json test" in out
        # Try to parse as JSON (lines may have structlog wrappers)
        for line in out.strip().splitlines():
            if "json test" in line:
                data = json.loads(line)
                assert data["event"] == "json test"
                break

    def test_file_logging(self, tmp_path):
        """log_file should create a log file with content."""
        log_path = tmp_path / "sentinel.log"
        setup_logging("INFO", log_file=str(log_path))
        logger = logging.getLogger("sentinel.test_file")
        logger.info("file test message")
        # Flush and close handlers to release file lock (Windows)
        root = logging.getLogger("sentinel")
        for handler in root.handlers:
            handler.flush()
        assert log_path.exists()
        content = log_path.read_text()
        assert "file test message" in content
        # Cleanup: close file handlers so tmp_path can be removed
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root.removeHandler(handler)

    def test_third_party_loggers_quieted(self):
        """ultralytics and urllib3 should be at WARNING level."""
        setup_logging("DEBUG")
        assert logging.getLogger("ultralytics").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING

    def test_repeated_setup_no_duplicate_handlers(self):
        """Calling setup_logging twice should not add duplicate handlers."""
        setup_logging("INFO")
        count1 = len(logging.getLogger("sentinel").handlers)
        setup_logging("INFO")
        count2 = len(logging.getLogger("sentinel").handlers)
        assert count2 == count1

    def test_debug_level(self):
        """DEBUG level should be settable."""
        setup_logging("DEBUG")
        root = logging.getLogger("sentinel")
        assert root.level == logging.DEBUG

    def test_file_creates_parent_dirs(self, tmp_path):
        """log_file in a non-existent directory should create parent dirs."""
        log_path = tmp_path / "subdir" / "deep" / "sentinel.log"
        setup_logging("INFO", log_file=str(log_path))
        logger = logging.getLogger("sentinel.test_dir")
        logger.info("dir test")
        root = logging.getLogger("sentinel")
        for handler in root.handlers:
            handler.flush()
        assert log_path.exists()
        # Cleanup file handlers
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root.removeHandler(handler)


class TestPipelineTiming:
    """Test that pipeline exposes timing_ms in status."""

    def test_timing_in_status(self):
        """get_system_status() should include timing_ms dict."""
        from omegaconf import OmegaConf

        from sentinel.core.pipeline import SentinelPipeline

        cfg = OmegaConf.create(
            {
                "sentinel": {
                    "system": {
                        "name": "TEST",
                        "version": "0.0.1",
                        "log_level": "WARNING",
                        "graceful_degradation": True,
                        "max_sensor_errors": 10,
                    },
                    "sensors": {
                        "camera": {
                            "enabled": True,
                            "source": 0,
                            "width": 320,
                            "height": 240,
                            "fps": 30,
                            "buffer_size": 1,
                            "backend": "auto",
                        },
                        "radar": {"enabled": False, "scan_rate_hz": 10, "max_range_m": 10000, "fov_deg": 120},
                    },
                    "detection": {
                        "model": "yolov8n.pt",
                        "confidence": 0.25,
                        "iou_threshold": 0.45,
                        "device": "cpu",
                        "classes": None,
                        "max_detections": 100,
                        "image_size": 640,
                    },
                    "tracking": {
                        "filter": {"type": "kf", "dt": 0.033, "process_noise_std": 1.0, "measurement_noise_std": 10.0},
                        "association": {
                            "method": "hungarian",
                            "gate_threshold": 9.21,
                            "iou_weight": 0.5,
                            "mahalanobis_weight": 0.5,
                        },
                        "track_management": {
                            "confirm_hits": 3,
                            "confirm_window": 5,
                            "max_coast_frames": 15,
                            "max_tracks": 100,
                            "tentative_delete_misses": 3,
                        },
                    },
                    "fusion": {"enabled": False},
                    "ui": {"hud": {"enabled": False, "display": False}},
                }
            }
        )

        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            MockDet.return_value = MagicMock(detect=MagicMock(return_value=[]))
            mock_cam = MagicMock()
            mock_cam.is_connected = True
            MockCam.return_value = mock_cam

            p = SentinelPipeline(cfg)
            status = p.get_system_status()

            assert "timing_ms" in status
            assert "detect_ms" in status["timing_ms"]
            assert "track_ms" in status["timing_ms"]
            assert "fusion_ms" in status["timing_ms"]
            assert "render_ms" in status["timing_ms"]

    def test_update_timing_ema(self):
        """_update_timing should compute EMA."""
        from omegaconf import OmegaConf

        from sentinel.core.pipeline import SentinelPipeline

        cfg = OmegaConf.create(
            {
                "sentinel": {
                    "system": {
                        "name": "T",
                        "version": "0.0.1",
                        "log_level": "WARNING",
                        "graceful_degradation": True,
                        "max_sensor_errors": 10,
                    },
                    "sensors": {
                        "camera": {
                            "enabled": True,
                            "source": 0,
                            "width": 320,
                            "height": 240,
                            "fps": 30,
                            "buffer_size": 1,
                            "backend": "auto",
                        },
                        "radar": {"enabled": False, "scan_rate_hz": 10, "max_range_m": 10000, "fov_deg": 120},
                    },
                    "detection": {
                        "model": "yolov8n.pt",
                        "confidence": 0.25,
                        "iou_threshold": 0.45,
                        "device": "cpu",
                        "classes": None,
                        "max_detections": 100,
                        "image_size": 640,
                    },
                    "tracking": {
                        "filter": {"type": "kf", "dt": 0.033, "process_noise_std": 1.0, "measurement_noise_std": 10.0},
                        "association": {
                            "method": "hungarian",
                            "gate_threshold": 9.21,
                            "iou_weight": 0.5,
                            "mahalanobis_weight": 0.5,
                        },
                        "track_management": {
                            "confirm_hits": 3,
                            "confirm_window": 5,
                            "max_coast_frames": 15,
                            "max_tracks": 100,
                            "tentative_delete_misses": 3,
                        },
                    },
                    "fusion": {"enabled": False},
                    "ui": {"hud": {"enabled": False, "display": False}},
                }
            }
        )

        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            MockDet.return_value = MagicMock(detect=MagicMock(return_value=[]))
            MockCam.return_value = MagicMock(is_connected=True)

            p = SentinelPipeline(cfg)
            # First update: EMA = 0.1*10 + 0.9*0 = 1.0
            p._update_timing("detect_ms", 10.0)
            first = p._timing["detect_ms"]
            assert first > 0.0

            # Second update: EMA = 0.1*10 + 0.9*first
            p._update_timing("detect_ms", 10.0)
            second = p._timing["detect_ms"]
            assert second > first  # converging toward 10.0

            # After many updates, should converge near the value
            for _ in range(100):
                p._update_timing("detect_ms", 10.0)
            assert abs(p._timing["detect_ms"] - 10.0) < 0.5
