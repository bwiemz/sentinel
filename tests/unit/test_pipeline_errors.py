"""Tests for pipeline error handling and graceful degradation.

Tests that sensor failures are isolated, error counters work,
sensors auto-disable after N errors, shutdown is fault-tolerant,
and system_status includes sensor health.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.pipeline import SentinelPipeline, _SensorHealth
from sentinel.core.types import SensorType
from sentinel.sensors.frame import SensorFrame

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_config(**overrides):
    """Build a minimal config that can construct a SentinelPipeline."""
    cfg = OmegaConf.create(
        {
            "sentinel": {
                "system": {
                    "name": "TEST",
                    "version": "0.0.1",
                    "log_level": "WARNING",
                    "graceful_degradation": True,
                    "max_sensor_errors": 3,
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
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _make_camera_frame():
    """Produce a SensorFrame wrapping a small synthetic image."""
    return SensorFrame(
        data=np.zeros((240, 320, 3), dtype=np.uint8),
        timestamp=1.0,
        sensor_type=SensorType.CAMERA,
        frame_number=1,
    )


# ---------------------------------------------------------------------------
# _SensorHealth unit tests
# ---------------------------------------------------------------------------


class TestSensorHealth:
    def test_initial_state(self):
        h = _SensorHealth("test", max_errors=5)
        assert h.enabled is True
        assert h.error_count == 0

    def test_record_success_resets(self):
        h = _SensorHealth("test", max_errors=5)
        h.error_count = 3
        h.record_success()
        assert h.error_count == 0

    def test_record_error_increments(self):
        h = _SensorHealth("test", max_errors=5)
        h.record_error()
        assert h.error_count == 1
        assert h.enabled is True

    def test_auto_disable_at_threshold(self):
        h = _SensorHealth("test", max_errors=3)
        h.record_error()
        h.record_error()
        disabled = h.record_error()
        assert disabled is True
        assert h.enabled is False

    def test_stays_disabled_after_threshold(self):
        h = _SensorHealth("test", max_errors=2)
        h.record_error()
        h.record_error()  # disables
        result = h.record_error()  # already disabled
        assert result is False  # not "just disabled"
        assert h.enabled is False

    def test_success_does_not_reenable(self):
        h = _SensorHealth("test", max_errors=2)
        h.record_error()
        h.record_error()
        assert h.enabled is False
        h.record_success()
        assert h.error_count == 0
        assert h.enabled is False  # stays disabled


# ---------------------------------------------------------------------------
# Pipeline construction (mocked YOLO)
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    """Create a pipeline with YOLO mocked out."""
    cfg = _minimal_config()
    with (
        patch("sentinel.core.pipeline.YOLODetector") as MockDetector,
        patch("sentinel.core.pipeline.CameraAdapter") as MockCamera,
    ):
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        MockDetector.return_value = mock_det

        mock_cam = MagicMock()
        mock_cam.is_connected = True
        mock_cam.connect.return_value = True
        mock_cam.read_frame.return_value = _make_camera_frame()
        MockCamera.return_value = mock_cam

        p = SentinelPipeline(cfg)
        yield p


@pytest.fixture
def pipeline_with_radar():
    """Pipeline with radar enabled but mocked."""
    cfg = _minimal_config(
        **{
            "sentinel": {
                "sensors": {
                    "radar": {
                        "enabled": True,
                        "mode": "simulator",
                        "scan_rate_hz": 10,
                        "max_range_m": 10000,
                        "fov_deg": 120,
                        "noise": {
                            "range_m": 5.0,
                            "azimuth_deg": 1.0,
                            "velocity_mps": 0.5,
                            "rcs_dbsm": 2.0,
                            "false_alarm_rate": 0.01,
                            "detection_probability": 0.9,
                        },
                        "scenario": {"targets": []},
                    },
                },
                "tracking": {
                    "radar": {
                        "filter": {"dt": 0.1, "type": "ekf"},
                        "association": {"gate_threshold": 9.21},
                        "track_management": {"confirm_hits": 3, "max_coast_frames": 5, "max_tracks": 50},
                    },
                },
            },
        }
    )
    with (
        patch("sentinel.core.pipeline.YOLODetector") as MockDetector,
        patch("sentinel.core.pipeline.CameraAdapter") as MockCamera,
    ):
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        MockDetector.return_value = mock_det

        mock_cam = MagicMock()
        mock_cam.is_connected = True
        mock_cam.connect.return_value = True
        mock_cam.read_frame.return_value = _make_camera_frame()
        MockCamera.return_value = mock_cam

        p = SentinelPipeline(cfg)
        yield p


# ---------------------------------------------------------------------------
# Pipeline health / status tests
# ---------------------------------------------------------------------------


class TestPipelineHealth:
    def test_sensor_health_in_status(self, pipeline):
        status = pipeline.get_system_status()
        assert "sensor_health" in status
        assert "camera" in status["sensor_health"]
        assert "detector" in status["sensor_health"]
        assert "radar" in status["sensor_health"]

    def test_health_shows_enabled(self, pipeline):
        status = pipeline.get_system_status()
        cam = status["sensor_health"]["camera"]
        assert cam["enabled"] is True
        assert cam["error_count"] == 0

    def test_max_errors_from_config(self, pipeline):
        h = pipeline._sensor_health["camera"]
        assert h.max_errors == 3

    def test_graceful_degradation_from_config(self, pipeline):
        assert pipeline._graceful_degradation is True


# ---------------------------------------------------------------------------
# Camera error isolation
# ---------------------------------------------------------------------------


class TestCameraErrors:
    def test_camera_exception_counted(self, pipeline):
        """Camera read exception should increment error counter, not crash."""
        pipeline._camera.read_frame.side_effect = RuntimeError("USB error")
        pipeline._running = True

        # Run one iteration -- should catch and continue
        pipeline._main_loop()  # will break after camera disabled (max_errors=3)

        h = pipeline._sensor_health["camera"]
        assert h.error_count >= 1

    def test_camera_disabled_after_max_errors(self, pipeline):
        """Camera should be auto-disabled after 3 consecutive errors."""
        pipeline._camera.read_frame.side_effect = RuntimeError("USB error")
        pipeline._running = True

        pipeline._main_loop()

        h = pipeline._sensor_health["camera"]
        assert h.enabled is False
        assert h.error_count == 3

    def test_camera_error_resets_on_success(self, pipeline):
        """A successful read should reset the error counter."""
        # First call: error, second call: success (returns frame)
        frame = _make_camera_frame()
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Glitch")
            elif call_count[0] <= 3:
                return frame
            else:
                # Stop the loop
                pipeline._running = False
                return frame

        pipeline._camera.read_frame.side_effect = side_effect
        pipeline._running = True
        pipeline._main_loop()

        h = pipeline._sensor_health["camera"]
        assert h.error_count == 0  # reset by successful reads


# ---------------------------------------------------------------------------
# Detection error isolation
# ---------------------------------------------------------------------------


class TestDetectorErrors:
    def test_detector_exception_counted(self, pipeline):
        """Detection exception should be caught, not crash pipeline."""
        pipeline._detector.detect.side_effect = RuntimeError("Model error")
        pipeline._running = True

        call_count = [0]

        def limited_read():
            call_count[0] += 1
            if call_count[0] > 4:
                pipeline._running = False
            return _make_camera_frame()

        pipeline._camera.read_frame = limited_read
        pipeline._main_loop()

        h = pipeline._sensor_health["detector"]
        assert h.error_count >= 1

    def test_detector_disabled_after_max_errors(self, pipeline):
        """Detector auto-disables after N consecutive errors."""
        pipeline._detector.detect.side_effect = RuntimeError("Model error")
        pipeline._running = True

        pipeline._camera.read_frame.return_value = _make_camera_frame()
        pipeline._main_loop()

        h = pipeline._sensor_health["detector"]
        assert h.enabled is False


# ---------------------------------------------------------------------------
# Radar error isolation
# ---------------------------------------------------------------------------


class TestRadarErrors:
    def test_radar_exception_isolated(self, pipeline_with_radar):
        """Radar failure should not crash pipeline."""
        p = pipeline_with_radar
        # Replace radar with a mock since real RadarSimulator methods can't have side_effect
        mock_radar = MagicMock()
        mock_radar.read_frame.side_effect = RuntimeError("Radar fault")
        mock_radar.is_connected = True
        p._radar = mock_radar
        p._running = True

        call_count = [0]

        def limited_read():
            call_count[0] += 1
            if call_count[0] > 2:
                p._running = False
            f = _make_camera_frame()
            f.timestamp = float(call_count[0])  # force radar interval trigger
            return f

        p._camera.read_frame = limited_read
        p._main_loop()

        h = p._sensor_health["radar"]
        assert h.error_count >= 1
        assert p._sensor_health["camera"].error_count == 0

    def test_radar_disabled_after_errors(self, pipeline_with_radar):
        """Radar should auto-disable after max_errors consecutive failures."""
        p = pipeline_with_radar
        mock_radar = MagicMock()
        mock_radar.read_frame.side_effect = RuntimeError("Radar fault")
        mock_radar.is_connected = True
        p._radar = mock_radar
        p._running = True

        call_count = [0]

        def limited_read():
            call_count[0] += 1
            if call_count[0] > 5:
                p._running = False
            f = _make_camera_frame()
            f.timestamp = float(call_count[0])
            return f

        p._camera.read_frame = limited_read
        p._main_loop()

        h = p._sensor_health["radar"]
        assert h.enabled is False


# ---------------------------------------------------------------------------
# Shutdown fault tolerance
# ---------------------------------------------------------------------------


class TestShutdownFaultTolerance:
    def test_shutdown_survives_camera_disconnect_error(self, pipeline):
        """Shutdown should complete even if camera.disconnect() raises."""
        pipeline._camera.disconnect.side_effect = RuntimeError("Cannot close device")
        pipeline._shutdown()  # should not raise
        assert pipeline._running is False

    def test_shutdown_survives_radar_disconnect_error(self, pipeline_with_radar):
        """Shutdown completes even if radar disconnect fails."""
        p = pipeline_with_radar
        mock_radar = MagicMock()
        mock_radar.disconnect.side_effect = RuntimeError("Radar hangup")
        p._radar = mock_radar
        p._shutdown()
        assert p._running is False

    def test_shutdown_disconnects_all_sensors(self, pipeline_with_radar):
        """Each sensor disconnect is called independently."""
        p = pipeline_with_radar
        mock_radar = MagicMock()
        p._radar = mock_radar
        p._shutdown()
        p._camera.disconnect.assert_called_once()
        mock_radar.disconnect.assert_called_once()

    def test_shutdown_survives_cv2_error(self, pipeline):
        """Shutdown survives cv2.destroyAllWindows failure."""
        with patch("sentinel.core.pipeline.cv2") as mock_cv2:
            mock_cv2.destroyAllWindows.side_effect = RuntimeError("No display")
            pipeline._shutdown()
            assert pipeline._running is False


# ---------------------------------------------------------------------------
# YOLO input validation
# ---------------------------------------------------------------------------


def _make_bare_yolo_detector():
    """Create a YOLODetector without calling __init__ (avoids ultralytics import)."""
    from sentinel.detection.yolo import YOLODetector

    det = YOLODetector.__new__(YOLODetector)
    det._model = MagicMock()
    det._model_path = "test.pt"
    det._confidence = 0.25
    det._iou_threshold = 0.45
    det._device = "cpu"
    det._classes = None
    det._max_detections = 100
    det._image_size = 640
    det._class_names = {}
    return det


class TestYOLOInputValidation:
    """Test the input validation added to YOLODetector.detect()."""

    def test_none_frame_returns_empty(self):
        """detect(None) should return [] without crashing."""
        det = _make_bare_yolo_detector()
        result = det.detect(None, 1.0)
        assert result == []
        det._model.assert_not_called()

    def test_non_ndarray_returns_empty(self):
        """detect('not-an-array') should return []."""
        det = _make_bare_yolo_detector()
        result = det.detect("bad_input", 1.0)
        assert result == []

    def test_empty_frame_returns_empty(self):
        """detect(empty array) should return []."""
        det = _make_bare_yolo_detector()
        result = det.detect(np.array([]), 1.0)
        assert result == []

    def test_1d_frame_returns_empty(self):
        """detect(1D array) should return []."""
        det = _make_bare_yolo_detector()
        result = det.detect(np.zeros(100), 1.0)
        assert result == []


# ---------------------------------------------------------------------------
# Graceful degradation disabled
# ---------------------------------------------------------------------------


class TestGracefulDegradationDisabled:
    def test_pipeline_breaks_on_first_camera_error(self):
        """With graceful_degradation=False, camera error should break loop."""
        cfg = _minimal_config(**{"sentinel": {"system": {"graceful_degradation": False, "max_sensor_errors": 3}}})
        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            MockDet.return_value = MagicMock(detect=MagicMock(return_value=[]))
            mock_cam = MagicMock()
            mock_cam.read_frame.side_effect = RuntimeError("Fail")
            mock_cam.is_connected = True
            MockCam.return_value = mock_cam

            p = SentinelPipeline(cfg)
            p._running = True
            p._main_loop()

            # Should have broken out on first error
            assert p._sensor_health["camera"].error_count == 1
