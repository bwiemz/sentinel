"""End-to-end pipeline integration tests.

Tests pipeline start/stop, sensor failure recovery, and config validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentinel.core.pipeline import SentinelPipeline
from tests.integration.conftest import _minimal_pipeline_config, make_camera_frame


class TestPipelineE2E:
    def test_pipeline_starts_and_stops(self):
        """Pipeline should start, run a few frames, and stop cleanly."""
        cfg = _minimal_pipeline_config()

        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            mock_det = MagicMock()
            mock_det.detect.return_value = []
            MockDet.return_value = mock_det

            call_count = [0]

            def limited_read():
                call_count[0] += 1
                if call_count[0] > 5:
                    return None  # signal disconnect
                return make_camera_frame(timestamp=float(call_count[0]))

            mock_cam = MagicMock()
            mock_cam.is_connected.__bool__ = lambda self: call_count[0] <= 5
            type(mock_cam).is_connected = property(lambda self: call_count[0] <= 5)
            mock_cam.connect.return_value = True
            mock_cam.read_frame = limited_read
            MockCam.return_value = mock_cam

            p = SentinelPipeline(cfg)
            p.run()

            # Should have processed some frames
            assert call_count[0] > 0

    def test_sensor_failure_recovery(self):
        """Pipeline should survive transient sensor failures."""
        cfg = _minimal_pipeline_config()

        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            mock_det = MagicMock()
            mock_det.detect.return_value = []
            MockDet.return_value = mock_det

            call_count = [0]

            def flaky_read():
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("Transient USB error")
                if call_count[0] > 5:
                    return None
                return make_camera_frame(timestamp=float(call_count[0]))

            mock_cam = MagicMock()
            type(mock_cam).is_connected = property(lambda self: call_count[0] <= 5)
            mock_cam.connect.return_value = True
            mock_cam.read_frame = flaky_read
            MockCam.return_value = mock_cam

            p = SentinelPipeline(cfg)
            p.run()

            # Should have recovered from the single error
            assert p._sensor_health["camera"].enabled is True
            assert call_count[0] > 3

    def test_system_status_complete(self):
        """get_system_status should include all expected keys."""
        cfg = _minimal_pipeline_config()

        with (
            patch("sentinel.core.pipeline.YOLODetector") as MockDet,
            patch("sentinel.core.pipeline.CameraAdapter") as MockCam,
        ):
            MockDet.return_value = MagicMock(detect=MagicMock(return_value=[]))
            MockCam.return_value = MagicMock(is_connected=True)

            p = SentinelPipeline(cfg)
            status = p.get_system_status()

            assert "fps" in status
            assert "detection_count" in status
            assert "track_count" in status
            assert "sensor_health" in status
            assert "timing_ms" in status
            assert "camera" in status["sensor_health"]

    def test_config_validation_on_startup(self):
        """--validate-config should catch invalid config before pipeline starts."""
        from sentinel.core.config import SentinelConfig

        # Valid config should pass
        config = SentinelConfig("config/default.yaml")
        cfg = config.load(validate=True)
        assert cfg.sentinel.system.name == "SENTINEL"

    def test_config_validation_rejects_invalid(self):
        """Invalid config should raise ValidationError."""
        from pydantic import ValidationError

        from sentinel.core.config_schema import validate_config

        invalid = {
            "sentinel": {
                "detection": {"confidence": 5.0},  # Must be 0-1
            }
        }
        with pytest.raises(ValidationError):
            validate_config(invalid)
