"""Tests for the configuration system."""

import pytest
from omegaconf import DictConfig

from sentinel.core.config import SentinelConfig


class TestSentinelConfig:
    def test_load_default(self, config_path):
        config = SentinelConfig(config_path)
        cfg = config.load()
        assert isinstance(cfg, DictConfig)
        assert cfg.sentinel.system.name == "SENTINEL"

    def test_load_missing_file(self, tmp_path):
        config = SentinelConfig(tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            config.load()

    def test_override(self, config_path):
        config = SentinelConfig(config_path)
        config.load()
        config.override("sentinel.sensors.camera.source", "test.mp4")
        assert config.cfg.sentinel.sensors.camera.source == "test.mp4"

    def test_override_before_load(self, config_path):
        config = SentinelConfig(config_path)
        with pytest.raises(RuntimeError):
            config.override("sentinel.sensors.camera.source", "test.mp4")

    def test_camera_config_defaults(self, config_path):
        config = SentinelConfig(config_path)
        cfg = config.load()
        cam = cfg.sentinel.sensors.camera
        assert cam.enabled is True
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 30

    def test_detection_config_defaults(self, config_path):
        config = SentinelConfig(config_path)
        cfg = config.load()
        det = cfg.sentinel.detection
        assert det.model == "yolov8n.pt"
        assert det.confidence == 0.25
        assert det.device == "auto"
