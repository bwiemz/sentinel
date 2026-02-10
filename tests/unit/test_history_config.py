"""Tests for sentinel.history.config — HistoryConfig dataclass."""

from __future__ import annotations

import pytest

from sentinel.history.config import HistoryConfig


class TestHistoryConfigDefaults:
    def test_defaults(self):
        cfg = HistoryConfig()
        assert cfg.enabled is False
        assert cfg.max_frames == 10000
        assert cfg.capture_interval == 1
        assert cfg.auto_record is False
        assert cfg.storage_directory == "data/recordings"
        assert cfg.storage_format == "msgpack"
        assert cfg.compression is False
        assert cfg.default_speed == 1.0
        assert cfg.loop is False


class TestHistoryConfigFromOmegaconf:
    def test_from_none(self):
        cfg = HistoryConfig.from_omegaconf(None)
        assert cfg.enabled is False

    def test_from_empty_dict(self):
        cfg = HistoryConfig.from_omegaconf({})
        assert cfg.enabled is False
        assert cfg.max_frames == 10000

    def test_from_full_dict(self):
        cfg = HistoryConfig.from_omegaconf(
            {
                "enabled": True,
                "max_frames": 5000,
                "capture_interval": 3,
                "auto_record": True,
                "storage": {
                    "directory": "/tmp/rec",
                    "format": "json",
                    "compression": True,
                },
                "replay": {
                    "default_speed": 2.0,
                    "loop": True,
                },
            }
        )
        assert cfg.enabled is True
        assert cfg.max_frames == 5000
        assert cfg.capture_interval == 3
        assert cfg.auto_record is True
        assert cfg.storage_directory == "/tmp/rec"
        assert cfg.storage_format == "json"
        assert cfg.compression is True
        assert cfg.default_speed == 2.0
        assert cfg.loop is True

    def test_partial_storage(self):
        cfg = HistoryConfig.from_omegaconf(
            {"storage": {"format": "json"}}
        )
        assert cfg.storage_format == "json"
        assert cfg.storage_directory == "data/recordings"

    def test_partial_replay(self):
        cfg = HistoryConfig.from_omegaconf(
            {"replay": {"loop": True}}
        )
        assert cfg.loop is True
        assert cfg.default_speed == 1.0

    def test_capture_interval_clamped_to_1(self):
        cfg = HistoryConfig.from_omegaconf({"capture_interval": 0})
        assert cfg.capture_interval == 1

    def test_none_storage(self):
        cfg = HistoryConfig.from_omegaconf({"storage": None})
        assert cfg.storage_directory == "data/recordings"

    def test_none_replay(self):
        cfg = HistoryConfig.from_omegaconf({"replay": None})
        assert cfg.default_speed == 1.0
