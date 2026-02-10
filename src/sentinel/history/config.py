"""History system configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HistoryConfig:
    """History recording & replay configuration."""

    enabled: bool = False
    max_frames: int = 10000
    capture_interval: int = 1
    auto_record: bool = False

    # Storage
    storage_directory: str = "data/recordings"
    storage_format: str = "msgpack"  # "msgpack" or "json"
    compression: bool = False

    # Replay
    default_speed: float = 1.0
    loop: bool = False

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> HistoryConfig:
        """Build from OmegaConf dict or plain dict."""
        if cfg is None:
            return cls()

        # Handle OmegaConf containers
        try:
            from omegaconf import OmegaConf

            if hasattr(cfg, "_metadata"):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass

        if not isinstance(cfg, dict):
            cfg = dict(cfg)

        storage = cfg.get("storage", {}) or {}
        replay = cfg.get("replay", {}) or {}

        return cls(
            enabled=bool(cfg.get("enabled", False)),
            max_frames=int(cfg.get("max_frames", 10000)),
            capture_interval=max(1, int(cfg.get("capture_interval", 1))),
            auto_record=bool(cfg.get("auto_record", False)),
            storage_directory=str(storage.get("directory", "data/recordings")),
            storage_format=str(storage.get("format", "msgpack")),
            compression=bool(storage.get("compression", False)),
            default_speed=float(replay.get("default_speed", 1.0)),
            loop=bool(replay.get("loop", False)),
        )
