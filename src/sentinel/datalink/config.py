"""Data Link gateway configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataLinkConfig:
    """Data Link gateway configuration."""

    enabled: bool = False
    source_id: str = "SENTINEL-01"
    transport_type: str = "in_memory"
    publish_rate_hz: float = 1.0
    publish_iff: bool = True
    publish_engagement: bool = True
    validate_outbound: bool = True
    validate_inbound: bool = True
    max_track_numbers: int = 8192
    accept_inbound: bool = True
    merge_with_local: bool = False

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> DataLinkConfig:
        """Build from OmegaConf dict or plain dict."""
        if cfg is None:
            return cls()

        try:
            from omegaconf import OmegaConf
            if hasattr(cfg, '_metadata'):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass

        if not isinstance(cfg, dict):
            cfg = dict(cfg)

        return cls(
            enabled=bool(cfg.get("enabled", False)),
            source_id=str(cfg.get("source_id", "SENTINEL-01")),
            transport_type=str(cfg.get("transport_type", "in_memory")),
            publish_rate_hz=float(cfg.get("publish_rate_hz", 1.0)),
            publish_iff=bool(cfg.get("publish_iff", True)),
            publish_engagement=bool(cfg.get("publish_engagement", True)),
            validate_outbound=bool(cfg.get("validate_outbound", True)),
            validate_inbound=bool(cfg.get("validate_inbound", True)),
            max_track_numbers=int(cfg.get("max_track_numbers", 8192)),
            accept_inbound=bool(cfg.get("accept_inbound", True)),
            merge_with_local=bool(cfg.get("merge_with_local", False)),
        )
