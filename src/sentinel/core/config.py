"""Hierarchical YAML configuration system using OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class SentinelConfig:
    """Loads and merges YAML configuration files.

    Supports a base config with optional per-component overrides
    and CLI-level parameter overrides.
    """

    def __init__(self, config_path: str | Path = "config/default.yaml"):
        self._config_path = Path(config_path)
        self._config: DictConfig | None = None

    def load(self, validate: bool = False) -> DictConfig:
        """Load base config and merge any component-level overrides.

        Args:
            validate: If True, validate the loaded config against the
                Pydantic schema and raise ``pydantic.ValidationError``
                on invalid values.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")

        base = OmegaConf.load(self._config_path)
        assert isinstance(base, DictConfig)

        # Merge component configs if they exist alongside the base
        config_dir = self._config_path.parent
        for subdir in ("sensors", "tracking", "detection", "ui"):
            sub_path = config_dir / subdir
            if sub_path.is_dir():
                for yaml_file in sorted(sub_path.glob("*.yaml")):
                    override = OmegaConf.load(yaml_file)
                    base = OmegaConf.merge(base, override)

        # Optional Pydantic validation
        if validate or OmegaConf.select(base, "sentinel.system.validate_config", default=False):
            from sentinel.core.config_schema import validate_config

            validate_config(OmegaConf.to_container(base, resolve=True))

        self._config = base
        return self._config

    def override(self, dotpath: str, value: Any) -> None:
        """Override a config value using dot notation.

        Example: config.override("sentinel.sensors.camera.source", "video.mp4")
        """
        if self._config is None:
            raise RuntimeError("Config not loaded yet. Call load() first.")
        OmegaConf.update(self._config, dotpath, value)

    @property
    def cfg(self) -> DictConfig:
        if self._config is None:
            raise RuntimeError("Config not loaded yet. Call load() first.")
        return self._config
