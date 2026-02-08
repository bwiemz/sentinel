"""HUD color palette, fonts, and mil-spec styling constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import TrackState

# BGR color type
Color = Tuple[int, int, int]


@dataclass
class HUDStyles:
    """Military HUD visual styling configuration."""

    # Colors (BGR format for OpenCV)
    color_tentative: Color = (0, 255, 255)    # Yellow
    color_confirmed: Color = (0, 255, 0)      # Green
    color_coasting: Color = (0, 165, 255)     # Orange
    color_reticle: Color = (0, 255, 0)        # Green
    color_text: Color = (0, 255, 0)           # Green
    color_text_dim: Color = (0, 140, 0)       # Dim green
    color_border: Color = (0, 100, 0)         # Dark green
    color_danger: Color = (0, 0, 255)         # Red
    color_background: Color = (0, 20, 0)      # Dark green tint
    color_panel_bg: Color = (0, 15, 0)        # Panel background

    # Font
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.45
    font_scale_large: float = 0.55
    font_scale_small: float = 0.35
    font_thickness: int = 1

    # Overlay
    overlay_alpha: float = 0.85
    scanline_enabled: bool = True
    scanline_spacing: int = 3
    scanline_alpha: float = 0.03

    @classmethod
    def from_config(cls, config: DictConfig) -> HUDStyles:
        """Create styles from HUD config section."""
        colors = config.get("colors", {})
        return cls(
            color_tentative=tuple(colors.get("tentative", [0, 255, 255])),
            color_confirmed=tuple(colors.get("confirmed", [0, 255, 0])),
            color_coasting=tuple(colors.get("coasting", [0, 165, 255])),
            color_reticle=tuple(colors.get("reticle", [0, 255, 0])),
            color_text=tuple(colors.get("text", [0, 255, 0])),
            color_border=tuple(colors.get("border", [0, 100, 0])),
            color_danger=tuple(colors.get("danger", [0, 0, 255])),
            color_background=tuple(colors.get("background", [0, 20, 0])),
            overlay_alpha=config.get("overlay_alpha", 0.85),
            scanline_enabled=config.get("scanline", {}).get("enabled", True),
            scanline_spacing=config.get("scanline", {}).get("spacing", 3),
            scanline_alpha=config.get("scanline", {}).get("alpha", 0.03),
        )

    def color_for_state(self, state: TrackState) -> Color:
        """Get the color associated with a track state."""
        return {
            TrackState.TENTATIVE: self.color_tentative,
            TrackState.CONFIRMED: self.color_confirmed,
            TrackState.COASTING: self.color_coasting,
            TrackState.DELETED: self.color_danger,
        }.get(state, self.color_text)
