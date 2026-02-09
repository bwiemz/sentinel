"""HUD color palette, fonts, and mil-spec styling constants."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
from omegaconf import DictConfig

from sentinel.core.types import TrackState

# BGR color type
Color = tuple[int, int, int]


@dataclass
class HUDStyles:
    """Military HUD visual styling configuration."""

    # Colors (BGR format for OpenCV)
    color_tentative: Color = (0, 255, 255)  # Yellow
    color_confirmed: Color = (0, 255, 0)  # Green
    color_coasting: Color = (0, 165, 255)  # Orange
    color_reticle: Color = (0, 255, 0)  # Green
    color_text: Color = (0, 255, 0)  # Green
    color_text_dim: Color = (0, 140, 0)  # Dim green
    color_border: Color = (0, 100, 0)  # Dark green
    color_danger: Color = (0, 0, 255)  # Red
    color_background: Color = (0, 20, 0)  # Dark green tint
    color_panel_bg: Color = (0, 15, 0)  # Panel background
    color_radar: Color = (255, 200, 0)  # Cyan for radar tracks
    color_fused: Color = (255, 255, 0)  # Bright cyan for fused
    color_thermal: Color = (0, 100, 255)  # Orange-red for thermal
    color_stealth: Color = (200, 0, 200)  # Magenta for stealth
    color_hypersonic: Color = (0, 0, 255)  # Red for hypersonic
    color_quantum: Color = (255, 100, 200)  # Pink-cyan for quantum radar
    color_threat_critical: Color = (0, 0, 255)  # Bright red
    color_threat_high: Color = (0, 80, 255)  # Orange-red
    color_threat_medium: Color = (0, 200, 255)  # Yellow
    color_threat_low: Color = (0, 200, 0)  # Green
    # IFF identification colors (BGR)
    color_iff_friendly: Color = (0, 255, 0)  # Green
    color_iff_assumed_friendly: Color = (0, 180, 0)  # Dim green
    color_iff_hostile: Color = (0, 0, 255)  # Red
    color_iff_assumed_hostile: Color = (0, 80, 255)  # Orange-red
    color_iff_unknown: Color = (0, 255, 255)  # Yellow
    color_iff_pending: Color = (180, 180, 180)  # Grey
    color_iff_spoof: Color = (200, 0, 200)  # Magenta
    # Engagement authorization colors (BGR)
    color_auth_weapons_free: Color = (0, 0, 255)  # Red
    color_auth_weapons_tight: Color = (0, 165, 255)  # Orange
    color_auth_weapons_hold: Color = (0, 255, 255)  # Yellow
    color_auth_hold_fire: Color = (0, 255, 0)  # Green

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
            color_radar=tuple(colors.get("radar", [255, 200, 0])),
            color_fused=tuple(colors.get("fused", [255, 255, 0])),
            color_thermal=tuple(colors.get("thermal", [0, 100, 255])),
            color_stealth=tuple(colors.get("stealth", [200, 0, 200])),
            color_hypersonic=tuple(colors.get("hypersonic", [0, 0, 255])),
            color_quantum=tuple(colors.get("quantum", [255, 100, 200])),
            color_threat_critical=tuple(colors.get("threat_critical", [0, 0, 255])),
            color_threat_high=tuple(colors.get("threat_high", [0, 80, 255])),
            color_threat_medium=tuple(colors.get("threat_medium", [0, 200, 255])),
            color_threat_low=tuple(colors.get("threat_low", [0, 200, 0])),
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

    def color_for_threat(self, threat_level: str) -> Color:
        """Get the color associated with a threat level."""
        return {
            "CRITICAL": self.color_threat_critical,
            "HIGH": self.color_threat_high,
            "MEDIUM": self.color_threat_medium,
            "LOW": self.color_threat_low,
        }.get(threat_level, self.color_text)

    def color_for_iff(self, iff_code: str) -> Color:
        """Get the color associated with an IFF identification code."""
        return {
            "friendly": self.color_iff_friendly,
            "assumed_friendly": self.color_iff_assumed_friendly,
            "hostile": self.color_iff_hostile,
            "assumed_hostile": self.color_iff_assumed_hostile,
            "unknown": self.color_iff_unknown,
            "pending": self.color_iff_pending,
            "spoof_suspect": self.color_iff_spoof,
        }.get(iff_code, self.color_iff_unknown)

    def color_for_engagement(self, auth: str) -> Color:
        """Get the color associated with an engagement authorization."""
        return {
            "weapons_free": self.color_auth_weapons_free,
            "weapons_tight": self.color_auth_weapons_tight,
            "weapons_hold": self.color_auth_weapons_hold,
            "hold_fire": self.color_auth_hold_fire,
        }.get(auth, self.color_text)
