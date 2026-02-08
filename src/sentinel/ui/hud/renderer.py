"""Main HUD compositor -- assembles all elements into final overlay."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.track import Track
from sentinel.ui.hud.elements import HUDElements
from sentinel.ui.hud.styles import HUDStyles


class HUDRenderer:
    """Military-style heads-up display compositor.

    Composites tracking data, detection boxes, status panels,
    and decorative elements onto camera frames.
    """

    def __init__(self, config: DictConfig):
        self._styles = HUDStyles.from_config(config)
        self._elements = HUDElements(self._styles)

    def render(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        detections: list[Detection],
        system_status: dict,
    ) -> np.ndarray:
        """Composite full HUD overlay onto camera frame.

        Args:
            frame: Raw camera frame (BGR).
            tracks: Current active tracks from TrackManager.
            detections: Raw detections from current frame.
            system_status: System metrics dict (fps, counts, etc.).

        Returns:
            Frame with HUD overlay composited.
        """
        display = frame.copy()

        # Background effects
        self._elements.draw_scanlines(display)

        # Track visualizations
        alive_tracks = [t for t in tracks if t.is_alive]
        for track in alive_tracks:
            self._elements.draw_track_box(display, track)
            self._elements.draw_velocity_vector(display, track)
            self._elements.draw_track_label(display, track)

        # Targeting reticle on highest-scoring confirmed track
        confirmed = [t for t in alive_tracks if t.state == TrackState.CONFIRMED]
        if confirmed:
            primary = max(confirmed, key=lambda t: t.score)
            self._elements.draw_reticle(display, primary)

        # Frame elements
        self._elements.draw_header_bar(display)
        self._elements.draw_crosshair_overlay(display)
        self._elements.draw_status_panel(display, system_status)
        self._elements.draw_track_list_panel(display, alive_tracks)
        self._elements.draw_border_frame(display)

        return display
