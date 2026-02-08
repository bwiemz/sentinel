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
    radar blips, and decorative elements onto camera frames.
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
        radar_tracks: Optional[list] = None,
        fused_tracks: Optional[list] = None,
    ) -> np.ndarray:
        """Composite full HUD overlay onto camera frame.

        Args:
            frame: Raw camera frame (BGR).
            tracks: Current active camera tracks from TrackManager.
            detections: Raw detections from current frame.
            system_status: System metrics dict (fps, counts, etc.).
            radar_tracks: Optional list of RadarTrack objects.
            fused_tracks: Optional list of FusedTrack objects.

        Returns:
            Frame with HUD overlay composited.
        """
        display = frame.copy()

        # Background effects
        self._elements.draw_scanlines(display)

        # Camera track visualizations
        alive_tracks = [t for t in tracks if t.is_alive]

        # Build set of camera track IDs that are fused (dual-sensor)
        fused_cam_ids: set[str] = set()
        if fused_tracks:
            for ft in fused_tracks:
                if ft.is_dual_sensor and ft.camera_track is not None:
                    fused_cam_ids.add(ft.camera_track.track_id)

        for track in alive_tracks:
            self._elements.draw_track_box(display, track)
            self._elements.draw_velocity_vector(display, track)
            self._elements.draw_track_label(display, track)
            if track.track_id in fused_cam_ids:
                self._elements.draw_fusion_indicator(display, track)

        # Targeting reticle on highest-scoring confirmed track
        confirmed = [t for t in alive_tracks if t.state == TrackState.CONFIRMED]
        if confirmed:
            primary = max(confirmed, key=lambda t: t.score)
            self._elements.draw_reticle(display, primary)

        # Radar blips
        if fused_tracks:
            h, w = display.shape[:2]
            for ft in fused_tracks:
                if ft.radar_track is not None:
                    self._elements.draw_radar_blip(
                        display,
                        azimuth_deg=ft.radar_track.azimuth_deg,
                        range_m=ft.radar_track.range_m,
                        track_id=ft.radar_track.track_id,
                        image_width=w,
                        is_fused=ft.is_dual_sensor,
                    )
        elif radar_tracks:
            h, w = display.shape[:2]
            for rt in radar_tracks:
                if rt.is_alive:
                    self._elements.draw_radar_blip(
                        display,
                        azimuth_deg=rt.azimuth_deg,
                        range_m=rt.range_m,
                        track_id=rt.track_id,
                        image_width=w,
                    )

        # Frame elements
        self._elements.draw_header_bar(display)
        self._elements.draw_crosshair_overlay(display)
        self._elements.draw_status_panel(display, system_status)
        if radar_tracks is not None or fused_tracks is not None:
            self._elements.draw_radar_status_line(display, system_status)
        self._elements.draw_track_list_panel(display, alive_tracks)
        self._elements.draw_border_frame(display)

        return display
