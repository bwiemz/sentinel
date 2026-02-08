"""HUD drawing primitives -- individual visual elements."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from sentinel.core.types import TrackState
from sentinel.tracking.track import Track
from sentinel.ui.hud.styles import HUDStyles


class HUDElements:
    """Individual HUD drawing primitives for military-style overlay."""

    def __init__(self, styles: HUDStyles):
        self.s = styles

    # === RETICLE ===

    def draw_reticle(self, frame: np.ndarray, track: Track) -> None:
        """Draw targeting reticle on the primary (highest-score) track.

        Concentric circles + crosshairs with gap + corner ticks.
        """
        cx, cy = int(track.position[0]), int(track.position[1])
        color = self.s.color_reticle

        # Outer circle (pulsing)
        pulse = 0.85 + 0.15 * math.sin(time.monotonic() * 4)
        r_outer = int(45 * pulse)
        r_inner = int(22 * pulse)

        cv2.circle(frame, (cx, cy), r_outer, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r_inner, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)  # Center dot

        # Crosshairs with gap
        gap = 12
        length = 55
        cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, 1, cv2.LINE_AA)

        # Diagonal tick marks
        d = int(35 * pulse)
        tick = 8
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x1 = cx + dx * d
            y1 = cy + dy * d
            x2 = cx + dx * (d + tick)
            y2 = cy + dy * (d + tick)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # === TRACK BOX ===

    def draw_track_box(self, frame: np.ndarray, track: Track) -> None:
        """Draw bounding box with corner brackets (mil-spec style)."""
        bbox = track.predicted_bbox
        if bbox is None:
            return

        x1, y1, x2, y2 = bbox.astype(int)
        color = self.s.color_for_state(track.state)
        bracket_len = max(12, int(min(x2 - x1, y2 - y1) * 0.2))

        # Corner brackets
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 1, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 1, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 1, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 1, cv2.LINE_AA)

    def draw_track_label(self, frame: np.ndarray, track: Track) -> None:
        """Draw track ID, class, and confidence above the bounding box."""
        bbox = track.predicted_bbox
        if bbox is None:
            return

        x1, y1 = int(bbox[0]), int(bbox[1])
        color = self.s.color_for_state(track.state)

        # Track ID + class
        cls_name = track.dominant_class or "UNK"
        conf = track.last_detection.confidence if track.last_detection else 0
        label = f"T:{track.track_id} {cls_name}"
        sublabel = f"{conf:.0%} | SCR:{track.score:.2f}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, self.s.font_face, self.s.font_scale, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - th * 2 - 12),
            (x1 + tw + 6, y1 - 2),
            self.s.color_panel_bg,
            -1,
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - th - 6),
            self.s.font_face,
            self.s.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            sublabel,
            (x1 + 3, y1 - 4),
            self.s.font_face,
            self.s.font_scale_small,
            self.s.color_text_dim,
            1,
            cv2.LINE_AA,
        )

    # === VELOCITY VECTOR ===

    def draw_velocity_vector(self, frame: np.ndarray, track: Track) -> None:
        """Draw velocity arrow from track center."""
        if track.state == TrackState.TENTATIVE:
            return  # Don't draw velocity for unconfirmed tracks

        cx, cy = int(track.position[0]), int(track.position[1])
        vx, vy = track.velocity
        speed = np.sqrt(vx**2 + vy**2)

        if speed < 0.5:  # Below threshold, skip
            return

        # Scale velocity for display (cap at 80 pixels)
        scale = min(80 / max(speed, 1e-6), 5.0)
        end_x = int(cx + vx * scale)
        end_y = int(cy + vy * scale)

        color = self.s.color_for_state(track.state)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 1, cv2.LINE_AA, tipLength=0.3)

    # === STATUS PANELS ===

    def draw_status_panel(self, frame: np.ndarray, status: dict) -> None:
        """Draw system status readout panel (top-left corner)."""
        h, w = frame.shape[:2]
        panel_w = 220
        panel_h = 90

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (4, 30), (panel_w, 30 + panel_h), self.s.color_panel_bg, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Panel border
        cv2.rectangle(frame, (4, 30), (panel_w, 30 + panel_h), self.s.color_border, 1)

        # Status text
        y = 48
        line_h = 16
        items = [
            f"FPS: {status.get('fps', 0):.1f}",
            f"TRACKS: {status.get('track_count', 0)} / {status.get('confirmed_count', 0)} CONF",
            f"DETECTIONS: {status.get('detection_count', 0)}",
            f"UPTIME: {status.get('uptime', 0):.0f}s",
        ]
        for text in items:
            cv2.putText(
                frame,
                text,
                (10, y),
                self.s.font_face,
                self.s.font_scale_small,
                self.s.color_text,
                1,
                cv2.LINE_AA,
            )
            y += line_h

    def draw_track_list_panel(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Draw confirmed track list panel (bottom-left)."""
        h, w = frame.shape[:2]
        confirmed = [t for t in tracks if t.state == TrackState.CONFIRMED]
        if not confirmed:
            return

        panel_w = 250
        line_h = 16
        panel_h = min(len(confirmed) * line_h + 24, 200)
        y_start = h - panel_h - 8

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (4, y_start), (panel_w, h - 8), self.s.color_panel_bg, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (4, y_start), (panel_w, h - 8), self.s.color_border, 1)

        # Header
        cv2.putText(
            frame,
            "ACTIVE TRACKS",
            (10, y_start + 14),
            self.s.font_face,
            self.s.font_scale_small,
            self.s.color_text,
            1,
            cv2.LINE_AA,
        )

        # Track entries
        y = y_start + 30
        for track in sorted(confirmed, key=lambda t: t.score, reverse=True)[:8]:
            cls = track.dominant_class or "UNK"
            text = f"{track.track_id} | {cls:8s} | SCR:{track.score:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y),
                self.s.font_face,
                self.s.font_scale_small,
                self.s.color_text_dim,
                1,
                cv2.LINE_AA,
            )
            y += line_h

    # === FRAME ELEMENTS ===

    def draw_header_bar(self, frame: np.ndarray, title: str = "SENTINEL v0.1") -> None:
        """Draw top header bar with system name."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 26), self.s.color_panel_bg, -1)
        cv2.line(frame, (0, 26), (w, 26), self.s.color_border, 1)

        cv2.putText(
            frame,
            title,
            (8, 18),
            self.s.font_face,
            self.s.font_scale_large,
            self.s.color_text,
            1,
            cv2.LINE_AA,
        )

        # Timestamp (right side)
        ts = time.strftime("%H:%M:%S", time.localtime())
        (tw, _), _ = cv2.getTextSize(ts, self.s.font_face, self.s.font_scale, 1)
        cv2.putText(
            frame,
            ts,
            (w - tw - 10, 18),
            self.s.font_face,
            self.s.font_scale,
            self.s.color_text_dim,
            1,
            cv2.LINE_AA,
        )

    def draw_border_frame(self, frame: np.ndarray) -> None:
        """Draw outer border frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (1, 1), (w - 2, h - 2), self.s.color_border, 1)

    def draw_scanlines(self, frame: np.ndarray) -> None:
        """Apply subtle CRT scanline effect."""
        if not self.s.scanline_enabled:
            return
        h, w = frame.shape[:2]
        scanline_overlay = np.zeros_like(frame)
        for y in range(0, h, self.s.scanline_spacing):
            scanline_overlay[y, :] = (20, 20, 20)
        cv2.addWeighted(frame, 1.0, scanline_overlay, self.s.scanline_alpha, 0, frame)

    def draw_crosshair_overlay(self, frame: np.ndarray) -> None:
        """Draw subtle center crosshair for the full frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = self.s.color_border

        # Very subtle center cross
        cv2.line(frame, (cx - 20, cy), (cx - 5, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 5, cy), (cx + 20, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 20), (cx, cy - 5), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 5), (cx, cy + 20), color, 1, cv2.LINE_AA)

    # === RADAR ELEMENTS ===

    def draw_radar_blip(
        self,
        frame: np.ndarray,
        azimuth_deg: float,
        range_m: float,
        track_id: str,
        image_width: int,
        camera_hfov_deg: float = 60.0,
        is_fused: bool = False,
    ) -> None:
        """Draw a radar track as a diamond blip with range/azimuth label.

        Maps radar azimuth to horizontal pixel position.
        """
        h, w = frame.shape[:2]
        # Map azimuth to pixel x
        px_x = int((azimuth_deg / camera_hfov_deg + 0.5) * image_width)
        px_x = max(0, min(px_x, w - 1))
        py = h - 50  # Bottom region

        color = self.s.color_fused if is_fused else self.s.color_radar
        size = 8

        # Diamond shape
        pts = np.array(
            [
                [px_x, py - size],
                [px_x + size, py],
                [px_x, py + size],
                [px_x - size, py],
            ],
            np.int32,
        )
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

        if is_fused:
            cv2.fillPoly(frame, [pts], color)

        # Label
        label = f"R:{range_m / 1000:.1f}km"
        cv2.putText(
            frame,
            label,
            (px_x + 12, py + 4),
            self.s.font_face,
            self.s.font_scale_small,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            track_id,
            (px_x + 12, py - 8),
            self.s.font_face,
            self.s.font_scale_small,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_radar_status_line(self, frame: np.ndarray, status: dict) -> None:
        """Draw radar status in the status panel area."""
        if not status.get("radar_connected", False):
            return
        text = f"RDR: {status.get('radar_track_count', 0)} TRK | {status.get('radar_detection_count', 0)} DET"
        cv2.putText(
            frame,
            text,
            (10, 126),
            self.s.font_face,
            self.s.font_scale_small,
            self.s.color_radar,
            1,
            cv2.LINE_AA,
        )
        fused = status.get("fused_track_count", 0)
        if fused > 0:
            cv2.putText(
                frame,
                f"FUSED: {fused}",
                (10, 142),
                self.s.font_face,
                self.s.font_scale_small,
                self.s.color_fused,
                1,
                cv2.LINE_AA,
            )

    def draw_fusion_indicator(self, frame: np.ndarray, track: Track) -> None:
        """Draw 'F' badge on a camera track that has been fused with radar."""
        bbox = track.predicted_bbox
        if bbox is None:
            return
        x2, y1 = int(bbox[2]), int(bbox[1])
        cv2.putText(
            frame,
            "F",
            (x2 + 4, y1 + 12),
            self.s.font_face,
            self.s.font_scale,
            self.s.color_fused,
            1,
            cv2.LINE_AA,
        )

    # === THERMAL ELEMENTS ===

    def draw_thermal_blip(
        self,
        frame: np.ndarray,
        azimuth_deg: float,
        temperature_k: float,
        track_id: str,
        image_width: int,
        camera_hfov_deg: float = 60.0,
        is_fused: bool = False,
    ) -> None:
        """Draw a thermal track as a triangle blip with temp label.

        Maps thermal azimuth to horizontal pixel position.
        """
        h, w = frame.shape[:2]
        px_x = int((azimuth_deg / camera_hfov_deg + 0.5) * image_width)
        px_x = max(0, min(px_x, w - 1))
        py = h - 80  # Above radar blip region

        color = self.s.color_fused if is_fused else self.s.color_thermal
        size = 7

        # Triangle shape (pointing up)
        pts = np.array(
            [
                [px_x, py - size],
                [px_x + size, py + size],
                [px_x - size, py + size],
            ],
            np.int32,
        )
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
        if is_fused:
            cv2.fillPoly(frame, [pts], color)

        # Temperature label
        label = f"{temperature_k:.0f}K"
        cv2.putText(
            frame,
            label,
            (px_x + 10, py + 4),
            self.s.font_face,
            self.s.font_scale_small,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_threat_indicator(
        self,
        frame: np.ndarray,
        fused_track,
        image_width: int,
        camera_hfov_deg: float = 60.0,
    ) -> None:
        """Draw threat level badge near the fused track position."""
        threat = getattr(fused_track, "threat_level", "UNKNOWN")
        if threat == "UNKNOWN":
            return

        color = self.s.color_for_threat(threat)

        # Determine position from best available source
        px_x, py = None, None
        if fused_track.camera_track is not None:
            bbox = fused_track.camera_track.predicted_bbox
            if bbox is not None:
                px_x = int(bbox[2]) + 4
                py = int(bbox[1]) + 24
        if px_x is None:
            az = getattr(fused_track, "azimuth_deg", None)
            if az is not None:
                px_x = int((az / camera_hfov_deg + 0.5) * image_width)
                px_x = max(0, min(px_x, frame.shape[1] - 1))
                py = frame.shape[0] - 95
            else:
                return

        label = f"THR:{threat}"
        (tw, th), _ = cv2.getTextSize(label, self.s.font_face, self.s.font_scale_small, 1)
        cv2.rectangle(frame, (px_x - 1, py - th - 2), (px_x + tw + 2, py + 2), self.s.color_panel_bg, -1)
        cv2.putText(
            frame,
            label,
            (px_x, py),
            self.s.font_face,
            self.s.font_scale_small,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_stealth_alert(self, frame: np.ndarray) -> None:
        """Draw STEALTH DETECTED alert banner."""
        h, w = frame.shape[:2]
        label = "! STEALTH DETECTED !"
        (tw, th), _ = cv2.getTextSize(label, self.s.font_face, self.s.font_scale_large, 1)
        cx = (w - tw) // 2
        cy = 55

        # Flashing background
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6)
        bg_alpha = 0.4 + 0.3 * pulse
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - 8, cy - th - 4), (cx + tw + 8, cy + 4), self.s.color_stealth, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        cv2.putText(
            frame,
            label,
            (cx, cy),
            self.s.font_face,
            self.s.font_scale_large,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_hypersonic_alert(self, frame: np.ndarray) -> None:
        """Draw HYPERSONIC THREAT alert banner."""
        h, w = frame.shape[:2]
        label = "!! HYPERSONIC THREAT !!"
        (tw, th), _ = cv2.getTextSize(label, self.s.font_face, self.s.font_scale_large, 1)
        cx = (w - tw) // 2
        cy = 75

        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 8)
        bg_alpha = 0.5 + 0.3 * pulse
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - 8, cy - th - 4), (cx + tw + 8, cy + 4), self.s.color_hypersonic, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        cv2.putText(
            frame,
            label,
            (cx, cy),
            self.s.font_face,
            self.s.font_scale_large,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_thermal_status_line(self, frame: np.ndarray, status: dict) -> None:
        """Draw thermal sensor status in the status panel area."""
        if not status.get("thermal_connected", False):
            return
        y = 158 if status.get("radar_connected", False) else 126
        text = f"THM: {status.get('thermal_track_count', 0)} TRK"
        cv2.putText(
            frame,
            text,
            (10, y),
            self.s.font_face,
            self.s.font_scale_small,
            self.s.color_thermal,
            1,
            cv2.LINE_AA,
        )
        # Threat counts
        threats = status.get("threat_counts", {})
        if threats:
            parts = []
            for level in ["CRITICAL", "HIGH", "MEDIUM"]:
                count = threats.get(level, 0)
                if count > 0:
                    parts.append(f"{level[0]}:{count}")
            if parts:
                cv2.putText(
                    frame,
                    "THR:" + " ".join(parts),
                    (10, y + 14),
                    self.s.font_face,
                    self.s.font_scale_small,
                    self.s.color_threat_high,
                    1,
                    cv2.LINE_AA,
                )

    def draw_quantum_status_line(self, frame: np.ndarray, status: dict) -> None:
        """Draw quantum radar status in the status panel area."""
        if not status.get("quantum_radar_connected", False):
            return
        # Position below thermal status
        y = 158
        if status.get("radar_connected", False):
            y += 14
        if status.get("thermal_connected", False):
            y += 28
        text = f"QI: {status.get('quantum_radar_track_count', 0)} TRK"
        cv2.putText(
            frame,
            text,
            (10, y),
            self.s.font_face,
            self.s.font_scale_small,
            self.s.color_quantum,
            1,
            cv2.LINE_AA,
        )
