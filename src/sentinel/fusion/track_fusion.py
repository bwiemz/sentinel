"""Track-level fusion -- correlates camera and radar track lists.

Uses angular correspondence between radar azimuth and camera pixel
position (mapped to azimuth via a simple pinhole model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import SensorType, generate_track_id
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)

# Large cost for infeasible correlation
_INFEASIBLE = 1e5


@dataclass
class FusedTrack:
    """A fused track combining camera and radar information."""

    fused_id: str
    camera_track: Optional[Track] = None
    radar_track: Optional[RadarTrack] = None
    position_px: Optional[np.ndarray] = None
    position_m: Optional[np.ndarray] = None
    range_m: Optional[float] = None
    azimuth_deg: Optional[float] = None
    velocity_mps: Optional[float] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    sensor_sources: set[SensorType] = field(default_factory=set)
    fusion_quality: float = 0.0

    @property
    def is_dual_sensor(self) -> bool:
        """Whether this track has both camera and radar contributions."""
        return self.camera_track is not None and self.radar_track is not None

    def to_dict(self) -> dict:
        return {
            "fused_id": self.fused_id,
            "dual_sensor": self.is_dual_sensor,
            "position_px": self.position_px.tolist() if self.position_px is not None else None,
            "position_m": self.position_m.tolist() if self.position_m is not None else None,
            "range_m": self.range_m,
            "azimuth_deg": self.azimuth_deg,
            "velocity_mps": self.velocity_mps,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "sources": [s.value for s in self.sensor_sources],
            "fusion_quality": round(self.fusion_quality, 3),
        }


class TrackFusion:
    """Correlates camera and radar tracks into fused track list.

    Uses angular correspondence: radar azimuth is correlated with the
    horizontal pixel position of camera tracks, mapped to azimuth via
    a simple pinhole approximation.

    Args:
        camera_hfov_deg: Camera horizontal field of view in degrees.
        image_width_px: Camera image width in pixels.
        azimuth_gate_deg: Maximum azimuth difference for correlation.
    """

    def __init__(
        self,
        camera_hfov_deg: float = 60.0,
        image_width_px: int = 1280,
        azimuth_gate_deg: float = 5.0,
    ):
        self._hfov_deg = camera_hfov_deg
        self._img_width = image_width_px
        self._gate_deg = azimuth_gate_deg

    def fuse(
        self,
        camera_tracks: list[Track],
        radar_tracks: list[RadarTrack],
    ) -> list[FusedTrack]:
        """Correlate camera and radar tracks into fused output.

        1. Build angular cost matrix.
        2. Hungarian assignment with azimuth gating.
        3. Matched -> dual-sensor FusedTrack.
        4. Unmatched camera -> camera-only FusedTrack.
        5. Unmatched radar -> radar-only FusedTrack.
        """
        if not camera_tracks and not radar_tracks:
            return []

        fused: list[FusedTrack] = []

        if camera_tracks and radar_tracks:
            cost = self._build_correlation_cost(camera_tracks, radar_tracks)
            row_indices, col_indices = linear_sum_assignment(cost)

            matched_cam: set[int] = set()
            matched_rdr: set[int] = set()

            for r, c in zip(row_indices, col_indices):
                if cost[r, c] < _INFEASIBLE:
                    fused.append(self._make_dual(camera_tracks[r], radar_tracks[c]))
                    matched_cam.add(r)
                    matched_rdr.add(c)

            # Unmatched camera
            for i, ct in enumerate(camera_tracks):
                if i not in matched_cam:
                    fused.append(self._make_camera_only(ct))

            # Unmatched radar
            for j, rt in enumerate(radar_tracks):
                if j not in matched_rdr:
                    fused.append(self._make_radar_only(rt))
        elif camera_tracks:
            for ct in camera_tracks:
                fused.append(self._make_camera_only(ct))
        else:
            for rt in radar_tracks:
                fused.append(self._make_radar_only(rt))

        return fused

    def pixel_to_azimuth(self, px_x: float) -> float:
        """Convert horizontal pixel position to approximate azimuth (degrees).

        Pinhole model: azimuth = (px_x / width - 0.5) * hfov
        Center pixel -> 0 degrees, left edge -> -hfov/2, right -> +hfov/2.
        """
        return (px_x / self._img_width - 0.5) * self._hfov_deg

    def _angular_distance(self, az1_deg: float, az2_deg: float) -> float:
        """Absolute angular difference in degrees, handling wraparound."""
        diff = abs(az1_deg - az2_deg) % 360.0
        return min(diff, 360.0 - diff)

    def _build_correlation_cost(
        self,
        camera_tracks: list[Track],
        radar_tracks: list[RadarTrack],
    ) -> np.ndarray:
        """Build cost matrix based on angular distance."""
        N = len(camera_tracks)
        M = len(radar_tracks)
        cost = np.full((N, M), _INFEASIBLE)

        for i, ct in enumerate(camera_tracks):
            cam_az = self.pixel_to_azimuth(ct.position[0])
            for j, rt in enumerate(radar_tracks):
                ang_dist = self._angular_distance(cam_az, rt.azimuth_deg)
                if ang_dist <= self._gate_deg:
                    cost[i, j] = ang_dist

        return cost

    def _make_dual(self, cam: Track, rdr: RadarTrack) -> FusedTrack:
        vel = rdr.velocity
        return FusedTrack(
            fused_id=generate_track_id(),
            camera_track=cam,
            radar_track=rdr,
            position_px=cam.position,
            position_m=rdr.position,
            range_m=rdr.range_m,
            azimuth_deg=rdr.azimuth_deg,
            velocity_mps=float(np.linalg.norm(vel)),
            class_name=cam.dominant_class,
            confidence=cam.last_detection.confidence if cam.last_detection else None,
            sensor_sources={SensorType.CAMERA, SensorType.RADAR},
            fusion_quality=min(cam.score, rdr.score),
        )

    def _make_camera_only(self, cam: Track) -> FusedTrack:
        return FusedTrack(
            fused_id=generate_track_id(),
            camera_track=cam,
            position_px=cam.position,
            class_name=cam.dominant_class,
            confidence=cam.last_detection.confidence if cam.last_detection else None,
            sensor_sources={SensorType.CAMERA},
            fusion_quality=cam.score * 0.5,
        )

    def _make_radar_only(self, rdr: RadarTrack) -> FusedTrack:
        vel = rdr.velocity
        return FusedTrack(
            fused_id=generate_track_id(),
            radar_track=rdr,
            position_m=rdr.position,
            range_m=rdr.range_m,
            azimuth_deg=rdr.azimuth_deg,
            velocity_mps=float(np.linalg.norm(vel)),
            sensor_sources={SensorType.RADAR},
            fusion_quality=rdr.score * 0.5,
        )
