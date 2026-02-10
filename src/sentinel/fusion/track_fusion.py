"""Track-level fusion -- correlates camera and radar track lists.

Uses angular correspondence between radar azimuth and camera pixel
position (mapped to azimuth via a simple pinhole model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import SensorType, generate_track_id
from sentinel.fusion.state_fusion import covariance_intersection
from sentinel.fusion.temporal_alignment import align_tracks_to_epoch, _extract_position_cov
from sentinel.tracking.cost_functions import track_to_track_mahalanobis
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)

# Large cost for infeasible correlation
_INFEASIBLE = 1e5


@dataclass
class FusedTrack:
    """A fused track combining camera and radar information."""

    fused_id: str
    camera_track: Track | None = None
    radar_track: RadarTrack | None = None
    position_px: np.ndarray | None = None
    position_m: np.ndarray | None = None
    range_m: float | None = None
    azimuth_deg: float | None = None
    velocity_mps: float | None = None
    class_name: str | None = None
    confidence: float | None = None
    sensor_sources: set[SensorType] = field(default_factory=set)
    fusion_quality: float = 0.0
    fused_state: np.ndarray | None = None
    fused_covariance: np.ndarray | None = None
    position_geo: dict | None = None

    @property
    def is_dual_sensor(self) -> bool:
        """Whether this track has both camera and radar contributions."""
        return self.camera_track is not None and self.radar_track is not None

    def to_dict(self) -> dict:
        d = {
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
        if self.position_geo is not None:
            d["position_geo"] = self.position_geo
        return d


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
        use_ci_fusion: bool = False,
        use_temporal_alignment: bool = False,
        use_statistical_distance: bool = False,
        statistical_distance_gate: float = 9.21,
    ):
        self._hfov_deg = camera_hfov_deg
        self._img_width = image_width_px
        self._gate_deg = azimuth_gate_deg
        self._use_ci = use_ci_fusion
        self._use_temporal_alignment = use_temporal_alignment
        self._use_statistical_distance = use_statistical_distance
        self._stat_gate = statistical_distance_gate

        # Persistent fused track ID mapping: (cam_id, rdr_id) -> fused_id
        self._fused_id_map: dict[tuple[str, str], str] = {}

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
            if self._use_statistical_distance:
                cost = self._build_statistical_correlation_cost(camera_tracks, radar_tracks)
            else:
                cost = self._build_correlation_cost(camera_tracks, radar_tracks)
            row_indices, col_indices = linear_sum_assignment(cost)

            matched_cam: set[int] = set()
            matched_rdr: set[int] = set()

            for r, c in zip(row_indices, col_indices, strict=False):
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

    def _build_statistical_correlation_cost(
        self,
        camera_tracks: list[Track],
        radar_tracks: list[RadarTrack],
    ) -> np.ndarray:
        """Build cost matrix using track-to-track Mahalanobis distance.

        Projects camera pixel positions to world coordinates using radar
        range estimates, then computes Mahalanobis distance in world frame.
        Optionally aligns tracks to a common reference epoch first.
        Uses angular gating as a fast pre-filter.
        """
        N = len(camera_tracks)
        M = len(radar_tracks)
        cost = np.full((N, M), _INFEASIBLE)

        # Optionally align radar tracks to common epoch
        if self._use_temporal_alignment:
            all_times = [t.last_update_time for t in camera_tracks] + \
                        [t.last_update_time for t in radar_tracks]
            ref_time = max(all_times) if all_times else 0.0
            rdr_aligned = align_tracks_to_epoch(radar_tracks, ref_time)
        else:
            rdr_aligned = None

        # Pre-extract radar positions/covariances once (avoid M extractions per camera track)
        rdr_data: list[tuple[np.ndarray, np.ndarray]] = []
        for j, rt in enumerate(radar_tracks):
            if rdr_aligned is not None:
                rdr_data.append((rdr_aligned[j].position, rdr_aligned[j].covariance))
            else:
                is_ca = rt.ekf.dim_state == 6 and not getattr(rt, '_use_3d', False)
                rdr_data.append(_extract_position_cov(rt.ekf.x, rt.ekf.P, is_ca=is_ca))

        hfov_rad = np.radians(self._hfov_deg)

        for i, ct in enumerate(camera_tracks):
            cam_az_deg = self.pixel_to_azimuth(ct.position[0])
            cam_az_rad = np.radians(cam_az_deg)
            cam_P00 = ct.kf.P[0, 0]

            for j, rt in enumerate(radar_tracks):
                ang_dist = self._angular_distance(cam_az_deg, rt.azimuth_deg)
                if ang_dist > self._gate_deg:
                    continue

                # Project camera bearing to world frame at radar range
                cam_world = np.array([
                    rt.range_m * np.cos(cam_az_rad),
                    rt.range_m * np.sin(cam_az_rad),
                ])
                # Camera covariance in world: scale pixel uncertainty to meters
                px_to_m = rt.range_m * hfov_rad / self._img_width
                cam_cov = np.eye(2) * (px_to_m ** 2) * cam_P00

                pos_r, cov_r = rdr_data[j]
                d2 = track_to_track_mahalanobis(cam_world, cam_cov, pos_r, cov_r)
                if d2 <= self._stat_gate:
                    cost[i, j] = d2

        return cost

    @staticmethod
    def _extract_position_geo(rdr: RadarTrack) -> dict | None:
        """Extract geodetic position dict from a radar track, or None."""
        geo = rdr.position_geo
        if geo is None:
            return None
        return {
            "lat": round(geo[0], 7),
            "lon": round(geo[1], 7),
            "alt": round(geo[2], 2),
        }

    def _make_dual(self, cam: Track, rdr: RadarTrack) -> FusedTrack:
        vel = rdr.velocity

        # Persistent fused ID
        key = (cam.track_id, rdr.track_id)
        if key not in self._fused_id_map:
            self._fused_id_map[key] = generate_track_id()
        fused_id = self._fused_id_map[key]

        # State-level CI fusion (if enabled)
        fused_state = None
        fused_cov = None
        if self._use_ci:
            fused_state, fused_cov = self._ci_fuse_states(cam, rdr)

        return FusedTrack(
            fused_id=fused_id,
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
            fused_state=fused_state,
            fused_covariance=fused_cov,
            position_geo=self._extract_position_geo(rdr),
        )

    def _ci_fuse_states(
        self,
        cam: Track,
        rdr: RadarTrack,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse camera and radar states via Covariance Intersection.

        Projects camera pixel position into world frame using radar range,
        then fuses the 2D position estimates.
        """
        # Camera state: position in pixels [cx, cy]
        cam_pos = cam.position  # [x_px, y_px]

        # Project camera to world frame using radar range and azimuth
        # Approximate: camera bearing -> world position at radar range
        cam_az_deg = self.pixel_to_azimuth(cam_pos[0])
        cam_az_rad = np.radians(cam_az_deg)
        cam_world = np.array(
            [
                rdr.range_m * np.cos(cam_az_rad),
                rdr.range_m * np.sin(cam_az_rad),
            ]
        )

        # Camera covariance in world frame (large -- pixel-based uncertainty)
        # Scale pixel covariance to world: rough projection
        pixel_to_meter = rdr.range_m * np.radians(self._hfov_deg) / self._img_width
        cam_P_world = np.eye(2) * (pixel_to_meter**2) * cam.kf.P[0, 0]

        # Radar position and 2D position covariance
        # Detect CA layout: 6D state without 3D flag means [x,vx,ax,y,vy,ay]
        is_ca = rdr.ekf.dim_state == 6 and not getattr(rdr, '_use_3d', False)
        rdr_pos, rdr_P = _extract_position_cov(rdr.ekf.x, rdr.ekf.P, is_ca=is_ca)

        return covariance_intersection(cam_world, cam_P_world, rdr_pos, rdr_P)

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
            position_geo=self._extract_position_geo(rdr),
        )
