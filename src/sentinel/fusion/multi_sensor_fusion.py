"""Enhanced multi-sensor fusion: camera + multi-freq radar + thermal.

Extends TrackFusion with thermal track integration and threat classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import SensorType, generate_track_id
from sentinel.fusion.multifreq_correlator import CorrelatedDetection
from sentinel.fusion.track_fusion import FusedTrack, TrackFusion
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)

_INFEASIBLE = 1e5

# Threat levels
THREAT_LOW = "LOW"
THREAT_MEDIUM = "MEDIUM"
THREAT_HIGH = "HIGH"
THREAT_CRITICAL = "CRITICAL"


@dataclass
class EnhancedFusedTrack(FusedTrack):
    """Fused track with multi-sensor metadata and threat classification."""

    thermal_track: Optional[ThermalTrack] = None
    quantum_radar_track: Optional[RadarTrack] = None
    correlated_detection: Optional[CorrelatedDetection] = None
    radar_bands_detected: list[str] = field(default_factory=list)
    thermal_bands_detected: list[str] = field(default_factory=list)
    temperature_k: Optional[float] = None
    qi_advantage_db: Optional[float] = None
    has_quantum_confirmation: bool = False
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False
    threat_level: str = "UNKNOWN"

    @property
    def sensor_count(self) -> int:
        count = 0
        if self.camera_track is not None:
            count += 1
        if self.radar_track is not None:
            count += 1
        if self.thermal_track is not None:
            count += 1
        if self.quantum_radar_track is not None:
            count += 1
        return count

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "sensor_count": self.sensor_count,
            "temperature_k": self.temperature_k,
            "qi_advantage_db": self.qi_advantage_db,
            "has_quantum_confirmation": self.has_quantum_confirmation,
            "is_stealth_candidate": self.is_stealth_candidate,
            "is_hypersonic_candidate": self.is_hypersonic_candidate,
            "threat_level": self.threat_level,
            "radar_bands": self.radar_bands_detected,
            "thermal_bands": self.thermal_bands_detected,
        })
        return d


class MultiSensorFusion:
    """Multi-sensor fusion: camera + multi-freq radar + thermal.

    Two-stage fusion:
    1. Camera <-> Radar angular correspondence (same as TrackFusion).
    2. Thermal tracks associated into fused tracks by bearing match.
    Then classifies threats based on sensor signatures.

    Args:
        camera_hfov_deg: Camera horizontal FOV in degrees.
        image_width_px: Camera image width in pixels.
        azimuth_gate_deg: Max azimuth difference for camera-radar correlation.
        thermal_azimuth_gate_deg: Max azimuth difference for thermal association.
    """

    def __init__(
        self,
        camera_hfov_deg: float = 60.0,
        image_width_px: int = 1280,
        azimuth_gate_deg: float = 5.0,
        thermal_azimuth_gate_deg: float = 3.0,
        min_fusion_quality: float = 0.0,
        hypersonic_temp_threshold_k: float = 1500.0,
        stealth_rcs_variation_db: float = 15.0,
    ):
        self._base_fusion = TrackFusion(
            camera_hfov_deg=camera_hfov_deg,
            image_width_px=image_width_px,
            azimuth_gate_deg=azimuth_gate_deg,
        )
        self._thermal_gate_deg = thermal_azimuth_gate_deg
        self._hfov_deg = camera_hfov_deg
        self._img_width = image_width_px
        self._min_fusion_quality = min_fusion_quality
        self._hypersonic_temp_k = hypersonic_temp_threshold_k
        self._stealth_rcs_var_db = stealth_rcs_variation_db

    def fuse(
        self,
        camera_tracks: list[Track],
        radar_tracks: list[RadarTrack],
        thermal_tracks: Optional[list[ThermalTrack]] = None,
        correlated_detections: Optional[list[CorrelatedDetection]] = None,
        quantum_radar_tracks: Optional[list[RadarTrack]] = None,
    ) -> list[EnhancedFusedTrack]:
        """Full multi-sensor fusion."""
        if not camera_tracks and not radar_tracks and not thermal_tracks and not quantum_radar_tracks:
            return []

        # Stage 1: Camera <-> Radar fusion (reuse existing logic)
        base_fused = self._base_fusion.fuse(camera_tracks, radar_tracks)

        # Convert to EnhancedFusedTrack
        enhanced: list[EnhancedFusedTrack] = []
        for ft in base_fused:
            eft = EnhancedFusedTrack(
                fused_id=ft.fused_id,
                camera_track=ft.camera_track,
                radar_track=ft.radar_track,
                position_px=ft.position_px,
                position_m=ft.position_m,
                range_m=ft.range_m,
                azimuth_deg=ft.azimuth_deg,
                velocity_mps=ft.velocity_mps,
                class_name=ft.class_name,
                confidence=ft.confidence,
                sensor_sources=ft.sensor_sources,
                fusion_quality=ft.fusion_quality,
            )
            enhanced.append(eft)

        # Stage 2: Associate thermal tracks
        if thermal_tracks:
            enhanced = self._associate_thermal(enhanced, thermal_tracks)

        # Stage 3: Associate quantum radar tracks
        if quantum_radar_tracks:
            enhanced = self._associate_quantum_radar(enhanced, quantum_radar_tracks)

        # Attach correlated detection metadata
        if correlated_detections:
            self._attach_correlation_metadata(enhanced, correlated_detections)

        # Classify threats
        for eft in enhanced:
            eft.threat_level = self._classify_threat(eft)
            eft.fusion_quality = self._compute_fusion_quality(eft)

        # Filter by minimum fusion quality
        if self._min_fusion_quality > 0:
            enhanced = [e for e in enhanced if e.fusion_quality >= self._min_fusion_quality]

        return enhanced

    def _associate_thermal(
        self,
        fused_tracks: list[EnhancedFusedTrack],
        thermal_tracks: list[ThermalTrack],
    ) -> list[EnhancedFusedTrack]:
        """Associate thermal tracks into existing fused tracks by bearing."""
        if not fused_tracks or not thermal_tracks:
            # No existing fused tracks -- create thermal-only
            for tt in thermal_tracks:
                fused_tracks.append(self._make_thermal_only(tt))
            return fused_tracks

        # Build cost matrix: fused track azimuth vs thermal bearing
        n_fused = len(fused_tracks)
        n_thermal = len(thermal_tracks)
        cost = np.full((n_fused, n_thermal), _INFEASIBLE)

        for i, eft in enumerate(fused_tracks):
            fused_az = self._get_fused_azimuth(eft)
            if fused_az is None:
                continue
            for j, tt in enumerate(thermal_tracks):
                ang_dist = self._angular_distance(fused_az, tt.azimuth_deg)
                if ang_dist <= self._thermal_gate_deg:
                    cost[i, j] = ang_dist

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_thermal: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < _INFEASIBLE:
                fused_tracks[r].thermal_track = thermal_tracks[c]
                fused_tracks[r].temperature_k = thermal_tracks[c].temperature_k
                fused_tracks[r].sensor_sources.add(SensorType.THERMAL)
                matched_thermal.add(c)

        # Unmatched thermal -> thermal-only tracks
        for j, tt in enumerate(thermal_tracks):
            if j not in matched_thermal:
                fused_tracks.append(self._make_thermal_only(tt))

        return fused_tracks

    def _get_fused_azimuth(self, eft: EnhancedFusedTrack) -> Optional[float]:
        """Get the best azimuth estimate for a fused track."""
        if eft.radar_track is not None:
            return eft.radar_track.azimuth_deg
        if eft.camera_track is not None:
            px_x = eft.camera_track.position[0]
            return self._base_fusion.pixel_to_azimuth(px_x)
        return None

    def _make_thermal_only(self, tt: ThermalTrack) -> EnhancedFusedTrack:
        return EnhancedFusedTrack(
            fused_id=generate_track_id(),
            thermal_track=tt,
            azimuth_deg=tt.azimuth_deg,
            temperature_k=tt.temperature_k,
            sensor_sources={SensorType.THERMAL},
            fusion_quality=tt.score * 0.3,
        )

    def _associate_quantum_radar(
        self,
        fused_tracks: list[EnhancedFusedTrack],
        quantum_tracks: list[RadarTrack],
    ) -> list[EnhancedFusedTrack]:
        """Associate quantum radar tracks into fused tracks by azimuth."""
        if not fused_tracks or not quantum_tracks:
            # Create quantum-only fused tracks
            for qt in quantum_tracks:
                fused_tracks.append(self._make_quantum_only(qt))
            return fused_tracks

        n_fused = len(fused_tracks)
        n_quantum = len(quantum_tracks)
        cost = np.full((n_fused, n_quantum), _INFEASIBLE)

        for i, eft in enumerate(fused_tracks):
            fused_az = self._get_fused_azimuth(eft)
            if fused_az is None:
                continue
            for j, qt in enumerate(quantum_tracks):
                ang_dist = self._angular_distance(fused_az, qt.azimuth_deg)
                if ang_dist <= self._base_fusion._gate_deg:
                    cost[i, j] = ang_dist

        row_idx, col_idx = linear_sum_assignment(cost)
        matched: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < _INFEASIBLE:
                fused_tracks[r].quantum_radar_track = quantum_tracks[c]
                fused_tracks[r].has_quantum_confirmation = True
                fused_tracks[r].sensor_sources.add(SensorType.QUANTUM_RADAR)
                # Extract QI advantage from the track's last detection
                last_det = quantum_tracks[c].last_detection
                if last_det and last_det.qi_advantage_db is not None:
                    fused_tracks[r].qi_advantage_db = last_det.qi_advantage_db
                matched.add(c)

        # Unmatched quantum tracks -> quantum-only
        for j, qt in enumerate(quantum_tracks):
            if j not in matched:
                fused_tracks.append(self._make_quantum_only(qt))

        return fused_tracks

    def _make_quantum_only(self, qt: RadarTrack) -> EnhancedFusedTrack:
        eft = EnhancedFusedTrack(
            fused_id=generate_track_id(),
            quantum_radar_track=qt,
            azimuth_deg=qt.azimuth_deg,
            range_m=qt.range_m,
            velocity_mps=float(np.linalg.norm(qt.velocity)),
            has_quantum_confirmation=True,
            sensor_sources={SensorType.QUANTUM_RADAR},
            fusion_quality=qt.score * 0.3,
        )
        last_det = qt.last_detection
        if last_det and last_det.qi_advantage_db is not None:
            eft.qi_advantage_db = last_det.qi_advantage_db
        return eft

    def _attach_correlation_metadata(
        self,
        enhanced: list[EnhancedFusedTrack],
        correlated: list[CorrelatedDetection],
    ) -> None:
        """Attach multi-freq correlation metadata to fused tracks."""
        for eft in enhanced:
            if eft.radar_track is None:
                continue
            # Find the correlated detection closest to this radar track
            best_cd = None
            best_dist = float("inf")
            rdr_az = eft.radar_track.azimuth_deg
            for cd in correlated:
                pd = cd.primary_detection
                if pd.azimuth_deg is not None:
                    dist = abs(pd.azimuth_deg - rdr_az)
                    if dist < best_dist:
                        best_dist = dist
                        best_cd = cd
            if best_cd is not None and best_dist < 5.0:
                eft.correlated_detection = best_cd
                eft.radar_bands_detected = best_cd.bands_detected
                eft.is_stealth_candidate = best_cd.is_stealth_candidate
                eft.is_hypersonic_candidate = best_cd.is_hypersonic_candidate

    def _classify_threat(self, eft: EnhancedFusedTrack) -> str:
        """Classify threat level based on sensor signatures."""
        # CRITICAL: hypersonic (extreme thermal + high speed indicators)
        if eft.is_hypersonic_candidate:
            return THREAT_CRITICAL
        if eft.temperature_k is not None and eft.temperature_k > self._hypersonic_temp_k:
            return THREAT_CRITICAL

        # CRITICAL: quantum-confirmed stealth (QI detected what classical missed)
        if eft.is_stealth_candidate and eft.has_quantum_confirmation:
            return THREAT_CRITICAL

        # HIGH: stealth (detected at low-freq radar but not high-freq)
        if eft.is_stealth_candidate:
            return THREAT_HIGH

        # HIGH: quantum-only detection (no classical radar, but QI sees it)
        if eft.has_quantum_confirmation and eft.radar_track is None and eft.camera_track is None:
            return THREAT_HIGH

        # MEDIUM: confirmed by multiple sensor modalities
        if eft.sensor_count >= 2:
            return THREAT_MEDIUM

        # LOW: single sensor, unconfirmed
        return THREAT_LOW

    def _compute_fusion_quality(self, eft: EnhancedFusedTrack) -> float:
        """Enhanced quality score based on sensor diversity."""
        base = 0.0

        # Sensor contributions
        if eft.camera_track is not None:
            base += eft.camera_track.score * 0.3
        if eft.radar_track is not None:
            base += eft.radar_track.score * 0.3
        if eft.thermal_track is not None:
            base += eft.thermal_track.score * 0.2
        if eft.quantum_radar_track is not None:
            base += eft.quantum_radar_track.score * 0.2

        # Multi-band radar bonus
        n_bands = len(eft.radar_bands_detected)
        if n_bands > 1:
            base += min(0.2, n_bands * 0.05)

        return min(1.0, base)

    @staticmethod
    def _angular_distance(az1_deg: float, az2_deg: float) -> float:
        diff = abs(az1_deg - az2_deg) % 360.0
        return min(diff, 360.0 - diff)
