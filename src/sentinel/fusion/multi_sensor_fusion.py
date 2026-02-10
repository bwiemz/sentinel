"""Enhanced multi-sensor fusion: camera + multi-freq radar + thermal.

Extends TrackFusion with thermal track integration and threat classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

    thermal_track: ThermalTrack | None = None
    quantum_radar_track: RadarTrack | None = None
    correlated_detection: CorrelatedDetection | None = None
    radar_bands_detected: list[str] = field(default_factory=list)
    thermal_bands_detected: list[str] = field(default_factory=list)
    temperature_k: float | None = None
    qi_advantage_db: float | None = None
    has_quantum_confirmation: bool = False
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False
    is_decoy_candidate: bool = False
    is_chaff_candidate: bool = False
    threat_level: str = "UNKNOWN"
    threat_confidence: float = 0.0
    threat_probabilities: dict[str, float] = field(default_factory=dict)
    threat_method: str = "rule_based"
    intent: str = "unknown"
    intent_confidence: float = 0.0
    # IFF identification
    iff_identification: str = "unknown"
    iff_confidence: float = 0.0
    iff_mode_3a_code: str | None = None
    iff_mode_s_address: str | None = None
    iff_last_auth_mode: str | None = None
    iff_spoof_suspect: bool = False
    # Rules of Engagement
    engagement_auth: str = "weapons_hold"
    # Engagement zones (Phase 21)
    zone_authorization: str = "weapons_free"
    engagement_feasibility: dict | None = None

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
        d.update(
            {
                "sensor_count": self.sensor_count,
                "temperature_k": self.temperature_k,
                "qi_advantage_db": self.qi_advantage_db,
                "has_quantum_confirmation": self.has_quantum_confirmation,
                "is_stealth_candidate": self.is_stealth_candidate,
                "is_hypersonic_candidate": self.is_hypersonic_candidate,
                "is_decoy_candidate": self.is_decoy_candidate,
                "is_chaff_candidate": self.is_chaff_candidate,
                "threat_level": self.threat_level,
                "threat_confidence": self.threat_confidence,
                "threat_method": self.threat_method,
                "intent": self.intent,
                "intent_confidence": self.intent_confidence,
                "radar_bands": self.radar_bands_detected,
                "thermal_bands": self.thermal_bands_detected,
                "threat_probabilities": self.threat_probabilities,
                "iff_identification": self.iff_identification,
                "iff_confidence": self.iff_confidence,
                "iff_mode_3a_code": self.iff_mode_3a_code,
                "iff_mode_s_address": self.iff_mode_s_address,
                "iff_last_auth_mode": self.iff_last_auth_mode,
                "iff_spoof_suspect": self.iff_spoof_suspect,
                "engagement_auth": self.engagement_auth,
            }
        )
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
        use_temporal_alignment: bool = False,
        use_statistical_distance: bool = False,
        statistical_distance_gate: float = 9.21,
        threat_classification_method: str = "rule_based",
        threat_model_path: str | None = None,
        threat_confidence_threshold: float = 0.6,
        intent_estimation_enabled: bool = False,
        iff_interrogator: "IFFInterrogator | None" = None,
        roe_engine: "ROEEngine | None" = None,
        controlled_airspace: bool = False,
    ):
        self._base_fusion = TrackFusion(
            camera_hfov_deg=camera_hfov_deg,
            image_width_px=image_width_px,
            azimuth_gate_deg=azimuth_gate_deg,
            use_temporal_alignment=use_temporal_alignment,
            use_statistical_distance=use_statistical_distance,
            statistical_distance_gate=statistical_distance_gate,
        )
        self._use_temporal_alignment = use_temporal_alignment
        self._thermal_gate_deg = thermal_azimuth_gate_deg
        self._hfov_deg = camera_hfov_deg
        self._img_width = image_width_px
        self._min_fusion_quality = min_fusion_quality
        self._hypersonic_temp_k = hypersonic_temp_threshold_k
        self._stealth_rcs_var_db = stealth_rcs_variation_db

        # ML threat classification (lazy imports — sklearn not required for rule_based)
        self._threat_classifier = None
        self._intent_estimator = None
        if threat_classification_method == "ml":
            try:
                from sentinel.classification.threat_classifier import ThreatClassifier

                self._threat_classifier = ThreatClassifier(
                    model_path=threat_model_path,
                    confidence_threshold=threat_confidence_threshold,
                )
            except Exception:
                logger.warning("ML threat classifier unavailable, using rule-based")
        if intent_estimation_enabled:
            try:
                from sentinel.classification.intent_estimator import IntentEstimator

                self._intent_estimator = IntentEstimator()
            except Exception:
                logger.warning("Intent estimator unavailable")

        self._iff_interrogator = iff_interrogator
        self._roe_engine = roe_engine
        self._controlled_airspace = controlled_airspace

    def fuse(
        self,
        camera_tracks: list[Track],
        radar_tracks: list[RadarTrack],
        thermal_tracks: list[ThermalTrack] | None = None,
        correlated_detections: list[CorrelatedDetection] | None = None,
        quantum_radar_tracks: list[RadarTrack] | None = None,
        iff_results: dict[str, "IFFResult"] | None = None,
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
                position_geo=ft.position_geo,
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
            rule_level = self._classify_threat(eft)
            if self._threat_classifier is not None:
                result = self._threat_classifier.classify(eft, rule_level)
                eft.threat_level = result.predicted_level
                eft.threat_confidence = result.confidence
                eft.threat_probabilities = result.probabilities
                eft.threat_method = result.method_used
            else:
                eft.threat_level = rule_level
                eft.threat_confidence = 1.0
                eft.threat_method = "rule_based"
            if self._intent_estimator is not None:
                intent_result = self._intent_estimator.estimate(eft)
                eft.intent = intent_result.intent.value
                eft.intent_confidence = intent_result.confidence

            # Apply IFF identification
            self._apply_iff(eft, iff_results)

            # Apply ROE
            if self._roe_engine is not None:
                eft.engagement_auth = self._roe_engine.evaluate(
                    iff_identification=eft.iff_identification,
                    threat_level=eft.threat_level,
                    intent=eft.intent,
                    controlled_airspace=self._controlled_airspace,
                ).value

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

        for r, c in zip(row_idx, col_idx, strict=False):
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

    def _get_fused_azimuth(self, eft: EnhancedFusedTrack) -> float | None:
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

        for r, c in zip(row_idx, col_idx, strict=False):
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
                eft.is_chaff_candidate = best_cd.is_chaff_candidate

    def _apply_iff(
        self,
        eft: EnhancedFusedTrack,
        iff_results: dict | None,
    ) -> None:
        """Apply IFF identification to a fused track."""
        if iff_results is None:
            return

        # Match IFF result to fused track via radar track's last_detection target_id
        target_id = None
        if eft.radar_track is not None and eft.radar_track.last_detection is not None:
            det = eft.radar_track.last_detection
            # Target ID may be stored on the detection (simulator tags it)
            target_id = getattr(det, "target_id", None)
        if eft.quantum_radar_track is not None and target_id is None:
            det = eft.quantum_radar_track.last_detection
            if det is not None:
                target_id = getattr(det, "target_id", None)

        if target_id is None:
            return

        result = iff_results.get(target_id)
        if result is None:
            return

        eft.iff_identification = result.identification.value
        eft.iff_confidence = result.confidence
        eft.iff_mode_3a_code = result.mode_3a_code
        eft.iff_mode_s_address = result.mode_s_address
        eft.iff_last_auth_mode = (
            result.last_authenticated_mode.value
            if result.last_authenticated_mode is not None
            else None
        )
        eft.iff_spoof_suspect = result.spoof_indicators > 0

        # IFF influences threat classification
        from sentinel.core.types import IFFCode

        iff_code = result.identification
        if iff_code in (IFFCode.FRIENDLY, IFFCode.ASSUMED_FRIENDLY):
            # Friendly targets: cap threat at MEDIUM
            if eft.threat_level in (THREAT_CRITICAL, THREAT_HIGH):
                eft.threat_level = THREAT_MEDIUM
        elif iff_code == IFFCode.SPOOF_SUSPECT:
            # Spoofed IFF: boost to at least HIGH
            if eft.threat_level in (THREAT_LOW, THREAT_MEDIUM):
                eft.threat_level = THREAT_HIGH
        elif iff_code == IFFCode.ASSUMED_HOSTILE:
            # No IFF confirmed hostile: boost by one level
            if eft.threat_level == THREAT_LOW:
                eft.threat_level = THREAT_MEDIUM
            elif eft.threat_level == THREAT_MEDIUM:
                eft.threat_level = THREAT_HIGH

    def _classify_threat(self, eft: EnhancedFusedTrack) -> str:
        """Classify threat level based on sensor signatures."""
        # --- EW discrimination: chaff and decoys are LOW threat ---
        # Chaff: flagged by multi-freq correlator (high uniform RCS across bands)
        if eft.is_chaff_candidate:
            return THREAT_LOW

        # Decoy: radar track exists but no thermal match (most decoys lack IR)
        # Only flag if not already identified as stealth (stealth also lacks some sensors)
        if (
            eft.radar_track is not None
            and eft.thermal_track is None
            and not eft.is_stealth_candidate
            and not eft.has_quantum_confirmation
            and eft.camera_track is None
        ):
            eft.is_decoy_candidate = True
            return THREAT_LOW

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
