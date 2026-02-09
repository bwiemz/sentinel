"""Feature extraction from EnhancedFusedTrack to ML feature vector.

Converts multi-sensor fused track data into a fixed-length (28,) numpy
array suitable for sklearn classifiers.  Tree-based models handle NaN
natively via surrogate splits, so missing sensor data is safe.
"""

from __future__ import annotations

import numpy as np

# Feature names — order matches vector indices
FEATURE_NAMES: list[str] = [
    # Kinematic (8)
    "speed_mps",
    "speed_mach",
    "heading_rad",
    "range_m",
    "approach_rate_mps",
    "cross_range_rate_mps",
    "acceleration_mag",
    "is_accelerating",
    # Signature (7)
    "rcs_dbsm",
    "rcs_variance_db",
    "temperature_k",
    "qi_advantage_db",
    "camera_confidence",
    "num_radar_bands",
    "has_low_freq_only",
    # Sensor coverage (6)
    "sensor_count",
    "has_camera",
    "has_radar",
    "has_thermal",
    "has_quantum",
    "has_correlated_det",
    # EW flags (3)
    "is_chaff_candidate",
    "is_decoy_candidate",
    "is_stealth_candidate",
    # Quality (4)
    "track_score",
    "track_age",
    "filter_health_score",
    "fusion_quality",
    # IFF (4)
    "iff_is_friendly",
    "iff_is_hostile",
    "iff_has_crypto_auth",
    "iff_spoof_suspect",
]

FEATURE_COUNT: int = len(FEATURE_NAMES)

_LOW_FREQ_BANDS = {"vhf", "uhf"}
_SPEED_OF_SOUND_DEFAULT = 343.0
_ACCEL_THRESHOLD_DEFAULT = 5.0


class FeatureExtractor:
    """Extracts fixed-length feature vectors from fused tracks.

    Args:
        speed_of_sound_mps: Used for Mach number computation.
        acceleration_threshold_mps2: Threshold to flag is_accelerating.
        sensor_position: Sensor position [x, y] for approach rate.
            Defaults to origin [0, 0].
    """

    def __init__(
        self,
        speed_of_sound_mps: float = _SPEED_OF_SOUND_DEFAULT,
        acceleration_threshold_mps2: float = _ACCEL_THRESHOLD_DEFAULT,
        sensor_position: np.ndarray | None = None,
    ):
        self._c = speed_of_sound_mps
        self._accel_thresh = acceleration_threshold_mps2
        self._sensor_pos = (
            np.asarray(sensor_position, dtype=np.float64)
            if sensor_position is not None
            else np.zeros(2)
        )

    def extract(self, eft) -> np.ndarray:
        """Extract feature vector from a single EnhancedFusedTrack.

        Returns:
            np.ndarray of shape (FEATURE_COUNT,).
        """
        features = np.full(FEATURE_COUNT, np.nan)

        pos = self._get_best_position(eft)
        vel = self._get_best_velocity(eft)

        # --- Kinematic features (0-7) ---
        speed = 0.0
        if vel is not None:
            speed = float(np.linalg.norm(vel))
            features[0] = speed
            features[1] = speed / self._c
            features[2] = float(np.arctan2(vel[1], vel[0]))
        else:
            features[0] = 0.0
            features[1] = 0.0
            features[2] = 0.0

        if eft.range_m is not None:
            features[3] = eft.range_m
        elif pos is not None:
            features[3] = float(np.linalg.norm(pos - self._sensor_pos))

        if pos is not None and vel is not None:
            rel_pos = pos - self._sensor_pos
            dist = float(np.linalg.norm(rel_pos))
            if dist > 1e-6:
                pos_unit = rel_pos / dist
                features[4] = -float(np.dot(vel, pos_unit))  # positive = closing
                features[5] = float(
                    np.linalg.norm(vel - np.dot(vel, pos_unit) * pos_unit)
                )
            else:
                features[4] = 0.0
                features[5] = speed
        else:
            features[4] = 0.0
            features[5] = 0.0

        accel = self._get_acceleration(eft)
        if accel is not None:
            accel_mag = float(np.linalg.norm(accel))
            features[6] = accel_mag
            features[7] = 1.0 if accel_mag > self._accel_thresh else 0.0
        else:
            features[7] = 0.0

        # --- Signature features (8-14) ---
        rcs = self._get_rcs(eft)
        if rcs is not None:
            features[8] = rcs

        rcs_var = self._get_rcs_variance(eft)
        if rcs_var is not None:
            features[9] = rcs_var

        if eft.temperature_k is not None:
            features[10] = eft.temperature_k

        if eft.qi_advantage_db is not None:
            features[11] = eft.qi_advantage_db

        if eft.camera_track is not None and eft.confidence is not None:
            features[12] = eft.confidence

        n_bands = len(eft.radar_bands_detected)
        features[13] = float(n_bands)

        if n_bands > 0 and all(b in _LOW_FREQ_BANDS for b in eft.radar_bands_detected):
            features[14] = 1.0
        else:
            features[14] = 0.0

        # --- Sensor coverage features (15-20) ---
        features[15] = float(eft.sensor_count)
        features[16] = 1.0 if eft.camera_track is not None else 0.0
        features[17] = 1.0 if eft.radar_track is not None else 0.0
        features[18] = 1.0 if eft.thermal_track is not None else 0.0
        features[19] = 1.0 if eft.quantum_radar_track is not None else 0.0
        features[20] = 1.0 if eft.correlated_detection is not None else 0.0

        # --- EW flags (21-23) ---
        features[21] = 1.0 if eft.is_chaff_candidate else 0.0
        features[22] = 1.0 if eft.is_decoy_candidate else 0.0
        features[23] = 1.0 if eft.is_stealth_candidate else 0.0

        # --- Quality features (24-27) ---
        scores = []
        ages = []
        health_scores = []
        for track in [
            eft.camera_track,
            eft.radar_track,
            eft.thermal_track,
            eft.quantum_radar_track,
        ]:
            if track is not None:
                scores.append(track.score)
                ages.append(track.age)
                if hasattr(track, "quality_monitor") and track.quality_monitor is not None:
                    health_scores.append(track.quality_monitor.consistency_score)

        features[24] = max(scores) if scores else 0.5
        features[25] = float(max(ages)) if ages else 0.0
        features[26] = max(health_scores) if health_scores else 1.0
        features[27] = eft.fusion_quality

        # --- IFF features (28-31) ---
        iff_id = getattr(eft, "iff_identification", "unknown")
        features[28] = 1.0 if iff_id in ("friendly", "assumed_friendly") else 0.0
        features[29] = 1.0 if iff_id in ("hostile", "assumed_hostile") else 0.0
        iff_auth = getattr(eft, "iff_last_auth_mode", None)
        features[30] = 1.0 if iff_auth in ("mode_4", "mode_5") else 0.0
        features[31] = 1.0 if getattr(eft, "iff_spoof_suspect", False) else 0.0

        return features

    def extract_batch(self, tracks: list) -> np.ndarray:
        """Extract features for multiple tracks.

        Returns:
            np.ndarray of shape (N, FEATURE_COUNT).
        """
        if not tracks:
            return np.empty((0, FEATURE_COUNT))
        return np.vstack([self.extract(t) for t in tracks])

    @staticmethod
    def _get_best_velocity(eft) -> np.ndarray | None:
        """Get velocity from the best available track (radar preferred)."""
        for track in [eft.radar_track, eft.thermal_track, eft.quantum_radar_track]:
            if track is not None:
                v = track.velocity
                if v is not None:
                    return np.asarray(v, dtype=np.float64)
        return None

    @staticmethod
    def _get_best_position(eft) -> np.ndarray | None:
        """Get position in meters from the best available track."""
        if eft.position_m is not None:
            return np.asarray(eft.position_m, dtype=np.float64)
        for track in [eft.radar_track, eft.thermal_track, eft.quantum_radar_track]:
            if track is not None:
                p = track.position
                if p is not None:
                    return np.asarray(p, dtype=np.float64)
        return None

    @staticmethod
    def _get_acceleration(eft) -> np.ndarray | None:
        """Get acceleration if CA mode is active."""
        for track in [eft.radar_track, eft.thermal_track, eft.quantum_radar_track]:
            if track is not None:
                x = getattr(track, "state_vector", None)
                if x is None:
                    ekf = getattr(track, "ekf", None)
                    if ekf is not None:
                        x = ekf.x
                if x is not None and len(x) >= 6:
                    # CA state: [x, vx, ax, y, vy, ay]
                    return np.array([x[2], x[5]])
        return None

    @staticmethod
    def _get_rcs(eft) -> float | None:
        """Get most recent RCS from radar detection."""
        for track in [eft.radar_track, eft.quantum_radar_track]:
            if track is not None:
                det = getattr(track, "last_detection", None)
                if det is not None and det.rcs_dbsm is not None:
                    return det.rcs_dbsm
        return None

    @staticmethod
    def _get_rcs_variance(eft) -> float | None:
        """Get RCS variance from multi-frequency correlated detection."""
        cd = eft.correlated_detection
        if cd is None:
            return None
        rcs_vals = []
        for band, det in cd.band_detections.items():
            if det.rcs_dbsm is not None:
                rcs_vals.append(det.rcs_dbsm)
        if len(rcs_vals) >= 2:
            return max(rcs_vals) - min(rcs_vals)
        return None
