"""Joint Probabilistic Data Association (JPDA) for multi-target tracking.

Replaces hard (Hungarian) assignment with soft probabilistic association.
Each detection contributes to each track with a probability weight (beta),
eliminating irreversible assignment errors in dense/crossing scenarios.

Core JPDA equations:
    beta_j(i) = L_ij * P_D / c_i          (detection j → track i)
    beta_0(i) = lambda_FA * (1 - P_D) / c_i   (missed detection)
    c_i = sum_j(L_ij * P_D) + lambda_FA * (1 - P_D)
    L_ij = N(y_ij; 0, S_i)                (Gaussian likelihood)

State update:
    y_c = sum_j beta_j * y_j               (combined innovation)
    x+ = x- + K * y_c
    P+ = beta_0 * P- + (1-beta_0)(Joseph form) + K * P_spread * K'
    P_spread = sum_j beta_j * y_j * y_j' - y_c * y_c'

Three sensor-specific variants:
    - JPDAAssociator: Camera (pixel-space KF, Mahalanobis+IoU gating)
    - RadarJPDAAssociator: Radar (polar EKF, angular wrapping)
    - ThermalJPDAAssociator: Thermal (1D bearing EKF, angular wrapping)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from sentinel.core.types import Detection, TrackState
from sentinel.tracking._accel import _HAS_CPP, _sentinel_core
from sentinel.tracking.association import AssociationResult
from sentinel.tracking.cost_functions import iou_bbox

if TYPE_CHECKING:
    from sentinel.tracking.radar_track import RadarTrack
    from sentinel.tracking.thermal_track import ThermalTrack
    from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gaussian_likelihood(innovation: np.ndarray, S: np.ndarray) -> float:
    """Multivariate Gaussian likelihood N(y; 0, S).

    L = (2*pi)^{-d/2} * |S|^{-1/2} * exp(-0.5 * y' S^{-1} y)

    Args:
        innovation: Innovation vector y.
        S: Innovation covariance matrix.

    Returns:
        Likelihood value (non-negative). Returns 0.0 for singular S.
    """
    if _HAS_CPP:
        return _sentinel_core.jpda.gaussian_likelihood(innovation, S)

    d = len(innovation)
    try:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 0.0
        mahal = float(innovation @ np.linalg.solve(S, innovation))
        log_L = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
        return float(np.exp(log_L))
    except np.linalg.LinAlgError:
        return 0.0


def _compute_beta_coefficients(
    likelihoods: np.ndarray,
    P_D: float,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Compute JPDA beta coefficients for one track.

    Args:
        likelihoods: Array of Gaussian likelihoods L_ij for each gated detection.
        P_D: Detection probability.
        lam: False alarm (clutter) density parameter.

    Returns:
        (betas, beta_0) where betas[j] is the probability that detection j
        originated from this track, and beta_0 is the missed-detection prob.
    """
    n = len(likelihoods)
    if n == 0:
        return np.array([]), 1.0

    if _HAS_CPP:
        betas_vec, beta_0 = _sentinel_core.jpda.compute_beta_coefficients(
            np.asarray(likelihoods), P_D, lam,
        )
        return betas_vec, beta_0

    miss_term = lam * (1.0 - P_D)
    det_terms = likelihoods * P_D
    c = float(np.sum(det_terms) + miss_term)

    if c <= 0:
        return np.zeros(n), 1.0

    betas = det_terms / c
    beta_0 = miss_term / c
    return betas, float(beta_0)


def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ===========================================================================
# Camera JPDA
# ===========================================================================

class JPDAAssociator:
    """JPDA for camera tracks (pixel-space Kalman filter).

    Uses Mahalanobis distance for gating and optionally IoU for
    additional filtering.

    Args:
        gate_threshold: Mahalanobis chi-squared gate.
        P_D: Detection probability.
        false_alarm_density: Clutter density (lambda).
        iou_gate: Minimum IoU for association (0 to disable).
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,
        P_D: float = 0.9,
        false_alarm_density: float = 1e-6,
        iou_gate: float = 0.0,
    ):
        self._gate = gate_threshold
        self._P_D = P_D
        self._lam = false_alarm_density
        self._iou_gate = iou_gate

    def associate_and_update(
        self,
        tracks: list[Track],
        detections: list[Detection],
    ) -> AssociationResult:
        """JPDA association and state update in one step.

        Unlike Hungarian, JPDA computes weighted innovations from ALL
        gated detections and updates each track with the combined
        innovation. This prevents hard assignment errors.

        Returns:
            AssociationResult with matched/unmatched indices.
        """
        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        n_tracks = len(tracks)
        n_dets = len(detections)

        # For each track, compute innovations and likelihoods for gated detections
        track_gated_dets: list[list[int]] = []  # per-track list of gated det indices
        track_innovations: list[list[np.ndarray]] = []
        track_likelihoods: list[np.ndarray] = []
        track_S: list[np.ndarray | None] = []

        for i, track in enumerate(tracks):
            H = track.kf.H
            S = H @ track.kf.P @ H.T + track.kf.R
            z_pred = track.kf.predicted_measurement
            track_S.append(S)

            gated = []
            innovations = []
            likelihoods_list = []

            for j, det in enumerate(detections):
                if det.bbox is None:
                    continue
                center = det.bbox_center
                if center is None:
                    continue

                y = center - z_pred
                mahal = float(y @ np.linalg.solve(S, y))
                if mahal > self._gate:
                    continue

                # Optional IoU gate
                if self._iou_gate > 0:
                    pred_bbox = track.predicted_bbox
                    if pred_bbox is not None and det.bbox is not None:
                        iou = iou_bbox(pred_bbox, det.bbox)
                        if iou < self._iou_gate:
                            continue

                gated.append(j)
                innovations.append(y)
                likelihoods_list.append(_gaussian_likelihood(y, S))

            track_gated_dets.append(gated)
            track_innovations.append(innovations)
            track_likelihoods.append(np.array(likelihoods_list) if likelihoods_list else np.array([]))

        # Compute betas and update each track
        matched_pairs = []
        all_matched_dets: set[int] = set()

        for i, track in enumerate(tracks):
            gated = track_gated_dets[i]
            innovations = track_innovations[i]
            likelihoods = track_likelihoods[i]

            if not gated:
                continue

            betas, beta_0 = _compute_beta_coefficients(likelihoods, self._P_D, self._lam)
            S = track_S[i]
            H = track.kf.H
            K = np.linalg.solve(S.T, (track.kf.P @ H.T).T).T

            # Combined innovation
            dim_meas = len(innovations[0])
            y_c = np.zeros(dim_meas)
            for j_idx, y_j in enumerate(innovations):
                y_c += betas[j_idx] * y_j

            # State update
            track.kf.x = track.kf.x + K @ y_c

            # Covariance update with spread of innovations
            P_spread = np.zeros((dim_meas, dim_meas))
            for j_idx, y_j in enumerate(innovations):
                P_spread += betas[j_idx] * np.outer(y_j, y_j)
            P_spread -= np.outer(y_c, y_c)

            I_KH = np.eye(track.kf.dim_state) - K @ H
            P_std = I_KH @ track.kf.P @ I_KH.T + K @ track.kf.R @ K.T
            track.kf.P = beta_0 * track.kf.P + (1.0 - beta_0) * P_std + K @ P_spread @ K.T

            # Record NIS for quality monitoring
            if track.quality_monitor is not None:
                track.quality_monitor.record_innovation(y_c, S)

            # Record best detection for metadata
            best_j = int(np.argmax(betas))
            best_det_idx = gated[best_j]
            track.last_detection = detections[best_det_idx]
            track.last_update_time = detections[best_det_idx].timestamp
            if detections[best_det_idx].class_name:
                track.class_histogram[detections[best_det_idx].class_name] = \
                    track.class_histogram.get(detections[best_det_idx].class_name, 0) + 1
            track._record_hit()
            matched_pairs.append((i, best_det_idx))
            all_matched_dets.update(gated)

        unmatched_tracks = [i for i in range(n_tracks) if not track_gated_dets[i]]
        unmatched_dets = [j for j in range(n_dets) if j not in all_matched_dets]

        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
        )


# ===========================================================================
# Radar JPDA
# ===========================================================================

class RadarJPDAAssociator:
    """JPDA for radar tracks (polar EKF with angular wrapping).

    Args:
        gate_threshold: Mahalanobis chi-squared gate.
        P_D: Detection probability.
        false_alarm_density: Clutter density.
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,
        P_D: float = 0.9,
        false_alarm_density: float = 1e-6,
    ):
        self._gate = gate_threshold
        self._P_D = P_D
        self._lam = false_alarm_density

    def associate_and_update(
        self,
        tracks: list[RadarTrack],
        detections: list[Detection],
    ) -> AssociationResult:
        """JPDA association and state update for radar tracks."""
        from sentinel.utils.coords import azimuth_deg_to_rad

        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        n_tracks = len(tracks)
        n_dets = len(detections)

        track_gated_dets: list[list[int]] = []
        track_innovations: list[list[np.ndarray]] = []
        track_likelihoods: list[np.ndarray] = []
        track_S: list[np.ndarray | None] = []

        for i, track in enumerate(tracks):
            H = track.ekf.H_jacobian(track.ekf.x)
            S = H @ track.ekf.P @ H.T + track.ekf.R
            z_pred = track.ekf.predicted_measurement
            track_S.append(S)

            gated = []
            innovations = []
            likelihoods_list = []

            for j, det in enumerate(detections):
                if det.range_m is None or det.azimuth_deg is None:
                    continue

                az_rad = azimuth_deg_to_rad(det.azimuth_deg)

                # Build measurement vector based on track mode
                if track._use_3d:
                    el_rad = np.radians(det.elevation_deg or 0.0)
                    z = np.array([det.range_m, az_rad, el_rad])
                elif track._use_doppler and det.velocity_mps is not None:
                    z = np.array([det.range_m, az_rad, det.velocity_mps])
                elif not track._use_doppler:
                    z = np.array([det.range_m, az_rad])
                else:
                    continue  # Doppler mode but no velocity

                y = z - z_pred
                # Wrap angular components
                y[1] = _wrap_angle(y[1])
                if track._use_3d and len(y) > 2:
                    y[2] = _wrap_angle(y[2])

                try:
                    mahal = float(y @ np.linalg.solve(S, y))
                except np.linalg.LinAlgError:
                    continue
                if mahal > self._gate:
                    continue

                gated.append(j)
                innovations.append(y)
                likelihoods_list.append(_gaussian_likelihood(y, S))

            track_gated_dets.append(gated)
            track_innovations.append(innovations)
            track_likelihoods.append(np.array(likelihoods_list) if likelihoods_list else np.array([]))

        # Compute betas and update each track
        matched_pairs = []
        all_matched_dets: set[int] = set()

        for i, track in enumerate(tracks):
            gated = track_gated_dets[i]
            innovations = track_innovations[i]
            likelihoods = track_likelihoods[i]

            if not gated:
                continue

            betas, beta_0 = _compute_beta_coefficients(likelihoods, self._P_D, self._lam)
            S = track_S[i]
            H = track.ekf.H_jacobian(track.ekf.x)
            K = np.linalg.solve(S.T, (track.ekf.P @ H.T).T).T

            # Combined innovation
            dim_meas = len(innovations[0])
            y_c = np.zeros(dim_meas)
            for j_idx, y_j in enumerate(innovations):
                y_c += betas[j_idx] * y_j

            # State update
            track.ekf.x = track.ekf.x + K @ y_c

            # Covariance update
            P_spread = np.zeros((dim_meas, dim_meas))
            for j_idx, y_j in enumerate(innovations):
                P_spread += betas[j_idx] * np.outer(y_j, y_j)
            P_spread -= np.outer(y_c, y_c)

            I_KH = np.eye(track.ekf.dim_state) - K @ H
            P_std = I_KH @ track.ekf.P @ I_KH.T + K @ track.ekf.R @ K.T
            track.ekf.P = beta_0 * track.ekf.P + (1.0 - beta_0) * P_std + K @ P_spread @ K.T

            # Record NIS
            if track.quality_monitor is not None:
                track.quality_monitor.record_innovation(y_c, S)

            # Record best detection
            best_j = int(np.argmax(betas))
            best_det_idx = gated[best_j]
            track.last_detection = detections[best_det_idx]
            track.last_update_time = detections[best_det_idx].timestamp
            track._record_hit()
            matched_pairs.append((i, best_det_idx))
            all_matched_dets.update(gated)

        unmatched_tracks = [i for i in range(n_tracks) if not track_gated_dets[i]]
        unmatched_dets = [j for j in range(n_dets) if j not in all_matched_dets]

        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
        )


# ===========================================================================
# Thermal JPDA
# ===========================================================================

class ThermalJPDAAssociator:
    """JPDA for thermal bearing-only tracks (1D EKF).

    Args:
        gate_threshold: Mahalanobis chi-squared gate (1 DOF).
        P_D: Detection probability.
        false_alarm_density: Clutter density.
    """

    def __init__(
        self,
        gate_threshold: float = 6.635,
        P_D: float = 0.9,
        false_alarm_density: float = 1e-6,
    ):
        self._gate = gate_threshold
        self._P_D = P_D
        self._lam = false_alarm_density

    def associate_and_update(
        self,
        tracks: list[ThermalTrack],
        detections: list[Detection],
    ) -> AssociationResult:
        """JPDA association and state update for thermal tracks."""
        from sentinel.utils.coords import azimuth_deg_to_rad

        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        n_tracks = len(tracks)
        n_dets = len(detections)

        track_gated_dets: list[list[int]] = []
        track_innovations: list[list[np.ndarray]] = []
        track_likelihoods: list[np.ndarray] = []
        track_S: list[np.ndarray | None] = []

        for i, track in enumerate(tracks):
            H = track.ekf.H_jacobian(track.ekf.x)
            S = H @ track.ekf.P @ H.T + track.ekf.R
            z_pred = track.ekf.predicted_measurement
            track_S.append(S)

            gated = []
            innovations = []
            likelihoods_list = []

            for j, det in enumerate(detections):
                az_deg = det.azimuth_deg
                if az_deg is None:
                    continue
                az_rad = azimuth_deg_to_rad(az_deg)
                z = np.array([az_rad])

                y = z - z_pred
                y[0] = _wrap_angle(y[0])

                try:
                    mahal = float(y @ np.linalg.solve(S, y))
                except np.linalg.LinAlgError:
                    continue
                if mahal > self._gate:
                    continue

                gated.append(j)
                innovations.append(y)
                likelihoods_list.append(_gaussian_likelihood(y, S))

            track_gated_dets.append(gated)
            track_innovations.append(innovations)
            track_likelihoods.append(np.array(likelihoods_list) if likelihoods_list else np.array([]))

        # Compute betas and update
        matched_pairs = []
        all_matched_dets: set[int] = set()

        for i, track in enumerate(tracks):
            gated = track_gated_dets[i]
            innovations = track_innovations[i]
            likelihoods = track_likelihoods[i]

            if not gated:
                continue

            betas, beta_0 = _compute_beta_coefficients(likelihoods, self._P_D, self._lam)
            S = track_S[i]
            H = track.ekf.H_jacobian(track.ekf.x)
            K = np.linalg.solve(S.T, (track.ekf.P @ H.T).T).T

            # Combined innovation (1D bearing)
            y_c = np.zeros(1)
            for j_idx, y_j in enumerate(innovations):
                y_c += betas[j_idx] * y_j

            # State update
            track.ekf.x = track.ekf.x + K @ y_c

            # Covariance update
            P_spread = np.zeros((1, 1))
            for j_idx, y_j in enumerate(innovations):
                P_spread += betas[j_idx] * np.outer(y_j, y_j)
            P_spread -= np.outer(y_c, y_c)

            I_KH = np.eye(track.ekf.dim_state) - K @ H
            P_std = I_KH @ track.ekf.P @ I_KH.T + K @ track.ekf.R @ K.T
            track.ekf.P = beta_0 * track.ekf.P + (1.0 - beta_0) * P_std + K @ P_spread @ K.T

            # Record NIS
            if track.quality_monitor is not None:
                track.quality_monitor.record_innovation(y_c, S)

            # Best detection
            best_j = int(np.argmax(betas))
            best_det_idx = gated[best_j]
            track.last_detection = detections[best_det_idx]
            track.last_update_time = detections[best_det_idx].timestamp
            track._last_temperature_k = detections[best_det_idx].temperature_k
            track._record_hit()
            matched_pairs.append((i, best_det_idx))
            all_matched_dets.update(gated)

        unmatched_tracks = [i for i in range(n_tracks) if not track_gated_dets[i]]
        unmatched_dets = [j for j in range(n_dets) if j not in all_matched_dets]

        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
        )
