"""Association for thermal (bearing-only) tracks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection
from sentinel.tracking.association import AssociationResult
from sentinel.tracking.thermal_track import ThermalTrack
from sentinel.utils.coords import azimuth_deg_to_rad

_INFEASIBLE = 1e5


class ThermalAssociator:
    """Association for thermal bearing-only tracks.

    Cost is based on Mahalanobis distance in bearing (azimuth) space only.

    Args:
        gate_threshold: Squared Mahalanobis distance threshold for gating.
    """

    def __init__(self, gate_threshold: float = 6.635):
        self._gate = gate_threshold

    def associate(
        self,
        tracks: list[ThermalTrack],
        detections: list[Detection],
    ) -> AssociationResult:
        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        cost = self._build_cost_matrix(tracks, detections)
        row_idx, col_idx = linear_sum_assignment(cost)

        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_idx, col_idx, strict=False):
            if cost[r, c] < _INFEASIBLE:
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return AssociationResult(
            matched_pairs=matched,
            unmatched_tracks=sorted(unmatched_tracks),
            unmatched_detections=sorted(unmatched_dets),
        )

    def _build_cost_matrix(
        self,
        tracks: list[ThermalTrack],
        detections: list[Detection],
    ) -> np.ndarray:
        n, m = len(tracks), len(detections)
        cost = np.full((n, m), _INFEASIBLE)

        # --- Batch EKF cost matrix computation ---
        valid_det = np.zeros(m, dtype=bool)
        measurements = []
        for j, det in enumerate(detections):
            if det.azimuth_deg is not None:
                valid_det[j] = True
                measurements.append(np.array([azimuth_deg_to_rad(det.azimuth_deg)]))

        if valid_det.any():
            from sentinel.tracking.batch_ops import batch_ekf_cost_matrix

            states = [t.ekf.x for t in tracks]
            covs = [t.ekf.P for t in tracks]
            jacs = [t.ekf.H_jacobian(t.ekf.x) for t in tracks]
            h_preds = [t.ekf.predicted_measurement for t in tracks]
            R = tracks[0].ekf.R

            maha_mat = batch_ekf_cost_matrix(
                states, covs, measurements, jacs, h_preds, R,
                angular_indices=[0], gate=self._gate,
            )

            # Map back to full cost matrix
            valid_j = 0
            for j in range(m):
                if not valid_det[j]:
                    continue
                for i in range(n):
                    if np.isfinite(maha_mat[i, valid_j]):
                        cost[i, j] = maha_mat[i, valid_j]
                valid_j += 1

        return cost
