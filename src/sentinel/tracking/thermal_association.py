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

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if det.azimuth_deg is None:
                    continue
                az_rad = azimuth_deg_to_rad(det.azimuth_deg)
                z = np.array([az_rad])
                dist = track.ekf.gating_distance(z)
                if dist <= self._gate:
                    cost[i, j] = dist

        return cost
