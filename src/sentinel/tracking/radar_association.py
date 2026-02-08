"""Radar detection-to-track association using Hungarian algorithm.

Cost matrix based on Mahalanobis distance in polar measurement space.
No IoU component since radar has no bounding boxes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection
from sentinel.tracking.association import INFEASIBLE, AssociationResult
from sentinel.tracking.radar_track import RadarTrack
from sentinel.utils.coords import azimuth_deg_to_rad


class RadarAssociator:
    """Hungarian algorithm association for radar tracks.

    Args:
        gate_threshold: Chi-squared gating threshold (2 DOF).
            Default 9.21 for 99% confidence.
    """

    def __init__(self, gate_threshold: float = 9.21):
        self._gate_threshold = gate_threshold

    def associate(
        self, tracks: list[RadarTrack], detections: list[Detection]
    ) -> AssociationResult:
        """Optimal assignment between radar tracks and detections."""
        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        cost_matrix = self._build_cost_matrix(tracks, detections)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < INFEASIBLE:
                matched.append((r, c))
                unmatched_t.discard(r)
                unmatched_d.discard(c)

        return AssociationResult(
            matched_pairs=matched,
            unmatched_tracks=sorted(unmatched_t),
            unmatched_detections=sorted(unmatched_d),
        )

    def _build_cost_matrix(
        self, tracks: list[RadarTrack], detections: list[Detection]
    ) -> np.ndarray:
        """Cost matrix using Mahalanobis distance in polar space."""
        N = len(tracks)
        M = len(detections)
        cost = np.full((N, M), INFEASIBLE)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if det.range_m is None or det.azimuth_deg is None:
                    continue

                z = np.array([det.range_m, azimuth_deg_to_rad(det.azimuth_deg)])
                maha = track.ekf.gating_distance(z)

                if maha > self._gate_threshold:
                    continue

                cost[i, j] = maha

        return cost
