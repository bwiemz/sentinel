"""Data association using the Hungarian (Munkres) algorithm.

Uses scipy.optimize.linear_sum_assignment for optimal assignment between
detections and existing tracks based on a combined cost matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection
from sentinel.tracking.cost_functions import iou_bbox

if TYPE_CHECKING:
    from sentinel.tracking.track import Track

logger = logging.getLogger(__name__)

# Large cost sentinel for infeasible assignments
INFEASIBLE = 1e5


@dataclass
class AssociationResult:
    """Result of data association."""
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)  # (track_idx, det_idx)
    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)


class HungarianAssociator:
    """Global nearest-neighbor data association using the Hungarian algorithm.

    Builds a cost matrix combining Mahalanobis distance and IoU,
    applies gating to eliminate infeasible assignments, then solves
    the optimal assignment problem.

    Args:
        gate_threshold: Mahalanobis distance threshold for gating (chi2).
        iou_weight: Weight for IoU cost component.
        mahalanobis_weight: Weight for Mahalanobis cost component.
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,
        iou_weight: float = 0.5,
        mahalanobis_weight: float = 0.5,
    ):
        self._gate_threshold = gate_threshold
        self._iou_weight = iou_weight
        self._maha_weight = mahalanobis_weight

    def associate(
        self, tracks: list[Track], detections: list[Detection]
    ) -> AssociationResult:
        """Perform optimal assignment between tracks and detections.

        Returns:
            AssociationResult with matched pairs, unmatched tracks, and
            unmatched detections.
        """
        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(tracks, detections)

        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Partition into matched / unmatched
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
        self, tracks: list[Track], detections: list[Detection]
    ) -> np.ndarray:
        """Build NxM cost matrix using weighted IoU + Mahalanobis distance."""
        N = len(tracks)
        M = len(detections)
        cost = np.full((N, M), INFEASIBLE)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if det.bbox is None:
                    continue

                center = det.bbox_center
                if center is None:
                    continue

                # Mahalanobis distance (gating)
                maha = track.kf.gating_distance(center)
                if maha > self._gate_threshold:
                    continue  # Outside gate, leave as INFEASIBLE

                # IoU cost (1 - IoU, so lower is better)
                iou_cost = 1.0
                pred_bbox = track.predicted_bbox
                if pred_bbox is not None:
                    iou_cost = 1.0 - iou_bbox(pred_bbox, det.bbox)

                # Combined cost
                cost[i, j] = (
                    self._maha_weight * maha + self._iou_weight * iou_cost
                )

        return cost
