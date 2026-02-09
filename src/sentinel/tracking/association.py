"""Data association using the Hungarian (Munkres) algorithm.

Uses scipy.optimize.linear_sum_assignment for optimal assignment between
detections and existing tracks based on a combined cost matrix.
Supports cascaded association (confirmed tracks get first pick).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection, TrackState
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
        cascaded: If True, confirmed tracks get first pick, then others.
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,
        iou_weight: float = 0.5,
        mahalanobis_weight: float = 0.5,
        cascaded: bool = False,
    ):
        self._gate_threshold = gate_threshold
        self._iou_weight = iou_weight
        self._maha_weight = mahalanobis_weight
        self._cascaded = cascaded

    def associate(self, tracks: list[Track], detections: list[Detection]) -> AssociationResult:
        """Perform optimal assignment between tracks and detections."""
        if not tracks or not detections:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=list(range(len(detections))),
            )

        if self._cascaded:
            return self._cascaded_associate(tracks, detections)

        return self._single_pass(tracks, detections, list(range(len(tracks))))

    def _single_pass(
        self,
        tracks: list[Track],
        detections: list[Detection],
        track_indices: list[int],
        available_dets: set[int] | None = None,
    ) -> AssociationResult:
        """Single-pass Hungarian association on a subset of tracks."""
        if available_dets is None:
            available_dets = set(range(len(detections)))

        det_list = sorted(available_dets)
        if not track_indices or not det_list:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(track_indices),
                unmatched_detections=det_list,
            )

        n_tracks = len(track_indices)
        n_dets = len(det_list)
        cost = np.full((n_tracks, n_dets), INFEASIBLE)

        # --- Batch cost matrix computation ---
        # Extract valid detection data
        valid_det = np.zeros(n_dets, dtype=bool)
        centers = []
        det_bboxes = []
        for j, dj in enumerate(det_list):
            det = detections[dj]
            c = det.bbox_center if det.bbox is not None else None
            if c is not None:
                valid_det[j] = True
                centers.append(c)
                det_bboxes.append(det.bbox)

        if valid_det.any():
            # Extract track KF data
            states = [tracks[ti].kf.x for ti in track_indices]
            covs = [tracks[ti].kf.P for ti in track_indices]
            H = tracks[track_indices[0]].kf.H
            R = tracks[track_indices[0]].kf.R

            # Batch Mahalanobis gating
            from sentinel.tracking.batch_ops import batch_kf_cost_matrix, batch_iou_matrix

            maha = batch_kf_cost_matrix(states, covs, centers, H, R, gate=self._gate_threshold)

            # Batch IoU
            track_bboxes = []
            has_pred_bbox = []
            for ti in track_indices:
                pb = tracks[ti].predicted_bbox
                has_pred_bbox.append(pb is not None)
                track_bboxes.append(pb if pb is not None else np.zeros(4))

            iou = batch_iou_matrix(np.array(track_bboxes), np.array(det_bboxes))

            # Combine into cost matrix (only for valid detections)
            valid_j = 0
            for j in range(n_dets):
                if not valid_det[j]:
                    continue
                for i in range(n_tracks):
                    m = maha[i, valid_j]
                    if np.isinf(m):
                        continue
                    iou_cost = (1.0 - iou[i, valid_j]) if has_pred_bbox[i] else 1.0
                    cost[i, j] = self._maha_weight * m + self._iou_weight * iou_cost
                valid_j += 1

        row_idx, col_idx = linear_sum_assignment(cost)

        matched = []
        unmatched_t = set(track_indices)
        unmatched_d = set(det_list)

        for r, c in zip(row_idx, col_idx, strict=False):
            if cost[r, c] < INFEASIBLE:
                matched.append((track_indices[r], det_list[c]))
                unmatched_t.discard(track_indices[r])
                unmatched_d.discard(det_list[c])

        return AssociationResult(
            matched_pairs=matched,
            unmatched_tracks=sorted(unmatched_t),
            unmatched_detections=sorted(unmatched_d),
        )

    def _cascaded_associate(self, tracks: list[Track], detections: list[Detection]) -> AssociationResult:
        """Two-pass: confirmed tracks first, then tentative/coasting."""
        confirmed_idx = [i for i, t in enumerate(tracks) if t.state == TrackState.CONFIRMED]
        other_idx = [i for i, t in enumerate(tracks) if t.state != TrackState.CONFIRMED]

        result1 = self._single_pass(tracks, detections, confirmed_idx)
        remaining_dets = set(result1.unmatched_detections)

        result2 = self._single_pass(tracks, detections, other_idx, available_dets=remaining_dets)

        return AssociationResult(
            matched_pairs=result1.matched_pairs + result2.matched_pairs,
            unmatched_tracks=sorted(set(result1.unmatched_tracks) | set(result2.unmatched_tracks)),
            unmatched_detections=sorted(set(result2.unmatched_detections)),
        )

    def _build_cost_matrix(self, tracks: list[Track], detections: list[Detection]) -> np.ndarray:
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
                cost[i, j] = self._maha_weight * maha + self._iou_weight * iou_cost

        return cost
