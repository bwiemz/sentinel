"""Radar detection-to-track association using Hungarian algorithm.

Cost matrix based on Mahalanobis distance in polar measurement space.
Supports velocity gating and cascaded association.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection, TrackState
from sentinel.tracking.association import INFEASIBLE, AssociationResult
from sentinel.tracking.radar_track import RadarTrack
from sentinel.utils.coords import azimuth_deg_to_rad


class RadarAssociator:
    """Hungarian algorithm association for radar tracks.

    Args:
        gate_threshold: Chi-squared gating threshold (2 DOF).
        velocity_gate_mps: Max radial velocity difference for gating (None=disabled).
        cascaded: If True, confirmed tracks get first pick, then tentative/coasting.
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,
        velocity_gate_mps: float | None = None,
        cascaded: bool = False,
    ):
        self._gate_threshold = gate_threshold
        self._velocity_gate = velocity_gate_mps
        self._cascaded = cascaded

    def associate(self, tracks: list[RadarTrack], detections: list[Detection]) -> AssociationResult:
        """Optimal assignment between radar tracks and detections."""
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
        tracks: list[RadarTrack],
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

        # Build cost matrix for this subset
        n_tracks = len(track_indices)
        n_dets = len(det_list)
        cost = np.full((n_tracks, n_dets), INFEASIBLE)

        for i, ti in enumerate(track_indices):
            track = tracks[ti]
            for j, dj in enumerate(det_list):
                det = detections[dj]
                if det.range_m is None or det.azimuth_deg is None:
                    continue

                z = np.array([det.range_m, azimuth_deg_to_rad(det.azimuth_deg)])
                maha = track.ekf.gating_distance(z)

                if maha > self._gate_threshold:
                    continue

                # Velocity gating
                if self._velocity_gate is not None and det.velocity_mps is not None:
                    track_speed = float(np.linalg.norm(track.velocity))
                    if abs(det.velocity_mps - track_speed) > self._velocity_gate:
                        continue

                cost[i, j] = maha

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

    def _cascaded_associate(self, tracks: list[RadarTrack], detections: list[Detection]) -> AssociationResult:
        """Two-pass cascaded association: confirmed first, then others."""
        confirmed_idx = [i for i, t in enumerate(tracks) if t.state == TrackState.CONFIRMED]
        other_idx = [i for i, t in enumerate(tracks) if t.state != TrackState.CONFIRMED]

        # Pass 1: confirmed tracks
        result1 = self._single_pass(tracks, detections, confirmed_idx)
        remaining_dets = set(result1.unmatched_detections)

        # Pass 2: tentative/coasting tracks with remaining detections
        result2 = self._single_pass(tracks, detections, other_idx, available_dets=remaining_dets)

        return AssociationResult(
            matched_pairs=result1.matched_pairs + result2.matched_pairs,
            unmatched_tracks=sorted(set(result1.unmatched_tracks) | set(result2.unmatched_tracks)),
            unmatched_detections=sorted(set(result2.unmatched_detections)),
        )

    def _build_cost_matrix(self, tracks: list[RadarTrack], detections: list[Detection]) -> np.ndarray:
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
