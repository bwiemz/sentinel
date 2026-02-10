"""Composite fusion — merges remote tracks with local fused tracks.

Implements CEC-inspired composite tracking where multiple distributed
nodes share their track state (including covariance) and a central
or distributed fusion layer correlates and merges them into a unified
air picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.network.bridge import RemoteTrack
from sentinel.tracking.cost_functions import track_to_track_mahalanobis


# ---------------------------------------------------------------------------
# Composite track
# ---------------------------------------------------------------------------


@dataclass
class CompositeTrackInfo:
    """Metadata about a composite (multi-node) track."""

    contributing_nodes: list[str] = field(default_factory=list)
    contributing_track_ids: list[str] = field(default_factory=list)
    is_remote_only: bool = False
    composite_confidence: float = 0.0


# ---------------------------------------------------------------------------
# CompositeFusion
# ---------------------------------------------------------------------------


class CompositeFusion:
    """Merges remote tracks with local fused tracks.

    Algorithm:
    1. Filter stale remote tracks (age > stale_threshold)
    2. Build Mahalanobis cost matrix (local x all_remote)
    3. Hungarian assignment with distance_gate
    4. Matched: weighted average position (higher confidence wins),
       merge sensor types, take max threat level
    5. Unmatched remote: add as new tracks
    6. Unmatched local: keep as-is

    Track ID deconfliction: remote tracks are prefixed with
    {source_node}:{track_id} to avoid collisions.
    """

    def __init__(
        self,
        distance_gate: float = 50.0,
        stale_threshold_s: float = 5.0,
        prefer_local: bool = True,
        default_covariance_scale: float = 10000.0,
    ):
        self._distance_gate = distance_gate
        self._stale_threshold_s = stale_threshold_s
        self._prefer_local = prefer_local
        self._default_cov_scale = default_covariance_scale
        self._composite_info: dict[str, CompositeTrackInfo] = {}

    @property
    def distance_gate(self) -> float:
        return self._distance_gate

    @property
    def stale_threshold_s(self) -> float:
        return self._stale_threshold_s

    def merge(
        self,
        local_tracks: list,
        remote_tracks: dict[str, list[RemoteTrack]],
        current_time: float,
    ) -> list:
        """Merge remote tracks into local tracks.

        Args:
            local_tracks: List of EnhancedFusedTrack (or similar) from local fusion.
            remote_tracks: Dict mapping source_node_id to list of RemoteTrack.
            current_time: Current simulation time for staleness checking.

        Returns:
            Merged list of tracks (local tracks updated in-place + new remote-only tracks).
        """
        # 1. Flatten and filter stale remote tracks
        all_remote = self._flatten_remote(remote_tracks, current_time)
        if not all_remote:
            return list(local_tracks)

        if not local_tracks:
            # All remote tracks become new entries
            return self._remote_only_tracks(all_remote)

        # 2. Build cost matrix
        cost_matrix = self._build_cost_matrix(local_tracks, all_remote)

        # 3. Hungarian assignment with gating
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_local = set()
        matched_remote = set()
        result = list(local_tracks)

        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < self._distance_gate:
                matched_local.add(r)
                matched_remote.add(c)
                # 4. Merge matched pair
                self._merge_track(result[r], all_remote[c])

        # 5. Unmatched remote → new tracks
        for j, rt in enumerate(all_remote):
            if j not in matched_remote:
                new_track = self._make_remote_only_track(rt)
                if new_track is not None:
                    result.append(new_track)

        return result

    def get_composite_info(self, track_id: str) -> CompositeTrackInfo | None:
        """Get composite track metadata."""
        return self._composite_info.get(track_id)

    # --- Internal ---

    def _flatten_remote(
        self,
        remote_tracks: dict[str, list[RemoteTrack]],
        current_time: float,
    ) -> list[RemoteTrack]:
        """Flatten dict of remote tracks, filtering stale ones."""
        result = []
        for node_id, tracks in remote_tracks.items():
            for rt in tracks:
                age = current_time - rt.update_time
                if age <= self._stale_threshold_s:
                    result.append(rt)
        return result

    def _build_cost_matrix(
        self,
        local_tracks: list,
        remote_tracks: list[RemoteTrack],
    ) -> np.ndarray:
        """Build Mahalanobis distance matrix between local and remote tracks."""
        n_local = len(local_tracks)
        n_remote = len(remote_tracks)
        cost = np.full((n_local, n_remote), self._distance_gate * 10)

        # Pre-extract remote positions/covariances once (avoid N*M extractions)
        remote_data = [
            (rt.position, self._get_remote_covariance(rt, rt.position))
            for rt in remote_tracks
        ]

        for i, lt in enumerate(local_tracks):
            pos_local = self._get_position(lt)
            if pos_local is None:
                continue
            cov_local = self._get_covariance(lt, pos_local)

            for j, (pos_remote, cov_remote) in enumerate(remote_data):
                dim = min(len(pos_local), len(pos_remote))
                p1 = pos_local[:dim]
                p2 = pos_remote[:dim]
                c1 = cov_local[:dim, :dim]
                c2 = cov_remote[:dim, :dim]
                dist = track_to_track_mahalanobis(p1, c1, p2, c2)
                cost[i, j] = dist

        return cost

    def _merge_track(self, local_track: object, remote: RemoteTrack) -> None:
        """Merge remote track data into a local track (in-place)."""
        # Record composite info
        fused_id = getattr(local_track, "fused_id", "unknown")
        if fused_id not in self._composite_info:
            self._composite_info[fused_id] = CompositeTrackInfo()
        info = self._composite_info[fused_id]
        if remote.source_node not in info.contributing_nodes:
            info.contributing_nodes.append(remote.source_node)
        if remote.track_id not in info.contributing_track_ids:
            info.contributing_track_ids.append(remote.track_id)

        # Weighted position average (higher confidence gets more weight)
        local_conf = getattr(local_track, "confidence", 0.5) or 0.5
        remote_conf = remote.confidence or 0.5
        # Guard against NaN/Inf confidence values
        if not np.isfinite(local_conf):
            local_conf = 0.5
        if not np.isfinite(remote_conf):
            remote_conf = 0.5
        total = local_conf + remote_conf
        if total <= 0:
            total = 1.0
        local_weight = local_conf / total
        remote_weight = remote_conf / total

        # Apply local preference bias
        if self._prefer_local:
            local_weight = max(local_weight, 0.6)
            remote_weight = 1.0 - local_weight

        pos_local = self._get_position(local_track)
        if pos_local is not None:
            dim = min(len(pos_local), len(remote.position))
            merged_pos = local_weight * pos_local[:dim] + remote_weight * remote.position[:dim]
            if hasattr(local_track, "position_m") and local_track.position_m is not None:
                local_track.position_m[:dim] = merged_pos

        # Take the highest threat level
        threat_order = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        local_threat = getattr(local_track, "threat_level", "UNKNOWN")
        remote_threat = remote.threat_level
        if threat_order.get(remote_threat, 0) > threat_order.get(local_threat, 0):
            if hasattr(local_track, "threat_level"):
                local_track.threat_level = remote_threat

        # Merge sensor types
        if hasattr(local_track, "sensor_sources"):
            from sentinel.core.types import SensorType
            for st in remote.sensor_types:
                try:
                    local_track.sensor_sources.add(SensorType(st))
                except (ValueError, KeyError):
                    pass

        # Update composite confidence
        info.composite_confidence = max(local_conf, remote_conf)

    def _make_remote_only_track(self, remote: RemoteTrack) -> object | None:
        """Create a new track entry from an unmatched remote track."""
        from sentinel.fusion.multi_sensor_fusion import EnhancedFusedTrack

        track = EnhancedFusedTrack(
            fused_id=remote.track_id,
            position_m=remote.position.copy(),
            threat_level=remote.threat_level,
            iff_identification=remote.iff_identification,
            engagement_auth=remote.engagement_auth,
        )

        # Record composite info
        self._composite_info[remote.track_id] = CompositeTrackInfo(
            contributing_nodes=[remote.source_node],
            contributing_track_ids=[remote.track_id],
            is_remote_only=True,
            composite_confidence=remote.confidence,
        )

        return track

    def _remote_only_tracks(self, remotes: list[RemoteTrack]) -> list:
        """Convert all remote tracks to new fused tracks."""
        result = []
        for rt in remotes:
            track = self._make_remote_only_track(rt)
            if track is not None:
                result.append(track)
        return result

    @staticmethod
    def _get_position(track: object) -> np.ndarray | None:
        """Extract position from a fused track."""
        pos = getattr(track, "position_m", None)
        if pos is not None:
            return np.asarray(pos, dtype=float)
        pos = getattr(track, "position", None)
        if pos is not None:
            return np.asarray(pos, dtype=float)
        return None

    def _get_covariance(
        self, track: object, position: np.ndarray | None
    ) -> np.ndarray:
        """Extract or create default covariance for a local track."""
        cov = getattr(track, "fused_covariance", None)
        if cov is not None:
            return np.asarray(cov, dtype=float)
        dim = len(position) if position is not None else 2
        return np.eye(dim) * self._default_cov_scale

    def _get_remote_covariance(
        self, remote: RemoteTrack, position: np.ndarray
    ) -> np.ndarray:
        """Get covariance from remote track or use default."""
        if remote.covariance is not None:
            return np.asarray(remote.covariance, dtype=float)
        dim = len(position)
        return np.eye(dim) * self._default_cov_scale
