"""Unit tests for composite fusion (remote track merging)."""

import numpy as np
import pytest
from dataclasses import dataclass, field

from sentinel.core.types import SensorType
from sentinel.network.bridge import RemoteTrack
from sentinel.network.composite_fusion import CompositeFusion, CompositeTrackInfo


# ===================================================================
# Helpers
# ===================================================================


@dataclass
class MockLocalTrack:
    """Mock local fused track for testing composite fusion."""
    fused_id: str = "L-001"
    position_m: np.ndarray = field(default_factory=lambda: np.array([1000.0, 2000.0]))
    velocity_mps: float = 100.0
    fused_covariance: np.ndarray | None = None
    sensor_sources: set = field(default_factory=lambda: {SensorType.RADAR})
    threat_level: str = "MEDIUM"
    confidence: float = 0.7
    iff_identification: str = "unknown"
    engagement_auth: str = "weapons_hold"


def _make_remote(
    track_id: str = "NODE-B:R-001",
    position: tuple = (1010.0, 2005.0),
    velocity: tuple = (-50.0, 10.0),
    source_node: str = "NODE-B",
    threat_level: str = "MEDIUM",
    confidence: float = 0.6,
    update_time: float = 1000.0,
    covariance: np.ndarray | None = None,
) -> RemoteTrack:
    return RemoteTrack(
        track_id=track_id,
        source_node=source_node,
        position=np.array(position),
        velocity=np.array(velocity),
        covariance=covariance,
        threat_level=threat_level,
        confidence=confidence,
        update_time=update_time,
    )


# ===================================================================
# Basic merge behavior
# ===================================================================


class TestCompositeFusionBasic:
    def test_empty_remote(self):
        cf = CompositeFusion()
        local = [MockLocalTrack()]
        result = cf.merge(local, {}, current_time=1000.0)
        assert len(result) == 1
        assert result[0].fused_id == "L-001"

    def test_empty_local(self):
        cf = CompositeFusion()
        remote = {"NODE-B": [_make_remote()]}
        result = cf.merge([], remote, current_time=1000.0)
        assert len(result) == 1
        assert result[0].fused_id == "NODE-B:R-001"

    def test_both_empty(self):
        cf = CompositeFusion()
        result = cf.merge([], {}, current_time=1000.0)
        assert result == []


# ===================================================================
# Track matching
# ===================================================================


class TestTrackMatching:
    def test_close_tracks_merge(self):
        """Tracks at nearby positions should be merged."""
        cf = CompositeFusion(distance_gate=100.0, default_covariance_scale=10000.0)
        local = [MockLocalTrack(position_m=np.array([1000.0, 2000.0]))]
        remote = {"NODE-B": [_make_remote(position=(1005.0, 2005.0))]}
        result = cf.merge(local, remote, current_time=1000.0)
        # Should merge into single track (not add remote as new)
        assert len(result) == 1

    def test_far_tracks_separate(self):
        """Tracks far apart should remain separate."""
        cf = CompositeFusion(distance_gate=5.0, default_covariance_scale=1.0)
        local = [MockLocalTrack(position_m=np.array([0.0, 0.0]))]
        remote = {"NODE-B": [_make_remote(position=(50000.0, 50000.0))]}
        result = cf.merge(local, remote, current_time=1000.0)
        assert len(result) == 2  # local + remote

    def test_multiple_local_multiple_remote(self):
        cf = CompositeFusion(distance_gate=100.0, default_covariance_scale=10000.0)
        local = [
            MockLocalTrack(fused_id="L-001", position_m=np.array([1000.0, 2000.0])),
            MockLocalTrack(fused_id="L-002", position_m=np.array([5000.0, 3000.0])),
        ]
        remote = {"NODE-B": [
            _make_remote(track_id="B:R-001", position=(1005.0, 2005.0)),
            _make_remote(track_id="B:R-002", position=(5005.0, 3005.0)),
        ]}
        result = cf.merge(local, remote, current_time=1000.0)
        # Both should match → still 2 tracks
        assert len(result) == 2


# ===================================================================
# Stale track filtering
# ===================================================================


class TestStaleTracking:
    def test_stale_remote_filtered(self):
        cf = CompositeFusion(stale_threshold_s=5.0)
        local = [MockLocalTrack()]
        remote = {"NODE-B": [_make_remote(update_time=990.0)]}  # 10s old at t=1000
        result = cf.merge(local, remote, current_time=1000.0)
        # Stale remote should be filtered, only local remains
        assert len(result) == 1
        assert result[0].fused_id == "L-001"

    def test_fresh_remote_kept(self):
        cf = CompositeFusion(stale_threshold_s=5.0, distance_gate=5.0, default_covariance_scale=1.0)
        local = [MockLocalTrack(position_m=np.array([0.0, 0.0]))]
        remote = {"NODE-B": [_make_remote(
            position=(50000.0, 50000.0), update_time=998.0  # 2s old
        )]}
        result = cf.merge(local, remote, current_time=1000.0)
        assert len(result) == 2  # both kept


# ===================================================================
# Merge behavior
# ===================================================================


class TestMergeBehavior:
    def test_threat_level_takes_max(self):
        """Higher threat from remote should override local."""
        cf = CompositeFusion(distance_gate=100.0, default_covariance_scale=10000.0)
        local = [MockLocalTrack(threat_level="MEDIUM")]
        remote = {"NODE-B": [_make_remote(threat_level="HIGH")]}
        result = cf.merge(local, remote, current_time=1000.0)
        assert result[0].threat_level == "HIGH"

    def test_threat_level_keeps_local_when_higher(self):
        cf = CompositeFusion(distance_gate=100.0, default_covariance_scale=10000.0)
        local = [MockLocalTrack(threat_level="CRITICAL")]
        remote = {"NODE-B": [_make_remote(threat_level="MEDIUM")]}
        result = cf.merge(local, remote, current_time=1000.0)
        assert result[0].threat_level == "CRITICAL"

    def test_prefer_local_bias(self):
        """With prefer_local=True, local position should dominate."""
        cf = CompositeFusion(
            distance_gate=100.0,
            prefer_local=True,
            default_covariance_scale=10000.0,
        )
        local_pos = np.array([1000.0, 2000.0])
        remote_pos = np.array([1100.0, 2100.0])
        local = [MockLocalTrack(position_m=local_pos.copy(), confidence=0.5)]
        remote = {"NODE-B": [_make_remote(position=tuple(remote_pos), confidence=0.5)]}
        result = cf.merge(local, remote, current_time=1000.0)
        merged_pos = result[0].position_m
        # Should be closer to local position due to prefer_local
        local_dist = np.linalg.norm(merged_pos - local_pos)
        remote_dist = np.linalg.norm(merged_pos - remote_pos)
        assert local_dist < remote_dist


# ===================================================================
# Composite track info
# ===================================================================


class TestCompositeInfo:
    def test_merged_track_has_info(self):
        cf = CompositeFusion(distance_gate=100.0, default_covariance_scale=10000.0)
        local = [MockLocalTrack(fused_id="L-001")]
        remote = {"NODE-B": [_make_remote()]}
        cf.merge(local, remote, current_time=1000.0)
        info = cf.get_composite_info("L-001")
        assert info is not None
        assert "NODE-B" in info.contributing_nodes
        assert not info.is_remote_only

    def test_remote_only_track_has_info(self):
        cf = CompositeFusion(distance_gate=5.0, default_covariance_scale=1.0)
        local = [MockLocalTrack(position_m=np.array([0.0, 0.0]))]
        remote = {"NODE-C": [_make_remote(
            track_id="C:R-005",
            position=(50000.0, 50000.0),
            source_node="NODE-C",
        )]}
        cf.merge(local, remote, current_time=1000.0)
        info = cf.get_composite_info("C:R-005")
        assert info is not None
        assert info.is_remote_only
        assert "NODE-C" in info.contributing_nodes

    def test_no_info_for_unknown(self):
        cf = CompositeFusion()
        assert cf.get_composite_info("nonexistent") is None


# ===================================================================
# Multi-node fusion
# ===================================================================


class TestMultiNodeFusion:
    def test_tracks_from_multiple_nodes(self):
        cf = CompositeFusion(distance_gate=5.0, default_covariance_scale=1.0)
        local = [MockLocalTrack(position_m=np.array([0.0, 0.0]))]
        remote = {
            "NODE-B": [_make_remote(
                track_id="B:R-001", position=(50000.0, 0.0), source_node="NODE-B"
            )],
            "NODE-C": [_make_remote(
                track_id="C:R-001", position=(0.0, 50000.0), source_node="NODE-C"
            )],
        }
        result = cf.merge(local, remote, current_time=1000.0)
        # All 3 should be separate (far apart)
        assert len(result) == 3

    def test_with_covariance(self):
        """Tracks with explicit covariance should use it for matching."""
        cf = CompositeFusion(distance_gate=50.0)
        cov = np.eye(2) * 100.0
        local = [MockLocalTrack(
            position_m=np.array([1000.0, 2000.0]),
            fused_covariance=cov,
        )]
        remote = {"NODE-B": [_make_remote(
            position=(1005.0, 2005.0),
            covariance=np.eye(2) * 100.0,
        )]}
        result = cf.merge(local, remote, current_time=1000.0)
        assert len(result) == 1  # Should match
