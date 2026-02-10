"""End-to-end integration tests for tactical network layer.

Tests multi-node scenarios with full stack: transport → pubsub → bridge →
composite fusion, verifying that distributed nodes can share tracks and
build a unified air picture.
"""

import numpy as np
import pytest
from dataclasses import dataclass, field

from sentinel.core.types import MessageType, NodeRole, NodeState, SensorType
from sentinel.network.bridge import NetworkBridge, RemoteTrack
from sentinel.network.composite_fusion import CompositeFusion
from sentinel.network.discovery import MeshDiscovery
from sentinel.network.messages import NetworkMessage, make_track_report
from sentinel.network.node import NetworkNode
from sentinel.network.pubsub import PubSubBroker
from sentinel.network.transport import (
    DegradationProfile,
    DEGRADATION_PRESETS,
    SimulatedTransport,
    TransportHub,
)
from sentinel.fusion.multi_sensor_fusion import EnhancedFusedTrack


# ===================================================================
# Helpers
# ===================================================================


def _make_full_node(
    hub: TransportHub,
    node_id: str,
    role: NodeRole = NodeRole.SENSOR,
    capabilities: list[str] | None = None,
    profile: DegradationProfile | None = None,
    seed: int = 42,
) -> dict:
    """Create a fully configured network node with all layers."""
    transport = SimulatedTransport(
        node_id, hub,
        profile=profile or DegradationProfile(packet_loss_rate=0.0),
        seed=seed,
    )
    node = NetworkNode(
        node_id, role,
        capabilities=capabilities or ["radar"],
        heartbeat_interval_s=2.0,
        heartbeat_timeout_s=6.0,
    )
    node.start()
    broker = PubSubBroker()
    bridge = NetworkBridge(node, transport, broker)
    discovery = MeshDiscovery(node, transport)
    composite = CompositeFusion(distance_gate=50.0, stale_threshold_s=5.0)
    return {
        "transport": transport,
        "node": node,
        "broker": broker,
        "bridge": bridge,
        "discovery": discovery,
        "composite": composite,
    }


def _make_local_track(
    fused_id: str = "R-001",
    position: tuple = (1000.0, 2000.0),
    threat_level: str = "MEDIUM",
    confidence: float = 0.7,
) -> EnhancedFusedTrack:
    return EnhancedFusedTrack(
        fused_id=fused_id,
        position_m=np.array(position),
        threat_level=threat_level,
        confidence=confidence,
    )


# ===================================================================
# Two-node scenarios
# ===================================================================


class TestTwoNodeNetwork:
    def test_track_sharing(self):
        """Node A publishes tracks, Node B receives and processes them."""
        hub = TransportHub()
        a = _make_full_node(hub, "RADAR-NORTH", seed=42)
        b = _make_full_node(hub, "RADAR-SOUTH", seed=43)

        # A has a local track
        track = _make_local_track("R-001", (5000.0, 1000.0), "HIGH", 0.85)
        a["bridge"].publish_tracks([track], current_time=1000.0)

        # B processes incoming
        b["bridge"].process_incoming()
        remote = b["bridge"].get_remote_tracks()
        assert "RADAR-NORTH" in remote
        assert len(remote["RADAR-NORTH"]) == 1
        assert remote["RADAR-NORTH"][0].threat_level == "HIGH"

    def test_composite_fusion_merge(self):
        """Two nodes see the same target — composite fusion merges them."""
        hub = TransportHub()
        a = _make_full_node(hub, "RADAR-NORTH", seed=42)
        b = _make_full_node(hub, "RADAR-SOUTH", seed=43)

        # A publishes a track near [5000, 1000]
        track_a = _make_local_track("R-001", (5000.0, 1000.0), "HIGH", 0.85)
        a["bridge"].publish_tracks([track_a], current_time=1000.0)

        # B has its own local track near the same position
        track_b = _make_local_track("R-010", (5010.0, 1005.0), "MEDIUM", 0.7)

        # B processes remote and runs composite fusion
        b["bridge"].process_incoming()
        remote = b["bridge"].get_remote_tracks()
        result = b["composite"].merge([track_b], remote, current_time=1000.0)

        # Should merge into single track (not two)
        assert len(result) == 1
        # Threat should be HIGH (max of HIGH and MEDIUM)
        assert result[0].threat_level == "HIGH"

    def test_bidirectional_exchange(self):
        """Both nodes share tracks bidirectionally."""
        hub = TransportHub()
        a = _make_full_node(hub, "NODE-A", seed=42)
        b = _make_full_node(hub, "NODE-B", seed=43)

        # A and B each publish different tracks
        track_a = _make_local_track("A-001", (1000.0, 0.0))
        track_b = _make_local_track("B-001", (0.0, 5000.0))
        a["bridge"].publish_tracks([track_a], 1000.0)
        b["bridge"].publish_tracks([track_b], 1000.0)

        # Both process incoming
        a["bridge"].process_incoming()
        b["bridge"].process_incoming()

        # A should see B's track and vice versa
        assert "NODE-B" in a["bridge"].get_remote_tracks()
        assert "NODE-A" in b["bridge"].get_remote_tracks()


# ===================================================================
# Three-node scenarios
# ===================================================================


class TestThreeNodeNetwork:
    def test_three_node_mesh(self):
        """Three nodes form a mesh and share tracks."""
        hub = TransportHub()
        a = _make_full_node(hub, "NORTH", capabilities=["radar"], seed=42)
        b = _make_full_node(hub, "SOUTH", capabilities=["thermal"], seed=43)
        c = _make_full_node(hub, "EAST", capabilities=["quantum_radar"], seed=44)

        # Each publishes tracks
        a["bridge"].publish_tracks([_make_local_track("N-001", (1000, 0))], 1000.0)
        b["bridge"].publish_tracks([_make_local_track("S-001", (0, 1000))], 1000.0)
        c["bridge"].publish_tracks([_make_local_track("E-001", (1000, 1000))], 1000.0)

        # All process incoming
        for node in [a, b, c]:
            node["bridge"].process_incoming()

        # Each should see 2 remote nodes
        assert len(a["bridge"].get_remote_tracks()) == 2
        assert len(b["bridge"].get_remote_tracks()) == 2
        assert len(c["bridge"].get_remote_tracks()) == 2

    def test_discovery_and_track_sharing(self):
        """Nodes discover each other via heartbeat then share tracks."""
        hub = TransportHub()
        a = _make_full_node(hub, "A", seed=42)
        b = _make_full_node(hub, "B", seed=43)

        # Discovery phase
        events_a = a["discovery"].step(0.0)
        events_b = b["discovery"].step(0.0)

        # Process discovery messages
        events_a2 = a["discovery"].step(0.1)
        events_b2 = b["discovery"].step(0.1)

        # After discovery, share tracks
        a["bridge"].publish_tracks([_make_local_track("A-001")], 1.0)
        b["bridge"].process_incoming()
        assert len(b["bridge"].get_remote_tracks()) > 0


# ===================================================================
# Degraded network scenarios
# ===================================================================


class TestDegradedNetwork:
    def test_packet_loss(self):
        """With high packet loss, some tracks may not arrive."""
        hub = TransportHub()
        profile = DegradationProfile(packet_loss_rate=0.5)
        a = _make_full_node(hub, "A", profile=profile, seed=42)
        b = _make_full_node(hub, "B", seed=43)

        # Send many track updates
        received_count = 0
        for i in range(100):
            track = _make_local_track(f"R-{i:03d}", (float(i * 100), 0.0))
            a["bridge"].publish_tracks([track], float(1000 + i))
            b["bridge"].process_incoming()

        remote = b["bridge"].get_remote_tracks()
        if "A" in remote:
            received_count = len(remote["A"])

        # With 50% loss, should receive roughly 30-70 tracks
        assert 20 < received_count < 80

    def test_per_link_degradation(self):
        """Specific link degradation while other links remain healthy."""
        hub = TransportHub()
        a = _make_full_node(hub, "A", seed=42)
        b = _make_full_node(hub, "B", seed=43)
        c = _make_full_node(hub, "C", seed=44)

        # A→B link is broken, A→C link is fine
        hub.set_link_profile("A", "B", DegradationProfile(packet_loss_rate=1.0))

        a["bridge"].publish_tracks([_make_local_track("A-001")], 1000.0)
        b["bridge"].process_incoming()
        c["bridge"].process_incoming()

        # B should get nothing, C should get the track
        assert len(b["bridge"].get_remote_tracks()) == 0
        assert len(c["bridge"].get_remote_tracks()) == 1

    def test_node_failure_and_recovery(self):
        """Node goes offline and recovers."""
        hub = TransportHub()
        a = _make_full_node(hub, "A", seed=42)
        b = _make_full_node(hub, "B", seed=43)

        # Initial discovery
        a["discovery"].step(0.0)
        b["discovery"].step(0.0)
        a["discovery"].step(0.5)
        b["discovery"].step(0.5)

        # B goes offline (heartbeat timeout)
        events = a["discovery"].process_incoming(10.0)  # 10s, timeout=6s
        offline = [e for e in events if e.event_type == "peer_left"]
        assert len(offline) == 1

        # B comes back
        b["discovery"].step(11.0)
        events = a["discovery"].process_incoming(11.0)
        recovered = [e for e in events if e.event_type == "peer_recovered"]
        assert len(recovered) == 1


# ===================================================================
# Config validation
# ===================================================================


class TestNetworkConfig:
    def test_config_schema_validates(self):
        from sentinel.core.config_schema import NetworkConfigSchema
        cfg = NetworkConfigSchema()
        assert cfg.enabled is False
        assert cfg.node_id == "LOCAL"
        assert cfg.transport.latency_ms == 5.0
        assert cfg.pubsub.track_publish_hz == 1.0
        assert cfg.discovery.heartbeat_timeout_s == 6.0
        assert cfg.composite_fusion.distance_gate == 50.0

    def test_root_config_includes_network(self):
        from sentinel.core.config_schema import SentinelRootConfig
        cfg = SentinelRootConfig()
        assert hasattr(cfg, "network")
        assert cfg.network.enabled is False
