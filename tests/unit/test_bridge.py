"""Unit tests for network bridge (SENTINEL object ↔ network message conversion)."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field

from sentinel.core.types import MessageType, NodeRole, SensorType
from sentinel.network.bridge import NetworkBridge, RemoteTrack
from sentinel.network.messages import NetworkMessage, make_track_report
from sentinel.network.node import NetworkNode
from sentinel.network.pubsub import PubSubBroker
from sentinel.network.transport import DegradationProfile, SimulatedTransport, TransportHub


# ===================================================================
# Helpers
# ===================================================================


def _make_bridge(node_id: str = "LOCAL") -> tuple[TransportHub, NetworkBridge]:
    """Create a bridge with its transport + broker infrastructure."""
    hub = TransportHub()
    transport = SimulatedTransport(
        node_id, hub,
        profile=DegradationProfile(packet_loss_rate=0.0),
        seed=42,
    )
    node = NetworkNode(node_id, NodeRole.SENSOR, capabilities=["radar"])
    node.start()
    broker = PubSubBroker()
    bridge = NetworkBridge(node, transport, broker)
    return hub, bridge


def _make_two_bridges() -> tuple[TransportHub, NetworkBridge, NetworkBridge]:
    """Create two connected bridges."""
    hub = TransportHub()
    t_a = SimulatedTransport("NODE-A", hub, DegradationProfile(packet_loss_rate=0.0), seed=42)
    t_b = SimulatedTransport("NODE-B", hub, DegradationProfile(packet_loss_rate=0.0), seed=43)
    node_a = NetworkNode("NODE-A", NodeRole.SENSOR, capabilities=["radar"])
    node_b = NetworkNode("NODE-B", NodeRole.SENSOR, capabilities=["thermal"])
    node_a.start()
    node_b.start()
    broker_a = PubSubBroker()
    broker_b = PubSubBroker()
    bridge_a = NetworkBridge(node_a, t_a, broker_a)
    bridge_b = NetworkBridge(node_b, t_b, broker_b)
    return hub, bridge_a, bridge_b


@dataclass
class MockFusedTrack:
    """Mock fused track for testing bridge conversion."""
    fused_id: str = "R-001"
    position_m: np.ndarray = field(default_factory=lambda: np.array([1000.0, 2000.0]))
    velocity_mps: float = 100.0
    fused_covariance: np.ndarray | None = None
    position_geo: dict | None = None
    sensor_sources: set = field(default_factory=lambda: {SensorType.RADAR})
    threat_level: str = "HIGH"
    iff_identification: str = "hostile"
    engagement_auth: str = "weapons_tight"
    confidence: float = 0.85


# ===================================================================
# RemoteTrack
# ===================================================================


class TestRemoteTrack:
    def test_create(self):
        rt = RemoteTrack(
            track_id="NODE-B:R-001",
            source_node="NODE-B",
            position=np.array([1000.0, 2000.0]),
            velocity=np.array([-50.0, 10.0]),
        )
        assert rt.track_id == "NODE-B:R-001"
        assert rt.source_node == "NODE-B"
        assert rt.threat_level == "UNKNOWN"

    def test_defaults(self):
        rt = RemoteTrack(
            track_id="T", source_node="N",
            position=np.zeros(2), velocity=np.zeros(2),
        )
        assert rt.iff_identification == "unknown"
        assert rt.engagement_auth == "weapons_hold"
        assert rt.confidence == 0.0


# ===================================================================
# NetworkBridge publishing
# ===================================================================


class TestBridgePublishing:
    def test_publish_tracks(self):
        hub, bridge = _make_bridge("NODE-A")
        # Add a peer to receive broadcasts
        peer_t = SimulatedTransport("PEER", hub, seed=99)
        track = MockFusedTrack()
        count = bridge.publish_tracks([track], current_time=1000.0)
        assert count == 1
        # Check peer received the message
        msg_data = peer_t.recv_sync()
        assert msg_data is not None
        _, raw = msg_data
        msg = NetworkMessage.deserialize(raw)
        assert msg.msg_type == MessageType.TRACK_REPORT
        assert msg.payload["track_id"] == "NODE-A:R-001"
        assert msg.payload["threat_level"] == "HIGH"

    def test_publish_tracks_empty(self):
        hub, bridge = _make_bridge()
        count = bridge.publish_tracks([], current_time=1000.0)
        assert count == 0

    def test_publish_iff(self):
        hub, bridge = _make_bridge("NODE-A")
        peer_t = SimulatedTransport("PEER", hub, seed=99)

        # Mock IFF result
        iff_result = MagicMock()
        iff_result.identification = MagicMock(value="friendly")
        iff_result.confidence = 0.95
        iff_result.mode_3a_code = "1200"
        iff_result.mode_s_address = None
        iff_result.spoof_indicators = 0

        count = bridge.publish_iff({"TGT-01": iff_result}, current_time=1000.0)
        assert count == 1
        msg_data = peer_t.recv_sync()
        assert msg_data is not None
        _, raw = msg_data
        msg = NetworkMessage.deserialize(raw)
        assert msg.msg_type == MessageType.IFF_REPORT
        assert msg.payload["identification"] == "friendly"

    def test_publish_engagement(self):
        hub, bridge = _make_bridge("NODE-A")
        peer_t = SimulatedTransport("PEER", hub, seed=99)
        bridge.publish_engagement("T-005", "weapons_free", 1000.0, "hostile confirmed")
        msg_data = peer_t.recv_sync()
        assert msg_data is not None
        _, raw = msg_data
        msg = NetworkMessage.deserialize(raw)
        assert msg.msg_type == MessageType.ENGAGEMENT_STATUS
        assert msg.payload["engagement_auth"] == "weapons_free"


# ===================================================================
# NetworkBridge receiving
# ===================================================================


class TestBridgeReceiving:
    def test_receive_track_report(self):
        hub, bridge_a, bridge_b = _make_two_bridges()
        track = MockFusedTrack(fused_id="R-001", threat_level="HIGH")
        bridge_a.publish_tracks([track], current_time=1000.0)
        # B processes incoming
        bridge_b.process_incoming()
        remote = bridge_b.get_remote_tracks()
        assert "NODE-A" in remote
        assert len(remote["NODE-A"]) == 1
        rt = remote["NODE-A"][0]
        assert rt.track_id == "NODE-A:R-001"
        assert rt.threat_level == "HIGH"

    def test_receive_iff_report(self):
        hub, bridge_a, bridge_b = _make_two_bridges()
        iff_result = MagicMock()
        iff_result.identification = MagicMock(value="hostile")
        iff_result.confidence = 0.8
        iff_result.mode_3a_code = None
        iff_result.mode_s_address = None
        iff_result.spoof_indicators = 2
        bridge_a.publish_iff({"TGT-02": iff_result}, 1000.0)
        bridge_b.process_incoming()
        remote_iff = bridge_b.get_remote_iff()
        assert "TGT-02" in remote_iff
        assert remote_iff["TGT-02"]["identification"] == "hostile"

    def test_receive_engagement_status(self):
        hub, bridge_a, bridge_b = _make_two_bridges()
        bridge_a.publish_engagement("T-005", "weapons_free", 1000.0)
        bridge_b.process_incoming()
        remote_eng = bridge_b.get_remote_engagement()
        assert remote_eng["T-005"] == "weapons_free"

    def test_ignores_own_messages(self):
        """Bridge should not buffer its own published messages."""
        hub, bridge = _make_bridge("NODE-A")
        track = MockFusedTrack()
        bridge.publish_tracks([track], 1000.0)
        # The broker callback should be invoked but filter own node
        remote = bridge.get_remote_tracks()
        assert len(remote) == 0

    def test_track_update_replaces(self):
        """Updated track with same ID should replace, not append."""
        hub, bridge_a, bridge_b = _make_two_bridges()
        t1 = MockFusedTrack(fused_id="R-001", threat_level="MEDIUM")
        bridge_a.publish_tracks([t1], 1000.0)
        bridge_b.process_incoming()
        t2 = MockFusedTrack(fused_id="R-001", threat_level="HIGH")
        bridge_a.publish_tracks([t2], 1001.0)
        bridge_b.process_incoming()
        remote = bridge_b.get_remote_tracks()
        assert len(remote["NODE-A"]) == 1
        assert remote["NODE-A"][0].threat_level == "HIGH"

    def test_clear_buffers(self):
        hub, bridge_a, bridge_b = _make_two_bridges()
        bridge_a.publish_tracks([MockFusedTrack()], 1000.0)
        bridge_b.process_incoming()
        assert len(bridge_b.get_remote_tracks()) > 0
        bridge_b.clear_buffers()
        assert len(bridge_b.get_remote_tracks()) == 0


# ===================================================================
# Track conversion details
# ===================================================================


class TestTrackConversion:
    def test_track_id_prefixed(self):
        hub, bridge = _make_bridge("RADAR-NORTH")
        peer_t = SimulatedTransport("PEER", hub, seed=99)
        track = MockFusedTrack(fused_id="R-007")
        bridge.publish_tracks([track], 1000.0)
        _, raw = peer_t.recv_sync()
        msg = NetworkMessage.deserialize(raw)
        assert msg.payload["track_id"] == "RADAR-NORTH:R-007"

    def test_geo_position_dict_format(self):
        hub, bridge = _make_bridge("N")
        peer_t = SimulatedTransport("PEER", hub, seed=99)
        track = MockFusedTrack(
            position_geo={"lat": 38.9, "lon": -77.0, "alt": 1000.0}
        )
        bridge.publish_tracks([track], 1000.0)
        _, raw = peer_t.recv_sync()
        msg = NetworkMessage.deserialize(raw)
        assert msg.payload["position_geo"] == [38.9, -77.0, 1000.0]

    def test_sensor_sources_serialized(self):
        hub, bridge = _make_bridge("N")
        peer_t = SimulatedTransport("PEER", hub, seed=99)
        track = MockFusedTrack(
            sensor_sources={SensorType.RADAR, SensorType.THERMAL}
        )
        bridge.publish_tracks([track], 1000.0)
        _, raw = peer_t.recv_sync()
        msg = NetworkMessage.deserialize(raw)
        assert set(msg.payload["sensor_types"]) == {"radar", "thermal"}
