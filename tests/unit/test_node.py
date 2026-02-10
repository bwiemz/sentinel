"""Unit tests for network node state management."""

import pytest

from sentinel.core.types import MessageType, NodeRole, NodeState
from sentinel.network.messages import NetworkMessage, make_heartbeat
from sentinel.network.node import NetworkNode, PeerInfo


class TestPeerInfo:
    def test_defaults(self):
        p = PeerInfo(node_id="PEER-1")
        assert p.node_id == "PEER-1"
        assert p.role == NodeRole.SENSOR
        assert p.state == NodeState.JOINING
        assert p.capabilities == []
        assert p.track_count == 0


class TestNetworkNode:
    def test_create(self):
        node = NetworkNode("RADAR-01", NodeRole.SENSOR, capabilities=["radar", "iff"])
        assert node.node_id == "RADAR-01"
        assert node.role == NodeRole.SENSOR
        assert node.capabilities == ["radar", "iff"]
        assert node.state == NodeState.JOINING

    def test_start(self):
        node = NetworkNode("N")
        node.start()
        assert node.state == NodeState.ACTIVE

    def test_stop(self):
        node = NetworkNode("N")
        node.start()
        node.stop()
        assert node.state == NodeState.OFFLINE

    def test_degraded_and_recover(self):
        node = NetworkNode("N")
        node.start()
        node.set_degraded()
        assert node.state == NodeState.DEGRADED
        assert not node.is_healthy
        node.recover()
        assert node.state == NodeState.ACTIVE
        assert node.is_healthy

    def test_degraded_only_from_active(self):
        node = NetworkNode("N")  # state = JOINING
        node.set_degraded()
        assert node.state == NodeState.JOINING  # unchanged

    def test_recover_only_from_degraded(self):
        node = NetworkNode("N")
        node.start()  # ACTIVE
        node.recover()  # no-op
        assert node.state == NodeState.ACTIVE

    def test_is_healthy(self):
        node = NetworkNode("N")
        assert node.is_healthy  # JOINING is healthy
        node.start()
        assert node.is_healthy  # ACTIVE is healthy
        node.set_degraded()
        assert not node.is_healthy
        node.recover()
        node.stop()
        assert not node.is_healthy  # OFFLINE is not healthy

    def test_track_count(self):
        node = NetworkNode("N")
        assert node.track_count == 0
        node.track_count = 5
        assert node.track_count == 5

    def test_make_heartbeat(self):
        node = NetworkNode("RADAR-01", NodeRole.SENSOR, capabilities=["radar"])
        node.start()
        node.track_count = 3
        msg = node.make_heartbeat(1000.0)
        assert msg.msg_type == MessageType.HEARTBEAT
        assert msg.source_node == "RADAR-01"
        assert msg.payload["state"] == "active"
        assert msg.payload["capabilities"] == ["radar"]
        assert msg.payload["track_count"] == 3

    def test_should_send_heartbeat(self):
        node = NetworkNode("N", heartbeat_interval_s=2.0)
        # First heartbeat should always be sent (initial _last_heartbeat_sent = -inf)
        assert node.should_send_heartbeat(0.0)
        node.make_heartbeat(0.0)
        assert not node.should_send_heartbeat(1.0)
        assert node.should_send_heartbeat(2.0)

    def test_update_peer(self):
        node = NetworkNode("N")
        hb = make_heartbeat(
            source_node="PEER-1",
            timestamp=1000.0,
            state="active",
            capabilities=["thermal"],
            track_count=2,
            uptime_s=300.0,
        )
        peer = node.update_peer(hb)
        assert peer.node_id == "PEER-1"
        assert peer.state == NodeState.ACTIVE
        assert peer.capabilities == ["thermal"]
        assert peer.track_count == 2
        assert peer.last_heartbeat == 1000.0

    def test_update_peer_repeated(self):
        node = NetworkNode("N")
        hb1 = make_heartbeat("PEER-1", 1000.0, state="active", track_count=1)
        hb2 = make_heartbeat("PEER-1", 1002.0, state="active", track_count=5)
        node.update_peer(hb1)
        node.update_peer(hb2)
        peers = node.peers
        assert len(peers) == 1
        assert peers["PEER-1"].track_count == 5

    def test_check_peer_health(self):
        node = NetworkNode("N", heartbeat_timeout_s=5.0)
        hb = make_heartbeat("PEER-1", 100.0, state="active")
        node.update_peer(hb)
        # At t=104, peer is fine (age=4s < timeout=5s)
        offline = node.check_peer_health(104.0)
        assert offline == []
        # At t=106, peer is stale (age=6s > timeout=5s)
        offline = node.check_peer_health(106.0)
        assert offline == ["PEER-1"]
        assert node.peers["PEER-1"].state == NodeState.OFFLINE

    def test_active_peer_count(self):
        node = NetworkNode("N")
        node.update_peer(make_heartbeat("P1", 0.0, state="active"))
        node.update_peer(make_heartbeat("P2", 0.0, state="active"))
        node.update_peer(make_heartbeat("P3", 0.0, state="degraded"))
        assert node.active_peer_count == 2

    def test_remove_peer(self):
        node = NetworkNode("N")
        node.update_peer(make_heartbeat("P1", 0.0))
        assert "P1" in node.peers
        node.remove_peer("P1")
        assert "P1" not in node.peers

    def test_get_peer(self):
        node = NetworkNode("N")
        assert node.get_peer("P1") is None
        node.update_peer(make_heartbeat("P1", 0.0))
        assert node.get_peer("P1") is not None

    def test_to_dict(self):
        node = NetworkNode("RADAR-01", NodeRole.SENSOR, capabilities=["radar"])
        node.start()
        node.track_count = 3
        d = node.to_dict()
        assert d["node_id"] == "RADAR-01"
        assert d["role"] == "sensor"
        assert d["state"] == "active"
        assert d["track_count"] == 3
