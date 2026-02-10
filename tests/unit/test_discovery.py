"""Unit tests for mesh network discovery."""

import pytest

from sentinel.core.types import NodeRole, NodeState
from sentinel.network.discovery import MeshDiscovery
from sentinel.network.node import NetworkNode
from sentinel.network.transport import DegradationProfile, SimulatedTransport, TransportHub


def _make_mesh(*node_names) -> tuple[TransportHub, dict[str, MeshDiscovery]]:
    """Create a mesh of discovery instances connected through a hub."""
    hub = TransportHub()
    meshes = {}
    for i, name in enumerate(node_names):
        transport = SimulatedTransport(
            name, hub,
            profile=DegradationProfile(packet_loss_rate=0.0),
            seed=42 + i,
        )
        node = NetworkNode(
            name,
            role=NodeRole.SENSOR,
            capabilities=["radar"],
            heartbeat_interval_s=2.0,
            heartbeat_timeout_s=6.0,
        )
        node.start()
        meshes[name] = MeshDiscovery(node, transport)
    return hub, meshes


class TestMeshDiscovery:
    def test_send_heartbeat(self):
        hub, meshes = _make_mesh("A", "B")
        count = meshes["A"].send_heartbeat(0.0)
        assert count == 1  # sent to B

    def test_discover_peer(self):
        hub, meshes = _make_mesh("A", "B")
        # A sends heartbeat at t=0
        meshes["A"].send_heartbeat(0.0)
        # B processes and discovers A
        events = meshes["B"].process_incoming(0.0)
        assert len(events) == 1
        assert events[0].event_type == "peer_joined"
        assert events[0].peer_id == "A"

    def test_bidirectional_discovery(self):
        hub, meshes = _make_mesh("A", "B")
        meshes["A"].send_heartbeat(0.0)
        meshes["B"].send_heartbeat(0.0)
        events_b = meshes["B"].process_incoming(0.0)
        events_a = meshes["A"].process_incoming(0.0)
        # Both should discover each other
        assert any(e.peer_id == "A" for e in events_b)
        assert any(e.peer_id == "B" for e in events_a)

    def test_three_node_discovery(self):
        hub, meshes = _make_mesh("A", "B", "C")
        # All send heartbeats
        for m in meshes.values():
            m.send_heartbeat(0.0)
        # All process incoming
        for m in meshes.values():
            events = m.process_incoming(0.0)
            assert len(events) == 2  # discovers 2 peers

    def test_peer_timeout(self):
        hub, meshes = _make_mesh("A", "B")
        # A discovers B
        meshes["B"].send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        assert "B" in meshes["A"].node.peers
        # B stops sending heartbeats, A checks at t=7 (timeout=6)
        events = meshes["A"].process_incoming(7.0)
        assert any(e.event_type == "peer_left" for e in events)
        assert meshes["A"].node.peers["B"].state == NodeState.OFFLINE

    def test_peer_recovery(self):
        hub, meshes = _make_mesh("A", "B")
        # Initial discovery
        meshes["B"].send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        # B goes offline (timeout)
        meshes["A"].process_incoming(7.0)
        assert meshes["A"].node.peers["B"].state == NodeState.OFFLINE
        # B comes back
        meshes["B"].send_heartbeat(8.0)
        events = meshes["A"].process_incoming(8.0)
        assert any(e.event_type == "peer_recovered" for e in events)

    def test_known_nodes(self):
        hub, meshes = _make_mesh("A", "B", "C")
        for m in meshes.values():
            m.send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        assert meshes["A"].known_nodes == {"B", "C"}

    def test_active_peers(self):
        hub, meshes = _make_mesh("A", "B", "C")
        for m in meshes.values():
            m.send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        assert set(meshes["A"].active_peers) == {"B", "C"}

    def test_event_log(self):
        hub, meshes = _make_mesh("A", "B")
        meshes["B"].send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        log = meshes["A"].event_log
        assert len(log) == 1
        assert log[0].event_type == "peer_joined"

    def test_step_convenience(self):
        hub, meshes = _make_mesh("A", "B")
        meshes["B"].send_heartbeat(0.0)
        events = meshes["A"].step(0.0)
        assert any(e.event_type == "peer_joined" for e in events)

    def test_topology(self):
        hub, meshes = _make_mesh("A", "B")
        meshes["B"].send_heartbeat(0.0)
        meshes["A"].process_incoming(0.0)
        topo = meshes["A"].get_topology()
        assert "B" in topo
        assert topo["B"]["state"] == "active"

    def test_degraded_peer_event(self):
        hub, meshes = _make_mesh("A", "B")
        # B reports degraded state
        meshes["B"].node.set_degraded()
        meshes["B"].send_heartbeat(0.0)
        events = meshes["A"].process_incoming(0.0)
        assert any(e.event_type == "peer_degraded" for e in events)
