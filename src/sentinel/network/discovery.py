"""Mesh network discovery and topology management.

Handles peer discovery, link quality tracking, topology changes,
and automatic reconnection for the tactical mesh network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sentinel.core.types import MessageType, NodeState
from sentinel.network.messages import NetworkMessage, make_heartbeat
from sentinel.network.node import NetworkNode, PeerInfo
from sentinel.network.transport import SimulatedTransport


# ---------------------------------------------------------------------------
# Discovery events
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryEvent:
    """An event in the discovery process."""

    event_type: str  # "peer_joined", "peer_left", "peer_degraded", "peer_recovered"
    peer_id: str
    timestamp: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MeshDiscovery
# ---------------------------------------------------------------------------


class MeshDiscovery:
    """Manages peer discovery and topology for a mesh network.

    Coordinates heartbeat sending/receiving, peer state tracking,
    and link quality monitoring through a NetworkNode and SimulatedTransport.
    """

    def __init__(
        self,
        node: NetworkNode,
        transport: SimulatedTransport,
        auto_discover: bool = True,
        heartbeat_timeout_s: float = 6.0,
    ):
        self._node = node
        self._transport = transport
        self._auto_discover = auto_discover
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._event_log: deque[DiscoveryEvent] = deque(maxlen=1000)
        self._known_nodes: set[str] = set()

    @property
    def node(self) -> NetworkNode:
        return self._node

    @property
    def transport(self) -> SimulatedTransport:
        return self._transport

    @property
    def known_nodes(self) -> set[str]:
        """All node IDs we've ever seen."""
        return set(self._known_nodes)

    @property
    def active_peers(self) -> list[str]:
        """Peer IDs currently in ACTIVE state."""
        return [
            pid for pid, p in self._node.peers.items()
            if p.state == NodeState.ACTIVE
        ]

    @property
    def event_log(self) -> list[DiscoveryEvent]:
        return list(self._event_log)

    def send_heartbeat(self, current_time: float) -> int:
        """Send heartbeat to all peers via broadcast.

        Returns number of peers the heartbeat was sent to.
        """
        if not self._node.should_send_heartbeat(current_time):
            return 0
        msg = self._node.make_heartbeat(current_time)
        raw = msg.serialize()
        return self._transport.broadcast_sync(raw)

    def process_incoming(self, current_time: float) -> list[DiscoveryEvent]:
        """Process all incoming messages and return discovery events.

        Handles heartbeats for peer discovery and health monitoring.
        Returns list of discovery events that occurred.
        """
        events: list[DiscoveryEvent] = []
        messages = self._transport.recv_all_sync()

        for source, raw in messages:
            try:
                msg = NetworkMessage.deserialize(raw)
            except Exception:
                continue

            if msg.msg_type == MessageType.HEARTBEAT:
                events.extend(self._handle_heartbeat(msg, current_time))

        # Check for timed-out peers
        offline = self._node.check_peer_health(current_time)
        for pid in offline:
            event = DiscoveryEvent(
                event_type="peer_left",
                peer_id=pid,
                timestamp=current_time,
                details={"reason": "heartbeat_timeout"},
            )
            events.append(event)
            self._event_log.append(event)

        return events

    def _handle_heartbeat(
        self, msg: NetworkMessage, current_time: float
    ) -> list[DiscoveryEvent]:
        """Handle a received heartbeat message."""
        events: list[DiscoveryEvent] = []
        peer_id = msg.source_node

        # New peer discovery
        is_new = peer_id not in self._known_nodes
        was_offline = False
        if not is_new:
            peer = self._node.get_peer(peer_id)
            was_offline = peer is not None and peer.state == NodeState.OFFLINE

        self._known_nodes.add(peer_id)
        peer_info = self._node.update_peer(msg)

        if is_new:
            event = DiscoveryEvent(
                event_type="peer_joined",
                peer_id=peer_id,
                timestamp=current_time,
                details={
                    "role": peer_info.state.value,
                    "capabilities": peer_info.capabilities,
                },
            )
            events.append(event)
            self._event_log.append(event)
        elif was_offline:
            event = DiscoveryEvent(
                event_type="peer_recovered",
                peer_id=peer_id,
                timestamp=current_time,
            )
            events.append(event)
            self._event_log.append(event)

        # Check if peer reported degraded state
        if peer_info.state == NodeState.DEGRADED:
            event = DiscoveryEvent(
                event_type="peer_degraded",
                peer_id=peer_id,
                timestamp=current_time,
            )
            events.append(event)
            self._event_log.append(event)

        return events

    def get_topology(self) -> dict[str, dict]:
        """Get current topology with peer details."""
        result: dict[str, dict] = {}
        for pid, peer in self._node.peers.items():
            link_stats = self._transport.get_link_stats(pid)
            result[pid] = {
                "state": peer.state.value,
                "capabilities": peer.capabilities,
                "track_count": peer.track_count,
                "link_loss_rate": link_stats.loss_rate,
                "link_avg_latency_ms": link_stats.avg_latency_ms,
            }
        return result

    def step(self, current_time: float) -> list[DiscoveryEvent]:
        """Run one discovery step: send heartbeat + process incoming.

        Convenience method combining send_heartbeat and process_incoming.
        """
        self.send_heartbeat(current_time)
        return self.process_incoming(current_time)
