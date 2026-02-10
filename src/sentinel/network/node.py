"""Network node identity and state management.

Each SENTINEL node has a unique identity, role, capabilities, and
participates in the mesh network with a state machine governing
its lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sentinel.core.types import MessageType, NodeRole, NodeState
from sentinel.network.messages import NetworkMessage, make_heartbeat
from sentinel.network.transport import SimulatedTransport


# ---------------------------------------------------------------------------
# Peer info
# ---------------------------------------------------------------------------


@dataclass
class PeerInfo:
    """Information about a connected peer node."""

    node_id: str
    role: NodeRole = NodeRole.SENSOR
    state: NodeState = NodeState.JOINING
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    track_count: int = 0
    link_quality: float = 1.0
    uptime_s: float = 0.0

    @property
    def is_stale(self) -> bool:
        """True if no heartbeat received recently (>6s at default rate)."""
        return False  # Evaluated externally with current time


# ---------------------------------------------------------------------------
# NetworkNode
# ---------------------------------------------------------------------------


class NetworkNode:
    """A node in the tactical mesh network.

    Manages identity, state transitions, heartbeat generation,
    and peer tracking.

    State machine:
        JOINING → ACTIVE → DEGRADED → OFFLINE
                   ↑          |
                   └──────────┘  (recovery)
    """

    def __init__(
        self,
        node_id: str,
        role: NodeRole = NodeRole.SENSOR,
        capabilities: list[str] | None = None,
        heartbeat_interval_s: float = 2.0,
        heartbeat_timeout_s: float = 6.0,
    ):
        self._node_id = node_id
        self._role = role
        self._capabilities = capabilities or []
        self._state = NodeState.JOINING
        self._heartbeat_interval_s = heartbeat_interval_s
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._peers: dict[str, PeerInfo] = {}
        self._start_time: float = 0.0
        self._last_heartbeat_sent: float = float("-inf")  # Always send first heartbeat
        self._track_count: int = 0

    # --- Properties ---

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def role(self) -> NodeRole:
        return self._role

    @property
    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def peers(self) -> dict[str, PeerInfo]:
        return dict(self._peers)

    @property
    def active_peer_count(self) -> int:
        """Number of peers in ACTIVE state."""
        return sum(
            1 for p in self._peers.values()
            if p.state == NodeState.ACTIVE
        )

    @property
    def uptime_s(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def is_healthy(self) -> bool:
        return self._state in (NodeState.ACTIVE, NodeState.JOINING)

    @property
    def track_count(self) -> int:
        return self._track_count

    @track_count.setter
    def track_count(self, value: int) -> None:
        self._track_count = value

    # --- State transitions ---

    def start(self, current_time: float | None = None) -> None:
        """Start the node — transitions from JOINING to ACTIVE."""
        self._start_time = time.monotonic()
        self._state = NodeState.ACTIVE

    def stop(self) -> None:
        """Stop the node — transitions to OFFLINE."""
        self._state = NodeState.OFFLINE

    def set_degraded(self) -> None:
        """Mark the node as degraded (e.g., sensor failure, partial connectivity)."""
        if self._state == NodeState.ACTIVE:
            self._state = NodeState.DEGRADED

    def recover(self) -> None:
        """Recover from degraded state back to active."""
        if self._state == NodeState.DEGRADED:
            self._state = NodeState.ACTIVE

    # --- Heartbeat ---

    def should_send_heartbeat(self, current_time: float) -> bool:
        """Check if it's time to send a heartbeat."""
        return (current_time - self._last_heartbeat_sent) >= self._heartbeat_interval_s

    def make_heartbeat(self, current_time: float) -> NetworkMessage:
        """Create a heartbeat message."""
        self._last_heartbeat_sent = current_time
        return make_heartbeat(
            source_node=self._node_id,
            timestamp=current_time,
            state=self._state.value,
            capabilities=self._capabilities,
            track_count=self._track_count,
            uptime_s=self.uptime_s,
        )

    # --- Peer management ---

    def update_peer(self, heartbeat: NetworkMessage) -> PeerInfo:
        """Update peer info from a received heartbeat message."""
        payload = heartbeat.payload
        peer_id = heartbeat.source_node
        if peer_id not in self._peers:
            self._peers[peer_id] = PeerInfo(node_id=peer_id)
        peer = self._peers[peer_id]
        try:
            peer.state = NodeState(payload.get("state", "active"))
        except (ValueError, KeyError):
            peer.state = NodeState.ACTIVE
        peer.capabilities = payload.get("capabilities", [])
        peer.last_heartbeat = heartbeat.timestamp
        peer.track_count = payload.get("track_count", 0)
        peer.uptime_s = payload.get("uptime_s", 0.0)
        return peer

    def check_peer_health(self, current_time: float) -> list[str]:
        """Check all peers for stale heartbeats.

        Returns list of peer IDs that became OFFLINE.
        """
        newly_offline = []
        for peer_id, peer in self._peers.items():
            if peer.state == NodeState.OFFLINE:
                continue
            age = current_time - peer.last_heartbeat
            if age > self._heartbeat_timeout_s:
                peer.state = NodeState.OFFLINE
                newly_offline.append(peer_id)
        return newly_offline

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from tracking."""
        self._peers.pop(peer_id, None)

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        """Get info for a specific peer."""
        return self._peers.get(peer_id)

    def to_dict(self) -> dict:
        """Serialize node state for monitoring/dashboard."""
        return {
            "node_id": self._node_id,
            "role": self._role.value,
            "state": self._state.value,
            "capabilities": self._capabilities,
            "track_count": self._track_count,
            "active_peers": self.active_peer_count,
            "total_peers": len(self._peers),
        }
