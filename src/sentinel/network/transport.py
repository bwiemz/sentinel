"""Tactical network transport layer — TTNT-inspired simulated mesh.

Provides a Transport protocol and SimulatedTransport with configurable
network degradation (latency, jitter, packet loss, bandwidth limiting,
packet reordering) for realistic testing without real sockets.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Transport Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Abstract transport interface for sending/receiving bytes."""

    @property
    def node_id(self) -> str: ...

    async def send(self, dest: str, data: bytes) -> bool:
        """Send data to a specific destination node. Returns True on success."""
        ...

    async def recv(self, timeout: float | None = None) -> tuple[str, bytes] | None:
        """Receive data. Returns (source_node_id, data) or None on timeout."""
        ...

    async def broadcast(self, data: bytes) -> int:
        """Broadcast data to all peers. Returns number of peers sent to."""
        ...

    def get_peers(self) -> list[str]:
        """Return list of connected peer node IDs."""
        ...


# ---------------------------------------------------------------------------
# Degradation profile
# ---------------------------------------------------------------------------


@dataclass
class DegradationProfile:
    """Configurable network degradation parameters."""

    latency_ms: float = 5.0
    jitter_ms: float = 2.0
    packet_loss_rate: float = 0.0
    bandwidth_bps: float = 10_000_000  # 10 Mbps
    reorder_probability: float = 0.0

    def effective_delay_s(self, rng: random.Random) -> float:
        """Compute delay in seconds with jitter."""
        jitter = rng.uniform(-self.jitter_ms, self.jitter_ms)
        return max(0.0, (self.latency_ms + jitter) / 1000.0)

    def should_drop(self, rng: random.Random) -> bool:
        """Determine if packet should be dropped."""
        return rng.random() < self.packet_loss_rate

    def transmission_delay_s(self, size_bytes: int) -> float:
        """Time to transmit given bytes at bandwidth limit."""
        if self.bandwidth_bps <= 0:
            return 0.0
        return (size_bytes * 8) / self.bandwidth_bps

    def should_reorder(self, rng: random.Random) -> bool:
        """Determine if packet should be delivered out of order."""
        return rng.random() < self.reorder_probability


# Named presets
DEGRADATION_PRESETS: dict[str, DegradationProfile] = {
    "ideal": DegradationProfile(latency_ms=5, jitter_ms=1, packet_loss_rate=0.0,
                                 bandwidth_bps=10e6, reorder_probability=0.0),
    "ttnt": DegradationProfile(latency_ms=10, jitter_ms=5, packet_loss_rate=0.01,
                                bandwidth_bps=10e6, reorder_probability=0.01),
    "degraded": DegradationProfile(latency_ms=50, jitter_ms=20, packet_loss_rate=0.05,
                                    bandwidth_bps=1e6, reorder_probability=0.05),
    "severe": DegradationProfile(latency_ms=200, jitter_ms=100, packet_loss_rate=0.15,
                                  bandwidth_bps=100_000, reorder_probability=0.1),
    "intermittent": DegradationProfile(latency_ms=20, jitter_ms=10, packet_loss_rate=0.30,
                                        bandwidth_bps=5e6, reorder_probability=0.05),
}


# ---------------------------------------------------------------------------
# Link statistics
# ---------------------------------------------------------------------------


@dataclass
class LinkStats:
    """Per-link transmission statistics."""

    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    messages_reordered: int = 0
    bytes_sent: int = 0
    total_latency_s: float = 0.0

    @property
    def loss_rate(self) -> float:
        total = self.messages_sent
        if total == 0:
            return 0.0
        return self.messages_dropped / total

    @property
    def avg_latency_ms(self) -> float:
        if self.messages_received == 0:
            return 0.0
        return (self.total_latency_s / self.messages_received) * 1000


# ---------------------------------------------------------------------------
# SimulatedTransport
# ---------------------------------------------------------------------------


class SimulatedTransport:
    """In-process transport with configurable degradation.

    Connects to a TransportHub for message routing between nodes.
    Uses asyncio queues internally but provides a synchronous API wrapper
    for use in SENTINEL's synchronous pipeline.
    """

    def __init__(
        self,
        node_id: str,
        hub: TransportHub,
        profile: DegradationProfile | None = None,
        seed: int = 42,
    ):
        self._node_id = node_id
        self._hub = hub
        self._profile = profile or DegradationProfile()
        self._rng = random.Random(seed)
        self._inbox: deque[tuple[str, bytes]] = deque()
        self._stats: dict[str, LinkStats] = {}  # per-peer stats

        # Register with hub
        hub.register(self)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def profile(self) -> DegradationProfile:
        return self._profile

    @profile.setter
    def profile(self, value: DegradationProfile) -> None:
        self._profile = value

    def get_peers(self) -> list[str]:
        """Return list of connected peer node IDs."""
        return [nid for nid in self._hub.nodes if nid != self._node_id]

    def get_link_stats(self, peer: str) -> LinkStats:
        """Get statistics for a specific peer link."""
        if peer not in self._stats:
            self._stats[peer] = LinkStats()
        return self._stats[peer]

    # --- Synchronous API (used by SENTINEL pipeline) ---

    def send_sync(self, dest: str, data: bytes) -> bool:
        """Synchronous send — applies degradation and enqueues at destination."""
        if dest not in self._hub.nodes:
            return False

        profile = self._hub.get_link_profile(self._node_id, dest) or self._profile
        stats = self.get_link_stats(dest)
        stats.messages_sent += 1
        stats.bytes_sent += len(data)

        # Packet loss
        if profile.should_drop(self._rng):
            stats.messages_dropped += 1
            return False

        # Compute delay
        delay = profile.effective_delay_s(self._rng)
        delay += profile.transmission_delay_s(len(data))
        stats.total_latency_s += delay

        # Reordering: insert at random position in dest inbox
        dest_transport = self._hub.nodes[dest]
        if profile.should_reorder(self._rng) and len(dest_transport._inbox) > 0:
            stats.messages_reordered += 1
            pos = self._rng.randint(0, len(dest_transport._inbox))
            # Convert deque to list, insert, convert back
            items = list(dest_transport._inbox)
            items.insert(pos, (self._node_id, data))
            dest_transport._inbox = deque(items)
        else:
            dest_transport._inbox.append((self._node_id, data))

        dest_transport.get_link_stats(self._node_id).messages_received += 1
        return True

    def recv_sync(self) -> tuple[str, bytes] | None:
        """Synchronous receive — returns oldest message or None."""
        if self._inbox:
            return self._inbox.popleft()
        return None

    def recv_all_sync(self) -> list[tuple[str, bytes]]:
        """Receive all pending messages."""
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs

    def broadcast_sync(self, data: bytes) -> int:
        """Synchronous broadcast to all peers."""
        count = 0
        for peer_id in self.get_peers():
            if self.send_sync(peer_id, data):
                count += 1
        return count

    @property
    def pending_count(self) -> int:
        """Number of messages waiting in inbox."""
        return len(self._inbox)

    # --- Async API (for future use / compatibility) ---

    async def send(self, dest: str, data: bytes) -> bool:
        return self.send_sync(dest, data)

    async def recv(self, timeout: float | None = None) -> tuple[str, bytes] | None:
        return self.recv_sync()

    async def broadcast(self, data: bytes) -> int:
        return self.broadcast_sync(data)


# ---------------------------------------------------------------------------
# TransportHub
# ---------------------------------------------------------------------------


class TransportHub:
    """Central message router connecting SimulatedTransport instances.

    Manages node registration and per-link degradation profiles.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, SimulatedTransport] = {}
        self._link_profiles: dict[tuple[str, str], DegradationProfile] = {}

    def register(self, transport: SimulatedTransport) -> None:
        """Register a transport instance."""
        self.nodes[transport.node_id] = transport

    def unregister(self, node_id: str) -> None:
        """Remove a node from the hub."""
        self.nodes.pop(node_id, None)
        # Clean up link profiles
        self._link_profiles = {
            k: v for k, v in self._link_profiles.items()
            if k[0] != node_id and k[1] != node_id
        }

    def set_link_profile(
        self, node_a: str, node_b: str, profile: DegradationProfile
    ) -> None:
        """Set degradation profile for a specific link (bidirectional)."""
        self._link_profiles[(node_a, node_b)] = profile
        self._link_profiles[(node_b, node_a)] = profile

    def get_link_profile(
        self, source: str, dest: str
    ) -> DegradationProfile | None:
        """Get the degradation profile for a specific link."""
        return self._link_profiles.get((source, dest))

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def get_topology(self) -> dict[str, list[str]]:
        """Get current network topology as adjacency list."""
        topology: dict[str, list[str]] = {}
        for nid in self.nodes:
            topology[nid] = [
                pid for pid in self.nodes if pid != nid
            ]
        return topology

    def reset_all_stats(self) -> None:
        """Reset statistics on all transports."""
        for transport in self.nodes.values():
            transport._stats.clear()
