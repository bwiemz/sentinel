"""Data link transport layer for Link 16 binary messages.

Provides a pluggable transport interface with an in-memory implementation
for testing and simulation.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class DataLinkTransport(Protocol):
    """Abstract transport for Link 16 binary messages."""

    def send(self, data: bytes, dest: str = "") -> bool: ...
    def recv(self) -> bytes | None: ...
    def recv_all(self) -> list[bytes]: ...


class InMemoryDataLinkTransport:
    """In-memory transport for testing without real hardware.

    Connected peers receive copies of all sent messages.
    """

    def __init__(self, name: str = "local") -> None:
        self.name = name
        self._inbox: deque[bytes] = deque()
        self._peers: list[InMemoryDataLinkTransport] = []
        self.messages_sent = 0
        self.messages_received = 0

    def connect_peer(self, peer: InMemoryDataLinkTransport) -> None:
        """Bidirectional peer connection."""
        if peer not in self._peers:
            self._peers.append(peer)
        if self not in peer._peers:
            peer._peers.append(self)

    def disconnect_peer(self, peer: InMemoryDataLinkTransport) -> None:
        """Remove bidirectional connection."""
        if peer in self._peers:
            self._peers.remove(peer)
        if self in peer._peers:
            peer._peers.remove(self)

    def send(self, data: bytes, dest: str = "") -> bool:
        """Push data to all connected peers' inboxes."""
        for peer in self._peers:
            peer._inbox.append(data)
            peer.messages_received += 1
        self.messages_sent += 1
        return len(self._peers) > 0

    def recv(self) -> bytes | None:
        """Pop oldest message from inbox, or None if empty."""
        if self._inbox:
            return self._inbox.popleft()
        return None

    def recv_all(self) -> list[bytes]:
        """Drain all pending messages."""
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs

    @property
    def pending_count(self) -> int:
        return len(self._inbox)
