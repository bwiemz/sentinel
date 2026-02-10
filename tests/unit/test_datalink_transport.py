"""Unit tests for DataLink transport layer."""

from __future__ import annotations

import pytest

from sentinel.datalink.transport import (
    DataLinkTransport,
    InMemoryDataLinkTransport,
)


class TestInMemoryTransport:
    def test_send_recv(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.send(b"hello")
        assert b.recv() == b"hello"

    def test_bidirectional(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.send(b"from_a")
        b.send(b"from_b")
        assert b.recv() == b"from_a"
        assert a.recv() == b"from_b"

    def test_recv_empty(self):
        t = InMemoryDataLinkTransport("T")
        assert t.recv() is None

    def test_recv_all(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.send(b"m1")
        a.send(b"m2")
        a.send(b"m3")
        msgs = b.recv_all()
        assert msgs == [b"m1", b"m2", b"m3"]
        assert b.recv() is None  # drained

    def test_multiple_peers(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        c = InMemoryDataLinkTransport("C")
        a.connect_peer(b)
        a.connect_peer(c)
        a.send(b"broadcast")
        assert b.recv() == b"broadcast"
        assert c.recv() == b"broadcast"

    def test_disconnect(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.disconnect_peer(b)
        a.send(b"data")
        assert b.recv() is None

    def test_pending_count(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.send(b"1")
        a.send(b"2")
        assert b.pending_count == 2
        b.recv()
        assert b.pending_count == 1

    def test_message_counters(self):
        a = InMemoryDataLinkTransport("A")
        b = InMemoryDataLinkTransport("B")
        a.connect_peer(b)
        a.send(b"x")
        a.send(b"y")
        assert a.messages_sent == 2
        assert b.messages_received == 2

    def test_protocol_compliance(self):
        t = InMemoryDataLinkTransport()
        assert isinstance(t, DataLinkTransport)

    def test_send_no_peers_returns_false(self):
        t = InMemoryDataLinkTransport()
        assert t.send(b"data") is False
