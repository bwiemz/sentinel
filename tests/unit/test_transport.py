"""Unit tests for network transport layer."""

import pytest

from sentinel.network.transport import (
    DegradationProfile,
    DEGRADATION_PRESETS,
    LinkStats,
    SimulatedTransport,
    TransportHub,
)
from sentinel.network.messages import NetworkMessage, make_heartbeat
from sentinel.core.types import MessageType


# ===================================================================
# DegradationProfile
# ===================================================================


class TestDegradationProfile:
    def test_default_values(self):
        p = DegradationProfile()
        assert p.latency_ms == 5.0
        assert p.jitter_ms == 2.0
        assert p.packet_loss_rate == 0.0
        assert p.bandwidth_bps == 10_000_000
        assert p.reorder_probability == 0.0

    def test_effective_delay(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(latency_ms=10, jitter_ms=5)
        delays = [p.effective_delay_s(rng) for _ in range(100)]
        assert all(d >= 0 for d in delays)
        # Average should be around 10ms = 0.01s
        avg = sum(delays) / len(delays)
        assert 0.005 < avg < 0.015

    def test_zero_jitter(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(latency_ms=10, jitter_ms=0)
        delay = p.effective_delay_s(rng)
        assert delay == pytest.approx(0.01)

    def test_should_drop_zero_rate(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(packet_loss_rate=0.0)
        assert not any(p.should_drop(rng) for _ in range(100))

    def test_should_drop_full_rate(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(packet_loss_rate=1.0)
        assert all(p.should_drop(rng) for _ in range(100))

    def test_should_drop_partial_rate(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(packet_loss_rate=0.5)
        drops = sum(1 for _ in range(1000) if p.should_drop(rng))
        assert 400 < drops < 600

    def test_transmission_delay(self):
        p = DegradationProfile(bandwidth_bps=1_000_000)  # 1 Mbps
        # 1000 bytes = 8000 bits, at 1 Mbps = 8ms
        delay = p.transmission_delay_s(1000)
        assert delay == pytest.approx(0.008)

    def test_transmission_delay_zero_bandwidth(self):
        p = DegradationProfile(bandwidth_bps=0)
        assert p.transmission_delay_s(1000) == 0.0

    def test_should_reorder(self):
        import random
        rng = random.Random(42)
        p = DegradationProfile(reorder_probability=0.5)
        reorders = sum(1 for _ in range(1000) if p.should_reorder(rng))
        assert 400 < reorders < 600


class TestDegradationPresets:
    def test_all_presets_exist(self):
        expected = {"ideal", "ttnt", "degraded", "severe", "intermittent"}
        assert set(DEGRADATION_PRESETS.keys()) == expected

    def test_ideal_no_loss(self):
        p = DEGRADATION_PRESETS["ideal"]
        assert p.packet_loss_rate == 0.0
        assert p.reorder_probability == 0.0

    def test_severe_high_loss(self):
        p = DEGRADATION_PRESETS["severe"]
        assert p.packet_loss_rate > 0.1
        assert p.latency_ms >= 200

    def test_presets_are_degradation_profiles(self):
        for name, preset in DEGRADATION_PRESETS.items():
            assert isinstance(preset, DegradationProfile), f"{name} is not a DegradationProfile"


# ===================================================================
# LinkStats
# ===================================================================


class TestLinkStats:
    def test_defaults(self):
        s = LinkStats()
        assert s.messages_sent == 0
        assert s.loss_rate == 0.0
        assert s.avg_latency_ms == 0.0

    def test_loss_rate(self):
        s = LinkStats(messages_sent=100, messages_dropped=15)
        assert s.loss_rate == pytest.approx(0.15)

    def test_avg_latency(self):
        s = LinkStats(messages_received=10, total_latency_s=0.5)
        assert s.avg_latency_ms == pytest.approx(50.0)


# ===================================================================
# TransportHub
# ===================================================================


class TestTransportHub:
    def test_register(self):
        hub = TransportHub()
        t = SimulatedTransport("A", hub)
        assert "A" in hub.nodes
        assert hub.node_count == 1

    def test_register_multiple(self):
        hub = TransportHub()
        SimulatedTransport("A", hub)
        SimulatedTransport("B", hub)
        SimulatedTransport("C", hub)
        assert hub.node_count == 3
        assert set(hub.node_ids) == {"A", "B", "C"}

    def test_unregister(self):
        hub = TransportHub()
        SimulatedTransport("A", hub)
        SimulatedTransport("B", hub)
        hub.unregister("A")
        assert hub.node_count == 1
        assert "A" not in hub.nodes

    def test_topology(self):
        hub = TransportHub()
        SimulatedTransport("A", hub)
        SimulatedTransport("B", hub)
        SimulatedTransport("C", hub)
        topo = hub.get_topology()
        assert set(topo["A"]) == {"B", "C"}
        assert set(topo["B"]) == {"A", "C"}

    def test_set_link_profile(self):
        hub = TransportHub()
        profile = DegradationProfile(latency_ms=100)
        hub.set_link_profile("A", "B", profile)
        assert hub.get_link_profile("A", "B") is profile
        assert hub.get_link_profile("B", "A") is profile  # bidirectional

    def test_get_link_profile_default_none(self):
        hub = TransportHub()
        assert hub.get_link_profile("A", "B") is None

    def test_unregister_cleans_link_profiles(self):
        hub = TransportHub()
        hub.set_link_profile("A", "B", DegradationProfile())
        hub.unregister("A")
        assert hub.get_link_profile("A", "B") is None


# ===================================================================
# SimulatedTransport
# ===================================================================


class TestSimulatedTransport:
    def _make_hub_with_nodes(self, *node_ids, **kwargs):
        hub = TransportHub()
        transports = {}
        for i, nid in enumerate(node_ids):
            transports[nid] = SimulatedTransport(
                nid, hub,
                profile=kwargs.get("profile", DegradationProfile(packet_loss_rate=0.0)),
                seed=42 + i,
            )
        return hub, transports

    def test_send_recv(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        data = b"hello from A"
        assert nodes["A"].send_sync("B", data) is True
        result = nodes["B"].recv_sync()
        assert result is not None
        source, payload = result
        assert source == "A"
        assert payload == data

    def test_send_to_nonexistent(self):
        hub, nodes = self._make_hub_with_nodes("A")
        assert nodes["A"].send_sync("NONEXISTENT", b"data") is False

    def test_recv_empty(self):
        hub, nodes = self._make_hub_with_nodes("A")
        assert nodes["A"].recv_sync() is None

    def test_broadcast(self):
        hub, nodes = self._make_hub_with_nodes("A", "B", "C")
        count = nodes["A"].broadcast_sync(b"broadcast msg")
        assert count == 2
        assert nodes["B"].recv_sync() is not None
        assert nodes["C"].recv_sync() is not None

    def test_get_peers(self):
        hub, nodes = self._make_hub_with_nodes("A", "B", "C")
        peers = nodes["A"].get_peers()
        assert set(peers) == {"B", "C"}

    def test_pending_count(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        assert nodes["B"].pending_count == 0
        nodes["A"].send_sync("B", b"msg1")
        nodes["A"].send_sync("B", b"msg2")
        assert nodes["B"].pending_count == 2

    def test_recv_all(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        nodes["A"].send_sync("B", b"msg1")
        nodes["A"].send_sync("B", b"msg2")
        msgs = nodes["B"].recv_all_sync()
        assert len(msgs) == 2
        assert nodes["B"].pending_count == 0

    def test_packet_loss(self):
        """With 100% loss, no messages should arrive."""
        hub = TransportHub()
        profile = DegradationProfile(packet_loss_rate=1.0)
        a = SimulatedTransport("A", hub, profile=profile, seed=42)
        b = SimulatedTransport("B", hub, seed=43)
        for _ in range(10):
            a.send_sync("B", b"data")
        assert b.pending_count == 0

    def test_partial_packet_loss(self):
        """With 50% loss, roughly half should arrive."""
        hub = TransportHub()
        profile = DegradationProfile(packet_loss_rate=0.5)
        a = SimulatedTransport("A", hub, profile=profile, seed=42)
        b = SimulatedTransport("B", hub, seed=43)
        for _ in range(1000):
            a.send_sync("B", b"data")
        received = b.pending_count
        assert 350 < received < 650

    def test_link_stats(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        nodes["A"].send_sync("B", b"msg1")
        nodes["A"].send_sync("B", b"msg2")
        stats = nodes["A"].get_link_stats("B")
        assert stats.messages_sent == 2
        assert stats.bytes_sent == len(b"msg1") + len(b"msg2")

    def test_per_link_degradation(self):
        """Per-link profile should override node-level profile."""
        hub = TransportHub()
        a = SimulatedTransport("A", hub, profile=DegradationProfile(packet_loss_rate=0.0))
        b = SimulatedTransport("B", hub)
        c = SimulatedTransport("C", hub)

        # Set A→B link to 100% loss
        hub.set_link_profile("A", "B", DegradationProfile(packet_loss_rate=1.0))

        for _ in range(10):
            a.send_sync("B", b"to-b")
            a.send_sync("C", b"to-c")

        assert b.pending_count == 0   # All dropped
        assert c.pending_count == 10  # All arrived

    def test_message_serialization_roundtrip_through_transport(self):
        """Full round-trip: create message → serialize → send → recv → deserialize."""
        hub, nodes = self._make_hub_with_nodes("A", "B")
        msg = make_heartbeat("A", 1000.0, state="active", capabilities=["radar"])
        raw = msg.serialize()
        nodes["A"].send_sync("B", raw)
        result = nodes["B"].recv_sync()
        assert result is not None
        _, data = result
        restored = NetworkMessage.deserialize(data)
        assert restored.msg_type == MessageType.HEARTBEAT
        assert restored.source_node == "A"
        assert restored.payload["state"] == "active"

    def test_profile_setter(self):
        hub = TransportHub()
        t = SimulatedTransport("A", hub)
        new_profile = DegradationProfile(latency_ms=100)
        t.profile = new_profile
        assert t.profile.latency_ms == 100

    def test_bidirectional_communication(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        nodes["A"].send_sync("B", b"hello B")
        nodes["B"].send_sync("A", b"hello A")
        msg_at_b = nodes["B"].recv_sync()
        msg_at_a = nodes["A"].recv_sync()
        assert msg_at_b[1] == b"hello B"
        assert msg_at_a[1] == b"hello A"

    def test_three_node_mesh(self):
        """All three nodes can communicate with each other."""
        hub, nodes = self._make_hub_with_nodes("A", "B", "C")
        nodes["A"].send_sync("B", b"A->B")
        nodes["A"].send_sync("C", b"A->C")
        nodes["B"].send_sync("C", b"B->C")
        assert nodes["B"].recv_sync()[1] == b"A->B"
        assert nodes["C"].pending_count == 2

    def test_reset_stats(self):
        hub, nodes = self._make_hub_with_nodes("A", "B")
        nodes["A"].send_sync("B", b"data")
        assert nodes["A"].get_link_stats("B").messages_sent == 1
        hub.reset_all_stats()
        assert nodes["A"].get_link_stats("B").messages_sent == 0
