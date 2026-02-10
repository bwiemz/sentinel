"""Unit tests for DDS-inspired pub/sub broker."""

import pytest

from sentinel.core.types import MessageType
from sentinel.network.messages import NetworkMessage
from sentinel.network.pubsub import (
    DEFAULT_QOS,
    PubSubBroker,
    QoSPolicy,
    TopicState,
)


def _msg(topic_hint: str = "tracks", ts: float = 1000.0, **payload_kw) -> NetworkMessage:
    """Helper to make a test message."""
    return NetworkMessage(
        msg_type=MessageType.TRACK_REPORT,
        source_node="TEST",
        timestamp=ts,
        payload=payload_kw,
    )


class TestQoSPolicy:
    def test_defaults(self):
        qos = QoSPolicy()
        assert qos.reliability == "best_effort"
        assert qos.priority == 0
        assert qos.history_depth == 1
        assert qos.lifespan_s == 10.0

    def test_custom(self):
        qos = QoSPolicy(reliability="reliable", priority=3, history_depth=10, lifespan_s=60.0)
        assert qos.reliability == "reliable"
        assert qos.priority == 3


class TestDefaultQoS:
    def test_tracks_reliable(self):
        assert DEFAULT_QOS["tracks"].reliability == "reliable"
        assert DEFAULT_QOS["tracks"].priority == 1

    def test_engagement_flash(self):
        assert DEFAULT_QOS["engagement"].priority == 3

    def test_all_topics_present(self):
        expected = {"tracks", "detections", "iff", "engagement", "heartbeat", "sensor_status"}
        assert set(DEFAULT_QOS.keys()) == expected


class TestTopicState:
    def test_add_to_history(self):
        ts = TopicState(qos=QoSPolicy(history_depth=3))
        for i in range(5):
            ts.add_to_history(_msg(ts=float(i)))
        assert len(ts.history) == 3  # capped at depth
        assert ts.message_count == 5

    def test_prune_expired(self):
        ts = TopicState(qos=QoSPolicy(lifespan_s=5.0, history_depth=10))
        for i in range(5):
            ts.add_to_history(_msg(ts=float(i)))
        removed = ts.prune_expired(current_time=10.0)
        # Messages at t=0,1,2,3,4; lifespan=5; current=10
        # All with age > 5 should be removed
        assert removed == 5
        assert len(ts.history) == 0


class TestPubSubBroker:
    def test_publish_no_subscribers(self):
        broker = PubSubBroker()
        msg = _msg()
        delivered = broker.publish("tracks", msg)
        assert delivered == 0

    def test_publish_with_subscriber(self):
        broker = PubSubBroker()
        received = []
        broker.subscribe("tracks", lambda m: received.append(m))
        msg = _msg()
        delivered = broker.publish("tracks", msg)
        assert delivered == 1
        assert len(received) == 1
        assert received[0] is msg

    def test_multiple_subscribers(self):
        broker = PubSubBroker()
        r1, r2 = [], []
        broker.subscribe("tracks", lambda m: r1.append(m))
        broker.subscribe("tracks", lambda m: r2.append(m))
        broker.publish("tracks", _msg())
        assert len(r1) == 1
        assert len(r2) == 1

    def test_topic_isolation(self):
        broker = PubSubBroker()
        received = []
        broker.subscribe("tracks", lambda m: received.append(m))
        broker.publish("iff", _msg())  # Different topic
        assert len(received) == 0

    def test_unsubscribe(self):
        broker = PubSubBroker()
        received = []
        sub_id = broker.subscribe("tracks", lambda m: received.append(m))
        broker.publish("tracks", _msg())
        assert len(received) == 1
        broker.unsubscribe(sub_id)
        broker.publish("tracks", _msg())
        assert len(received) == 1  # No new delivery

    def test_unsubscribe_nonexistent(self):
        broker = PubSubBroker()
        assert broker.unsubscribe("nonexistent") is False

    def test_content_filter(self):
        broker = PubSubBroker()
        received = []
        broker.subscribe(
            "tracks",
            lambda m: received.append(m),
            content_filter=lambda m: m.payload.get("threat_level") == "HIGH",
        )
        broker.publish("tracks", _msg(threat_level="LOW"))
        broker.publish("tracks", _msg(threat_level="HIGH"))
        broker.publish("tracks", _msg(threat_level="MEDIUM"))
        assert len(received) == 1
        assert received[0].payload["threat_level"] == "HIGH"

    def test_history(self):
        broker = PubSubBroker()
        broker.set_qos("tracks", QoSPolicy(history_depth=3))
        for i in range(5):
            broker.publish("tracks", _msg(ts=float(i), seq=i))
        history = broker.get_history("tracks")
        assert len(history) == 3
        # Should have the 3 most recent
        assert history[0].payload["seq"] == 2
        assert history[2].payload["seq"] == 4

    def test_history_empty_topic(self):
        broker = PubSubBroker()
        assert broker.get_history("nonexistent") == []

    def test_lifespan_expiry(self):
        broker = PubSubBroker()
        broker.set_qos("tracks", QoSPolicy(lifespan_s=5.0, history_depth=10))
        broker.publish("tracks", _msg(ts=0.0))
        broker.publish("tracks", _msg(ts=1.0))
        # Publish at t=10 triggers prune of old messages
        broker.publish("tracks", _msg(ts=10.0))
        history = broker.get_history("tracks")
        # Only the t=10 message should remain (t=0,1 are > 5s old)
        assert len(history) == 1
        assert history[0].timestamp == 10.0

    def test_topics_list(self):
        broker = PubSubBroker()
        broker.publish("tracks", _msg())
        broker.publish("iff", _msg())
        assert set(broker.topics) == {"tracks", "iff"}

    def test_subscriber_count(self):
        broker = PubSubBroker()
        assert broker.subscriber_count("tracks") == 0
        broker.subscribe("tracks", lambda m: None)
        broker.subscribe("tracks", lambda m: None)
        assert broker.subscriber_count("tracks") == 2

    def test_total_deliveries(self):
        broker = PubSubBroker()
        broker.subscribe("tracks", lambda m: None)
        broker.subscribe("tracks", lambda m: None)
        broker.publish("tracks", _msg())
        assert broker.total_deliveries == 2

    def test_message_count(self):
        broker = PubSubBroker()
        broker.publish("tracks", _msg())
        broker.publish("tracks", _msg())
        broker.publish("tracks", _msg())
        assert broker.topic_message_count("tracks") == 3
        assert broker.topic_message_count("nonexistent") == 0

    def test_set_qos_enforces_history(self):
        broker = PubSubBroker()
        broker.set_qos("tracks", QoSPolicy(history_depth=5))
        for i in range(5):
            broker.publish("tracks", _msg(ts=float(i)))
        assert len(broker.get_history("tracks")) == 5
        # Shrink history
        broker.set_qos("tracks", QoSPolicy(history_depth=2))
        assert len(broker.get_history("tracks")) == 2
