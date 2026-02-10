"""DDS-inspired publish/subscribe broker with QoS policies.

Provides topic-based routing with configurable reliability, priority,
history depth, and message lifespan. Content filtering allows subscribers
to receive only messages matching specific predicates.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.network.messages import NetworkMessage


# ---------------------------------------------------------------------------
# QoS Policy
# ---------------------------------------------------------------------------


@dataclass
class QoSPolicy:
    """Quality of Service policy for a topic.

    reliability: "best_effort" (fire and forget) or "reliable" (retry on failure)
    priority: 0=routine, 1=priority, 2=immediate, 3=flash
    history_depth: Number of messages retained per topic (for late joiners)
    lifespan_s: Messages older than this are automatically discarded
    """

    reliability: str = "best_effort"
    priority: int = 0
    history_depth: int = 1
    lifespan_s: float = 10.0


# Default QoS policies per topic
DEFAULT_QOS: dict[str, QoSPolicy] = {
    "tracks": QoSPolicy(reliability="reliable", priority=1, history_depth=5, lifespan_s=10.0),
    "detections": QoSPolicy(reliability="best_effort", priority=0, history_depth=1, lifespan_s=2.0),
    "iff": QoSPolicy(reliability="reliable", priority=2, history_depth=3, lifespan_s=30.0),
    "engagement": QoSPolicy(reliability="reliable", priority=3, history_depth=10, lifespan_s=60.0),
    "heartbeat": QoSPolicy(reliability="best_effort", priority=0, history_depth=1, lifespan_s=6.0),
    "sensor_status": QoSPolicy(reliability="reliable", priority=1, history_depth=3, lifespan_s=30.0),
}


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


@dataclass
class Subscription:
    """A single subscription to a topic."""

    sub_id: str
    topic: str
    callback: Callable[[NetworkMessage], None]
    content_filter: Callable[[NetworkMessage], bool] | None = None


# ---------------------------------------------------------------------------
# Topic state
# ---------------------------------------------------------------------------


@dataclass
class TopicState:
    """Internal state for a topic."""

    qos: QoSPolicy
    history: deque = field(default_factory=deque)
    subscriptions: dict[str, Subscription] = field(default_factory=dict)
    message_count: int = 0

    def add_to_history(self, msg: NetworkMessage) -> None:
        """Add message to history, enforcing depth limit."""
        self.history.append(msg)
        while len(self.history) > self.qos.history_depth:
            self.history.popleft()
        self.message_count += 1

    def prune_expired(self, current_time: float) -> int:
        """Remove expired messages from history. Returns count removed."""
        removed = 0
        while self.history:
            oldest = self.history[0]
            age = current_time - oldest.timestamp
            if age > self.qos.lifespan_s:
                self.history.popleft()
                removed += 1
            else:
                break
        return removed


# ---------------------------------------------------------------------------
# PubSubBroker
# ---------------------------------------------------------------------------


class PubSubBroker:
    """DDS-inspired publish-subscribe broker.

    Routes messages to subscribers by topic, applies QoS policies,
    supports content filtering, and maintains per-topic message history.
    """

    def __init__(self, default_qos: dict[str, QoSPolicy] | None = None) -> None:
        self._topics: dict[str, TopicState] = {}
        self._default_qos = default_qos or DEFAULT_QOS
        self._delivery_count = 0
        self._drop_count = 0

    def _get_or_create_topic(self, topic: str) -> TopicState:
        """Get existing topic state or create with default QoS."""
        if topic not in self._topics:
            qos = self._default_qos.get(topic, QoSPolicy())
            self._topics[topic] = TopicState(qos=qos)
        return self._topics[topic]

    def set_qos(self, topic: str, qos: QoSPolicy) -> None:
        """Set QoS policy for a topic."""
        ts = self._get_or_create_topic(topic)
        ts.qos = qos
        # Re-enforce history depth
        while len(ts.history) > qos.history_depth:
            ts.history.popleft()

    def get_qos(self, topic: str) -> QoSPolicy:
        """Get QoS policy for a topic."""
        ts = self._get_or_create_topic(topic)
        return ts.qos

    def publish(self, topic: str, message: NetworkMessage) -> int:
        """Publish a message to a topic.

        Returns the number of subscribers that received the message.
        """
        ts = self._get_or_create_topic(topic)

        # Check message lifespan
        current_time = message.timestamp
        ts.prune_expired(current_time)

        # Add to history
        ts.add_to_history(message)

        # Deliver to subscribers
        delivered = 0
        for sub in ts.subscriptions.values():
            # Apply content filter
            if sub.content_filter and not sub.content_filter(message):
                continue
            sub.callback(message)
            delivered += 1

        self._delivery_count += delivered
        return delivered

    def subscribe(
        self,
        topic: str,
        callback: Callable[[NetworkMessage], None],
        content_filter: Callable[[NetworkMessage], bool] | None = None,
    ) -> str:
        """Subscribe to a topic.

        Returns a subscription ID that can be used to unsubscribe.
        Optionally accepts a content_filter predicate to filter messages.
        """
        ts = self._get_or_create_topic(topic)
        sub_id = uuid.uuid4().hex[:8]
        sub = Subscription(
            sub_id=sub_id,
            topic=topic,
            callback=callback,
            content_filter=content_filter,
        )
        ts.subscriptions[sub_id] = sub
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscription by ID. Returns True if found and removed."""
        for ts in self._topics.values():
            if sub_id in ts.subscriptions:
                del ts.subscriptions[sub_id]
                return True
        return False

    def get_history(self, topic: str) -> list[NetworkMessage]:
        """Get message history for a topic."""
        if topic not in self._topics:
            return []
        return list(self._topics[topic].history)

    @property
    def topics(self) -> list[str]:
        """List of active topics."""
        return list(self._topics.keys())

    @property
    def total_deliveries(self) -> int:
        return self._delivery_count

    def subscriber_count(self, topic: str) -> int:
        """Number of subscribers for a topic."""
        if topic not in self._topics:
            return 0
        return len(self._topics[topic].subscriptions)

    def topic_message_count(self, topic: str) -> int:
        """Total messages published to a topic."""
        if topic not in self._topics:
            return 0
        return self._topics[topic].message_count
