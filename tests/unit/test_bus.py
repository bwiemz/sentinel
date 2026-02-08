"""Tests for the event bus."""

from sentinel.core.bus import EventBus


class TestEventBus:
    def test_subscribe_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda **kw: received.append(kw))
        bus.publish("test", value=42)
        assert received == [{"value": 42}]

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe("evt", lambda **kw: results.append("a"))
        bus.subscribe("evt", lambda **kw: results.append("b"))
        bus.publish("evt")
        assert results == ["a", "b"]

    def test_no_crosstalk(self):
        bus = EventBus()
        results = []
        bus.subscribe("a", lambda **kw: results.append("a"))
        bus.subscribe("b", lambda **kw: results.append("b"))
        bus.publish("a")
        assert results == ["a"]

    def test_unsubscribe(self):
        bus = EventBus()
        results = []
        cb = lambda **kw: results.append(1)
        bus.subscribe("evt", cb)
        bus.publish("evt")
        bus.unsubscribe("evt", cb)
        bus.publish("evt")
        assert results == [1]

    def test_publish_no_subscribers(self):
        bus = EventBus()
        bus.publish("nonexistent")  # Should not raise
