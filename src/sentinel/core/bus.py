"""Lightweight in-process pub/sub event bus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    """Simple synchronous event bus for decoupling pipeline stages."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[..., Any]) -> None:
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., Any]) -> None:
        if callback in self._subscribers[event]:
            self._subscribers[event].remove(callback)

    def publish(self, event: str, **kwargs: Any) -> None:
        for callback in self._subscribers.get(event, []):
            callback(**kwargs)
