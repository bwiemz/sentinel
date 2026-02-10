"""Lightweight in-process pub/sub event bus."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous event bus for decoupling pipeline stages.

    Thread-safe: all subscribe/unsubscribe/publish operations are protected
    by a lock, and callbacks are invoked outside the lock on a snapshot of
    the subscriber list to prevent deadlocks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            if callback in self._subscribers[event]:
                self._subscribers[event].remove(callback)

    def publish(self, event: str, **kwargs: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event, []))
        for callback in callbacks:
            try:
                callback(**kwargs)
            except Exception:
                logger.exception("EventBus callback error on '%s'", event)
