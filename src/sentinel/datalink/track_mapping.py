"""Bidirectional mapping between SENTINEL track IDs and Link 16 track numbers.

Link 16 track numbers are 13-bit (0-8191).  SENTINEL uses 8-char hex IDs.
TrackNumberAllocator manages the mapping with LRU eviction.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class TrackNumberAllocator:
    """Thread-safe bidirectional SENTINEL ID <-> L16 track number mapping."""

    MAX_TRACK_NUMBER = 8191

    def __init__(self, max_entries: int = 8192) -> None:
        self._max_entries = min(max_entries, self.MAX_TRACK_NUMBER + 1)
        self._sentinel_to_l16: OrderedDict[str, int] = OrderedDict()
        self._l16_to_sentinel: dict[int, str] = {}
        self._next_number = 0
        self._lock = threading.Lock()

    def get_or_allocate(self, sentinel_id: str) -> int:
        """Get existing track number or allocate a new one."""
        with self._lock:
            if sentinel_id in self._sentinel_to_l16:
                self._sentinel_to_l16.move_to_end(sentinel_id)
                return self._sentinel_to_l16[sentinel_id]

            # Evict oldest if at capacity
            if len(self._sentinel_to_l16) >= self._max_entries:
                oldest_id, oldest_num = self._sentinel_to_l16.popitem(last=False)
                del self._l16_to_sentinel[oldest_num]

            # Find next free number
            num = self._find_free_number()
            self._sentinel_to_l16[sentinel_id] = num
            self._l16_to_sentinel[num] = sentinel_id
            return num

    def get_sentinel_id(self, track_number: int) -> str | None:
        """Reverse lookup: L16 track number -> SENTINEL ID."""
        with self._lock:
            return self._l16_to_sentinel.get(track_number)

    def release(self, sentinel_id: str) -> None:
        """Release a track number (track dropped)."""
        with self._lock:
            num = self._sentinel_to_l16.pop(sentinel_id, None)
            if num is not None:
                self._l16_to_sentinel.pop(num, None)

    def clear(self) -> None:
        """Reset all mappings."""
        with self._lock:
            self._sentinel_to_l16.clear()
            self._l16_to_sentinel.clear()
            self._next_number = 0

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sentinel_to_l16)

    def _find_free_number(self) -> int:
        """Find next available track number (unlocked, caller holds lock)."""
        for _ in range(self.MAX_TRACK_NUMBER + 1):
            num = self._next_number
            self._next_number = (self._next_number + 1) % (self.MAX_TRACK_NUMBER + 1)
            if num not in self._l16_to_sentinel:
                return num
        raise RuntimeError("No free track numbers available")
