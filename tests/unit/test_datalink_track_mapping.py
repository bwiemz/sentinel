"""Unit tests for TrackNumberAllocator."""

from __future__ import annotations

import threading

import pytest

from sentinel.datalink.track_mapping import TrackNumberAllocator


class TestTrackNumberAllocator:
    def test_allocate_new(self):
        alloc = TrackNumberAllocator()
        num = alloc.get_or_allocate("abc12345")
        assert 0 <= num <= 8191

    def test_same_id_returns_same_number(self):
        alloc = TrackNumberAllocator()
        n1 = alloc.get_or_allocate("abc")
        n2 = alloc.get_or_allocate("abc")
        assert n1 == n2

    def test_different_ids_different_numbers(self):
        alloc = TrackNumberAllocator()
        n1 = alloc.get_or_allocate("aaa")
        n2 = alloc.get_or_allocate("bbb")
        assert n1 != n2

    def test_reverse_lookup(self):
        alloc = TrackNumberAllocator()
        num = alloc.get_or_allocate("test_id")
        assert alloc.get_sentinel_id(num) == "test_id"

    def test_reverse_lookup_unknown(self):
        alloc = TrackNumberAllocator()
        assert alloc.get_sentinel_id(9999) is None

    def test_release(self):
        alloc = TrackNumberAllocator()
        num = alloc.get_or_allocate("test")
        alloc.release("test")
        assert alloc.get_sentinel_id(num) is None
        assert alloc.active_count == 0

    def test_active_count(self):
        alloc = TrackNumberAllocator()
        alloc.get_or_allocate("a")
        alloc.get_or_allocate("b")
        alloc.get_or_allocate("c")
        assert alloc.active_count == 3

    def test_clear(self):
        alloc = TrackNumberAllocator()
        for i in range(10):
            alloc.get_or_allocate(f"t{i}")
        alloc.clear()
        assert alloc.active_count == 0

    def test_no_collisions_100_tracks(self):
        alloc = TrackNumberAllocator()
        numbers = set()
        for i in range(100):
            num = alloc.get_or_allocate(f"track_{i}")
            numbers.add(num)
        assert len(numbers) == 100

    def test_eviction_at_capacity(self):
        alloc = TrackNumberAllocator(max_entries=5)
        for i in range(5):
            alloc.get_or_allocate(f"t{i}")
        assert alloc.active_count == 5

        # Adding 6th should evict oldest (t0)
        alloc.get_or_allocate("t5")
        assert alloc.active_count == 5
        assert alloc.get_sentinel_id(alloc.get_or_allocate("t5")) == "t5"

    def test_thread_safety(self):
        alloc = TrackNumberAllocator()
        results = []

        def allocate_batch(offset):
            for i in range(50):
                num = alloc.get_or_allocate(f"thread_{offset}_{i}")
                results.append(num)

        threads = [threading.Thread(target=allocate_batch, args=(j,)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert alloc.active_count == 200
        assert len(set(results)) == 200  # all unique
