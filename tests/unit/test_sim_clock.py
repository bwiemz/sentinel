"""Tests for SimClock deterministic clock, Clock protocol, and create_clock factory."""

from __future__ import annotations

import pytest

from sentinel.core.clock import Clock, SimClock, SystemClock, create_clock


# ---------------------------------------------------------------
# SimClock basics
# ---------------------------------------------------------------

class TestSimClockInitialState:
    def test_default_start_epoch(self):
        clock = SimClock()
        assert clock.start_epoch == 1_000_000.0

    def test_default_elapsed_zero(self):
        clock = SimClock()
        assert clock.elapsed() == 0.0

    def test_default_now_equals_start_epoch(self):
        clock = SimClock()
        assert clock.now() == 1_000_000.0

    def test_custom_start_epoch(self):
        clock = SimClock(start_epoch=5000.0)
        assert clock.now() == 5000.0
        assert clock.start_epoch == 5000.0
        assert clock.elapsed() == 0.0


class TestSimClockStep:
    def test_step_advances_time(self):
        clock = SimClock()
        clock.step(0.1)
        assert clock.elapsed() == pytest.approx(0.1)
        assert clock.now() == pytest.approx(1_000_000.1)

    def test_multiple_steps_accumulate(self):
        clock = SimClock()
        for _ in range(10):
            clock.step(0.5)
        assert clock.elapsed() == pytest.approx(5.0)
        assert clock.now() == pytest.approx(1_000_005.0)

    def test_step_zero_is_ok(self):
        clock = SimClock()
        clock.step(0.0)
        assert clock.elapsed() == 0.0

    def test_step_negative_raises(self):
        clock = SimClock()
        with pytest.raises(ValueError, match="dt >= 0"):
            clock.step(-0.1)

    def test_step_large_dt(self):
        clock = SimClock()
        clock.step(86400.0)  # one day
        assert clock.elapsed() == pytest.approx(86400.0)

    def test_step_small_dt(self):
        clock = SimClock()
        clock.step(0.001)
        assert clock.elapsed() == pytest.approx(0.001)


class TestSimClockSetTime:
    def test_set_time(self):
        clock = SimClock(start_epoch=1000.0)
        clock.set_time(1005.0)
        assert clock.elapsed() == pytest.approx(5.0)
        assert clock.now() == pytest.approx(1005.0)

    def test_set_time_at_start_epoch(self):
        clock = SimClock(start_epoch=1000.0)
        clock.step(5.0)
        clock.set_time(1000.0)
        assert clock.elapsed() == 0.0

    def test_set_time_before_start_raises(self):
        clock = SimClock(start_epoch=1000.0)
        with pytest.raises(ValueError, match="before start_epoch"):
            clock.set_time(999.0)


class TestSimClockSetElapsed:
    def test_set_elapsed(self):
        clock = SimClock()
        clock.set_elapsed(3.0)
        assert clock.elapsed() == pytest.approx(3.0)
        assert clock.now() == pytest.approx(1_000_003.0)

    def test_set_elapsed_zero(self):
        clock = SimClock()
        clock.step(5.0)
        clock.set_elapsed(0.0)
        assert clock.elapsed() == 0.0

    def test_set_elapsed_negative_raises(self):
        clock = SimClock()
        with pytest.raises(ValueError, match="elapsed >= 0"):
            clock.set_elapsed(-1.0)


class TestSimClockDeterminism:
    def test_two_clocks_same_sequence(self):
        c1 = SimClock(start_epoch=100.0)
        c2 = SimClock(start_epoch=100.0)
        for dt in [0.1, 0.2, 0.05, 1.0, 0.001]:
            c1.step(dt)
            c2.step(dt)
            assert c1.now() == c2.now()
            assert c1.elapsed() == c2.elapsed()

    def test_no_wall_clock_dependency(self):
        """SimClock values are identical regardless of real time."""
        clock = SimClock(start_epoch=42.0)
        # No matter when we read, if we haven't stepped, values are fixed
        val1 = clock.now()
        val2 = clock.now()
        assert val1 == val2 == 42.0


# ---------------------------------------------------------------
# Clock protocol
# ---------------------------------------------------------------

class TestClockProtocol:
    def test_system_clock_satisfies_protocol(self):
        assert isinstance(SystemClock(), Clock)

    def test_sim_clock_satisfies_protocol(self):
        assert isinstance(SimClock(), Clock)

    def test_both_have_now(self):
        for clock in [SystemClock(), SimClock()]:
            assert callable(getattr(clock, "now", None))

    def test_both_have_elapsed(self):
        for clock in [SystemClock(), SimClock()]:
            assert callable(getattr(clock, "elapsed", None))


# ---------------------------------------------------------------
# create_clock factory
# ---------------------------------------------------------------

class TestCreateClock:
    def test_default_returns_system_clock(self):
        clock = create_clock()
        assert isinstance(clock, SystemClock)

    def test_none_config_returns_system_clock(self):
        clock = create_clock(None)
        assert isinstance(clock, SystemClock)

    def test_realtime_mode(self):
        clock = create_clock({"mode": "realtime"})
        assert isinstance(clock, SystemClock)

    def test_simulated_mode(self):
        clock = create_clock({"mode": "simulated"})
        assert isinstance(clock, SimClock)

    def test_simulated_custom_epoch(self):
        clock = create_clock({"mode": "simulated", "start_epoch": 5000.0})
        assert isinstance(clock, SimClock)
        assert clock.now() == 5000.0

    def test_empty_config_returns_system_clock(self):
        clock = create_clock({})
        assert isinstance(clock, SystemClock)

    def test_unknown_mode_returns_system_clock(self):
        clock = create_clock({"mode": "unknown"})
        assert isinstance(clock, SystemClock)
