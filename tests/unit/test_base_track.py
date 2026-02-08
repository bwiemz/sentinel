"""Tests for TrackBase state machine and M-of-N confirmation."""

import pytest

from sentinel.core.types import TrackState
from sentinel.tracking.base_track import TrackBase


class TestTrackBaseLifecycle:
    """Test the state machine transitions with configurable thresholds."""

    def test_initial_state(self):
        t = TrackBase()
        assert t.state == TrackState.TENTATIVE
        assert t.hits == 1
        assert t.misses == 0
        assert t.age == 0
        assert t.is_alive

    def test_default_confirmation(self):
        """Default: 3 consecutive hits to confirm."""
        t = TrackBase(confirm_hits=3)
        assert t.state == TrackState.TENTATIVE
        t._record_hit()  # hit 2
        assert t.state == TrackState.TENTATIVE
        t._record_hit()  # hit 3
        assert t.state == TrackState.CONFIRMED

    def test_custom_confirm_hits(self):
        t = TrackBase(confirm_hits=5)
        for _ in range(3):
            t._record_hit()
        assert t.state == TrackState.TENTATIVE
        t._record_hit()  # hit 5
        assert t.state == TrackState.CONFIRMED

    def test_tentative_delete_default(self):
        """Default: 3 consecutive misses deletes tentative track."""
        t = TrackBase()
        t._record_miss()
        assert t.state == TrackState.TENTATIVE
        t._record_miss()
        assert t.state == TrackState.TENTATIVE
        t._record_miss()
        assert t.state == TrackState.DELETED
        assert not t.is_alive

    def test_tentative_delete_custom(self):
        t = TrackBase(tentative_delete_misses=2)
        t._record_miss()
        assert t.state == TrackState.TENTATIVE
        t._record_miss()
        assert t.state == TrackState.DELETED

    def test_confirmed_to_coasting(self):
        """Default: 5 consecutive misses starts coasting a confirmed track."""
        t = TrackBase(confirm_hits=2)
        t._record_hit()  # confirm
        assert t.state == TrackState.CONFIRMED
        for i in range(4):
            t._record_miss()
            assert t.state == TrackState.CONFIRMED, f"Should not coast after {i+1} misses"
        t._record_miss()  # 5th miss
        assert t.state == TrackState.COASTING

    def test_confirmed_to_coasting_custom(self):
        t = TrackBase(confirm_hits=2, confirmed_coast_misses=3)
        t._record_hit()  # confirm
        for _ in range(2):
            t._record_miss()
        assert t.state == TrackState.CONFIRMED
        t._record_miss()  # 3rd miss
        assert t.state == TrackState.COASTING

    def test_coasting_reconfirm(self):
        """Default: 2 consecutive hits re-confirms a coasting track."""
        t = TrackBase(confirm_hits=2, confirmed_coast_misses=2)
        t._record_hit()  # confirm
        t._record_miss()
        t._record_miss()  # now coasting
        assert t.state == TrackState.COASTING
        t._record_hit()
        assert t.state == TrackState.COASTING
        t._record_hit()  # 2nd consecutive hit
        assert t.state == TrackState.CONFIRMED

    def test_coasting_reconfirm_custom(self):
        t = TrackBase(confirm_hits=2, confirmed_coast_misses=2, coast_reconfirm_hits=1)
        t._record_hit()  # confirm
        t._record_miss()
        t._record_miss()  # coasting
        assert t.state == TrackState.COASTING
        t._record_hit()  # 1 hit reconfirms
        assert t.state == TrackState.CONFIRMED

    def test_coasting_to_deleted(self):
        t = TrackBase(confirm_hits=2, confirmed_coast_misses=2, max_coast=5)
        t._record_hit()  # confirm
        t._record_miss()
        t._record_miss()  # 2 consecutive -> coasting, consecutive_misses=2
        assert t.state == TrackState.COASTING
        t._record_miss()  # 3 consecutive
        t._record_miss()  # 4 consecutive
        assert t.state == TrackState.COASTING
        t._record_miss()  # 5 consecutive >= max_coast -> DELETED
        assert t.state == TrackState.DELETED

    def test_miss_resets_consecutive_hits(self):
        t = TrackBase(confirm_hits=3)
        t._record_hit()  # 2 consecutive
        assert t.consecutive_hits == 2
        t._record_miss()  # resets
        assert t.consecutive_hits == 0
        t._record_hit()  # 1 consecutive
        assert t.consecutive_hits == 1
        assert t.state == TrackState.TENTATIVE  # needs 3 consecutive

    def test_hit_resets_consecutive_misses(self):
        t = TrackBase()
        t._record_miss()
        t._record_miss()
        assert t.consecutive_misses == 2
        t._record_hit()
        assert t.consecutive_misses == 0


class TestMofNConfirmation:
    """Test M-of-N sliding window confirmation logic."""

    def test_m_of_n_basic(self):
        """3 hits in 5 frames confirms."""
        t = TrackBase(confirm_hits=3, confirm_window=5)
        # Initial hit counted in window (1 hit)
        t._record_miss()  # 1 hit, 1 miss
        t._record_hit()   # 2 hits, 1 miss
        assert t.state == TrackState.TENTATIVE
        t._record_hit()   # 3 hits, 1 miss -> confirm
        assert t.state == TrackState.CONFIRMED

    def test_m_of_n_with_gaps(self):
        """Hits with misses between them still confirm with M-of-N."""
        t = TrackBase(confirm_hits=3, confirm_window=5)
        # Window: [True (init)]
        t._record_miss()   # [True, False]
        t._record_hit()    # [True, False, True]
        t._record_miss()   # [True, False, True, False]
        t._record_hit()    # [True, False, True, False, True] = 3 hits -> confirm
        assert t.state == TrackState.CONFIRMED

    def test_m_of_n_window_slides(self):
        """Old hits slide out of window, preventing confirmation."""
        t = TrackBase(confirm_hits=3, confirm_window=4)
        # Window: [True (init)]
        t._record_hit()    # [True, True]
        t._record_miss()   # [True, True, False]
        t._record_miss()   # [True, True, False, False] = 2 hits -> not confirmed
        assert t.state == TrackState.TENTATIVE
        t._record_miss()   # [True, False, False, False] = 1 hit (old True slid out)
        # Still tentative, but 3 consecutive misses -> DELETED
        assert t.state == TrackState.DELETED

    def test_m_of_n_would_not_confirm_without_window(self):
        """Without M-of-N, non-consecutive hits would not confirm."""
        t_no_window = TrackBase(confirm_hits=3)
        t_no_window._record_miss()
        t_no_window._record_hit()
        t_no_window._record_miss()
        t_no_window._record_hit()
        # consecutive_hits = 1, so no confirmation
        assert t_no_window.state == TrackState.TENTATIVE

    def test_consecutive_fallback_when_no_window(self):
        """Without confirm_window, uses consecutive hits."""
        t = TrackBase(confirm_hits=3, confirm_window=None)
        assert t._hit_window is None
        t._record_hit()  # 2 consecutive
        t._record_hit()  # 3 consecutive -> confirm
        assert t.state == TrackState.CONFIRMED


class TestTrackBaseScore:
    """Test score computation."""

    def test_initial_score(self):
        t = TrackBase()
        assert t.score == 0.5  # age == 0

    def test_score_increases_with_hits(self):
        t = TrackBase(confirm_hits=2)
        t.age = 1  # Force age > 0 so _update_score runs real calculation
        t._record_hit()  # confirm
        score_confirmed = t.score
        assert score_confirmed > 0.5  # confirmed bonus

    def test_score_decreases_with_misses(self):
        t = TrackBase(confirm_hits=2)
        t._record_hit()  # confirm (age still 0 -> score=0.5)
        t.age = 5  # Force age > 0
        t._update_score()
        score_high = t.score
        t._record_miss()
        t._record_miss()
        score_low = t.score
        assert score_low < score_high

    def test_score_capped_at_one(self):
        t = TrackBase(confirm_hits=2)
        for _ in range(100):
            t._record_hit()
        assert t.score <= 1.0

    def test_auto_generated_track_id(self):
        t1 = TrackBase()
        t2 = TrackBase()
        assert t1.track_id != t2.track_id
        assert len(t1.track_id) > 0

    def test_custom_track_id(self):
        t = TrackBase(track_id="MY-TRACK-001")
        assert t.track_id == "MY-TRACK-001"
