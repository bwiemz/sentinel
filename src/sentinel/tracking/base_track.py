"""Base track class with shared state machine and scoring logic.

Provides the lifecycle state machine (TENTATIVE -> CONFIRMED -> COASTING -> DELETED)
and scoring logic that is common to all track types (camera, radar, thermal).
"""

from __future__ import annotations

from collections import deque

from sentinel.core.types import TrackState, generate_track_id
from sentinel.tracking.track_quality import FilterConsistencyMonitor


class TrackBase:
    """Base class for all track types.

    Provides:
    - State machine with configurable transition thresholds
    - M-of-N confirmation logic (sliding window)
    - Quality score computation
    - Hit/miss counting

    Args:
        track_id: Unique identifier (auto-generated if None).
        confirm_hits: Hits needed for confirmation (M in M-of-N).
        max_coast: Max frames to coast before deletion.
        confirm_window: Sliding window size for M-of-N (None = use consecutive).
        tentative_delete_misses: Consecutive misses to delete a tentative track.
        confirmed_coast_misses: Consecutive misses to start coasting a confirmed track.
        coast_reconfirm_hits: Consecutive hits to re-confirm a coasting track.
    """

    def __init__(
        self,
        track_id: str | None = None,
        confirm_hits: int = 3,
        max_coast: int = 15,
        confirm_window: int | None = None,
        tentative_delete_misses: int = 3,
        confirmed_coast_misses: int = 5,
        coast_reconfirm_hits: int = 2,
        measurement_dim: int | None = None,
    ):
        self.track_id = track_id or generate_track_id()
        self.state = TrackState.TENTATIVE

        # Lifecycle counters
        self.hits = 1
        self.consecutive_hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.age = 0
        self.score = 0.0

        # Temporal alignment support
        self.last_update_time: float = 0.0

        # Filter consistency monitoring (NIS-based)
        self.quality_monitor: FilterConsistencyMonitor | None = None
        if measurement_dim is not None:
            self.quality_monitor = FilterConsistencyMonitor(dim_meas=measurement_dim)

        # Configurable thresholds
        self._confirm_hits = confirm_hits
        self._max_coast = max_coast
        self._tent_delete = tentative_delete_misses
        self._conf_coast = confirmed_coast_misses
        self._coast_reconfirm = coast_reconfirm_hits

        # M-of-N sliding window
        self._confirm_window: int | None = confirm_window
        self._hit_window: deque | None = None
        if confirm_window is not None and confirm_window > 0:
            self._hit_window = deque(maxlen=confirm_window)
            self._hit_window.append(True)  # Initial detection counts as a hit

        self._update_score()

    def _record_hit(self) -> None:
        """Record a detection hit."""
        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        if self._hit_window is not None:
            self._hit_window.append(True)
        self._update_state()
        self._update_score()

    def _record_miss(self) -> None:
        """Record a missed detection."""
        self.misses += 1
        self.consecutive_misses += 1
        self.consecutive_hits = 0
        if self._hit_window is not None:
            self._hit_window.append(False)
        self._update_state()
        self._update_score()

    def _update_state(self) -> None:
        """State machine transitions with configurable thresholds."""
        if self.state == TrackState.TENTATIVE:
            if self._check_confirmation():
                self.state = TrackState.CONFIRMED
            elif self.consecutive_misses >= self._tent_delete:
                self.state = TrackState.DELETED
        elif self.state == TrackState.CONFIRMED:
            if self.consecutive_misses >= self._conf_coast:
                self.state = TrackState.COASTING
        elif self.state == TrackState.COASTING:
            if self.consecutive_hits >= self._coast_reconfirm:
                self.state = TrackState.CONFIRMED
            elif self.consecutive_misses >= self._max_coast:
                self.state = TrackState.DELETED

    def _check_confirmation(self) -> bool:
        """Check if track should be confirmed (supports M-of-N)."""
        if self._hit_window is not None and len(self._hit_window) > 0:
            # M-of-N: at least confirm_hits within the window
            return sum(self._hit_window) >= self._confirm_hits
        # Fallback: consecutive hits
        return self.consecutive_hits >= self._confirm_hits

    def _update_score(self) -> None:
        """Compute track quality score in [0, 1].

        Incorporates NIS-based filter consistency when quality monitor
        is active. Weights: hit_ratio 0.3, recency 0.25, confirmation 0.25,
        filter consistency 0.2.
        """
        if self.age == 0:
            self.score = 0.5
            return
        hit_ratio = self.hits / max(self.age, 1)
        recency = max(0, 1.0 - self.consecutive_misses * 0.15)
        confirmation = 1.0 if self.state == TrackState.CONFIRMED else 0.5

        if self.quality_monitor is not None and self.quality_monitor.sample_count >= 3:
            quality_factor = self.quality_monitor.consistency_score
            self.score = min(
                1.0,
                hit_ratio * 0.3
                + recency * 0.25
                + confirmation * 0.25
                + quality_factor * 0.2,
            )
        else:
            self.score = min(1.0, hit_ratio * 0.4 + recency * 0.3 + confirmation * 0.3)

    @property
    def is_alive(self) -> bool:
        return self.state != TrackState.DELETED
