"""Unit tests for kinematic intent estimator."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.classification.intent_estimator import IntentEstimate, IntentEstimator
from sentinel.core.types import IntentType


def _mock_track(position=None, velocity=None, age=10):
    t = MagicMock()
    t.position = np.array(position) if position is not None else None
    t.velocity = np.array(velocity) if velocity is not None else None
    t.age = age
    return t


def _mock_eft(
    radar_track=None,
    thermal_track=None,
    quantum_radar_track=None,
    camera_track=None,
    position_m=None,
):
    eft = MagicMock()
    eft.radar_track = radar_track
    eft.thermal_track = thermal_track
    eft.quantum_radar_track = quantum_radar_track
    eft.camera_track = camera_track
    eft.position_m = np.array(position_m) if position_m is not None else None
    return eft


class TestIntentEstimator:
    def test_approaching_target(self):
        """Target closing at moderate speed → APPROACH."""
        est = IntentEstimator()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[-50, 0], age=10),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.APPROACH
        assert result.confidence > 0

    def test_fast_approaching_target(self):
        """High-speed target directly approaching → ATTACK."""
        est = IntentEstimator(attack_speed_threshold_mps=300.0)
        eft = _mock_eft(
            radar_track=_mock_track(position=[10000, 0], velocity=[-500, 0], age=10),
            position_m=[10000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.ATTACK

    def test_receding_target(self):
        """Target moving away → EVASION."""
        est = IntentEstimator()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[50, 0], age=10),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.EVASION

    def test_crossing_target(self):
        """Target moving perpendicular → TRANSIT."""
        est = IntentEstimator()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[0, 200], age=10),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.TRANSIT

    def test_slow_loitering_target(self):
        """Slow target with low approach rate → PATROL."""
        est = IntentEstimator(patrol_speed_threshold_mps=80.0)
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[0, 30], age=10),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.PATROL

    def test_young_track_unknown(self):
        """Track with age < min_track_age → UNKNOWN."""
        est = IntentEstimator(min_track_age=5)
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[-200, 0], age=2),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent == IntentType.UNKNOWN

    def test_cpa_computation(self):
        """Verify CPA fields are populated."""
        est = IntentEstimator()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 1000], velocity=[-100, 0], age=10),
            position_m=[5000, 1000],
        )
        result = est.estimate(eft)
        assert result.time_to_cpa_s is not None
        assert result.cpa_range_m is not None
        assert result.cpa_range_m >= 0

    def test_zero_velocity(self):
        """Stationary target → doesn't crash."""
        est = IntentEstimator()
        eft = _mock_eft(
            radar_track=_mock_track(position=[5000, 0], velocity=[0, 0], age=10),
            position_m=[5000, 0],
        )
        result = est.estimate(eft)
        assert result.intent in list(IntentType)

    def test_confidence_range(self):
        """Confidence is always in [0, 1]."""
        est = IntentEstimator()
        for vel in [[-500, 0], [500, 0], [0, 200], [0, 30], [-50, 0]]:
            eft = _mock_eft(
                radar_track=_mock_track(position=[5000, 0], velocity=vel, age=10),
                position_m=[5000, 0],
            )
            result = est.estimate(eft)
            assert 0 <= result.confidence <= 1.0

    def test_no_position_available(self):
        """Graceful handling when no position can be extracted."""
        est = IntentEstimator()
        eft = _mock_eft(camera_track=_mock_track(age=10))
        result = est.estimate(eft)
        assert result.intent == IntentType.UNKNOWN
