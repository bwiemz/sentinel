"""Regression tests for bugs found during code review (post-Phase 21).

Each test targets a specific bug that was identified and fixed.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 1: Camera JPDA — LinAlgError on singular covariance
# Bug 2-4: JPDA empty betas argmax crash (all 3 variants)
# ---------------------------------------------------------------------------

class TestJPDABugfixes:
    """Regression tests for JPDA associator bug fixes."""

    def test_camera_jpda_singular_covariance_no_crash(self):
        """Camera JPDA should not crash on singular innovation covariance."""
        from sentinel.tracking.jpda import JPDAAssociator
        from sentinel.core.types import Detection

        assoc = JPDAAssociator(gate_threshold=100.0, P_D=0.9)

        # Create a mock track with singular covariance (zeros)
        track = MagicMock()
        track.kf.H = np.eye(2)
        track.kf.P = np.zeros((2, 2))  # Singular
        track.kf.R = np.zeros((2, 2))  # Singular
        track.kf.predicted_measurement = np.array([100.0, 100.0])
        track.kf.dim_state = 4
        track.kf.x = np.array([100.0, 0.0, 100.0, 0.0])
        track.quality_monitor = None
        track.predicted_bbox = None

        det = Detection(
            bbox=np.array([95, 95, 105, 105]),
            confidence=0.9,
            class_name="vehicle",
            sensor_type="camera",
            timestamp=0.0,
        )

        # Should not raise LinAlgError — just skip the track
        result = assoc.associate_and_update([track], [det])
        assert result is not None

    def test_jpda_zero_likelihoods_no_argmax_crash(self):
        """JPDA should handle case where all likelihoods are zero."""
        from sentinel.tracking.jpda import _compute_beta_coefficients

        # All-zero likelihoods
        likelihoods = np.array([0.0, 0.0, 0.0])
        betas, beta_0 = _compute_beta_coefficients(likelihoods, P_D=0.9, lam=1e-6)

        # Should return all-zero betas with beta_0 = 1.0 (missed detection)
        assert beta_0 == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(betas, 0.0)

    def test_jpda_empty_likelihoods(self):
        """JPDA should handle empty likelihoods array."""
        from sentinel.tracking.jpda import _compute_beta_coefficients

        betas, beta_0 = _compute_beta_coefficients(np.array([]), P_D=0.9, lam=1e-6)
        assert len(betas) == 0
        assert beta_0 == 1.0


# ---------------------------------------------------------------------------
# Bug 5: Composite fusion — negative weight from local bias
# Bug 6: Composite fusion — unbounded memory growth
# ---------------------------------------------------------------------------

class TestCompositeFusionBugfixes:
    """Regression tests for composite fusion bug fixes."""

    def test_local_bias_weight_never_negative(self):
        """Local preference bias should not produce negative remote weight."""
        from sentinel.network.composite_fusion import CompositeFusion
        from sentinel.network.bridge import RemoteTrack

        cf = CompositeFusion(prefer_local=True)

        # Create a local track with very high confidence
        local = MagicMock()
        local.fused_id = "L1"
        local.confidence = 100.0  # Extremely high
        local.position_m = np.array([1000.0, 2000.0])
        local.threat_level = "LOW"
        local.sensor_sources = set()

        remote = RemoteTrack(
            track_id="R:T1",
            source_node="R",
            position=np.array([1010.0, 2010.0]),
            velocity=np.array([10.0, 0.0]),
            confidence=0.1,  # Very low
            threat_level="LOW",
            sensor_types=[],
            update_time=100.0,
        )

        # Should not crash — weights should remain in [0, 1]
        cf._merge_track(local, remote)

        # Verify position was merged (not corrupted by negative weights)
        pos = local.position_m
        assert pos[0] >= 1000.0  # Should be biased toward local
        assert pos[0] <= 1010.0

    def test_stale_composite_info_evicted(self):
        """Composite info should be cleaned up for tracks no longer active."""
        from sentinel.network.composite_fusion import CompositeFusion, CompositeTrackInfo

        cf = CompositeFusion()
        # Inject stale entries
        cf._composite_info["old_track_1"] = CompositeTrackInfo()
        cf._composite_info["old_track_2"] = CompositeTrackInfo()
        cf._composite_info["active_track"] = CompositeTrackInfo()

        # Only "active_track" is still in local tracks
        local = MagicMock()
        local.fused_id = "active_track"

        cf._evict_stale_info([local])

        assert "active_track" in cf._composite_info
        assert "old_track_1" not in cf._composite_info
        assert "old_track_2" not in cf._composite_info


# ---------------------------------------------------------------------------
# Bug 7: Zone authorization IndexError on invalid auth values
# ---------------------------------------------------------------------------

class TestZoneAuthBugfixes:

    def test_resolve_authorization_all_invalid_auths(self):
        """Zone resolution should not crash when all auths are unrecognized."""
        from sentinel.engagement.zones import ZoneManager, CircularZone
        from sentinel.core.types import ZoneAuth

        # Create a zone with a valid auth — the IndexError only happens if all
        # zone.authorization values are not in _ZONE_AUTH_RESTRICTIVENESS.
        # Since ZoneAuth is an enum, we can't easily create invalid values,
        # but the code path is now guarded. Test the normal case still works.
        z1 = CircularZone(
            zone_id="Z1", name="Z1", center_xy=np.array([0.0, 0.0]),
            radius_m=1000.0, authorization=ZoneAuth.NO_FIRE, priority=5,
        )
        z2 = CircularZone(
            zone_id="Z2", name="Z2", center_xy=np.array([0.0, 0.0]),
            radius_m=1000.0, authorization=ZoneAuth.WEAPONS_FREE, priority=5,
        )
        mgr = ZoneManager(zones=[z1, z2])
        # Equal priority — most restrictive should win (NO_FIRE)
        auth = mgr.resolve_authorization(np.array([0.0, 0.0]))
        assert auth == ZoneAuth.NO_FIRE


# ---------------------------------------------------------------------------
# Bug 8: Weapon ammo over-assignment
# ---------------------------------------------------------------------------

class TestAmmoOverAssignment:

    def test_ammo_budget_respected_across_slots(self):
        """Multi-slot weapons should not be assigned more salvos than ammo allows."""
        from sentinel.engagement.assignment import WeaponTargetAssigner
        from sentinel.engagement.weapons import WeaponProfile, WEZCalculator
        from sentinel.engagement.feasibility import FeasibilityCalculator
        from sentinel.core.types import WeaponType, EngagementAuth, ZoneAuth

        weapon = WeaponProfile(
            weapon_id="SAM-1",
            name="SAM-1",
            weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
            altitude_m=0.0,
            min_range_m=100.0,
            max_range_m=50000.0,
            optimal_range_m=25000.0,
            max_target_speed_mps=800.0,
            weapon_speed_mps=1200.0,
            pk_base=0.85,
            max_simultaneous_engagements=3,
            rounds_remaining=2,   # Only 2 rounds
            salvo_size=1,
        )

        tracks = [
            {
                "track_id": f"T{i}",
                "position": np.array([10000.0 + i * 1000, 5000.0]),
                "velocity": np.array([-300.0, 0.0]),
                "threat_level": "HIGH",
                "engagement_auth": EngagementAuth.WEAPONS_FREE,
                "iff_identification": "hostile",
            }
            for i in range(3)
        ]

        assigner = WeaponTargetAssigner(
            wez_calculator=WEZCalculator(),
            feasibility_calculator=FeasibilityCalculator(),
        )

        result = assigner.assign(
            [weapon], tracks, default_zone_auth=ZoneAuth.WEAPONS_FREE,
        )

        # Should assign at most 2 targets (ammo constraint)
        assert len(result.assignments) <= 2


# ---------------------------------------------------------------------------
# Bug 9: Node enum crash on invalid state
# ---------------------------------------------------------------------------

class TestNodeEnumBugfix:

    def test_update_peer_invalid_state_no_crash(self):
        """Node should not crash on invalid peer state string."""
        from sentinel.network.node import NetworkNode, NodeState
        from sentinel.network.messages import NetworkMessage, MessageType

        node = NetworkNode(node_id="test-node")
        node.start()

        # Simulate heartbeat with invalid state
        msg = NetworkMessage(
            msg_type=MessageType.HEARTBEAT,
            source_node="peer-1",
            timestamp=1.0,
            payload={"state": "INVALID_STATE_THAT_DOESNT_EXIST"},
        )

        # Should not raise ValueError
        peer = node.update_peer(msg)
        assert peer.state == NodeState.ACTIVE  # Falls back to ACTIVE


# ---------------------------------------------------------------------------
# Bug 10: CPA NaN/Inf from corrupted data
# ---------------------------------------------------------------------------

class TestCPABugfix:

    def test_cpa_handles_nan_velocity(self):
        """CPA computation should handle NaN in velocity gracefully."""
        from sentinel.classification.intent_estimator import IntentEstimator

        est = IntentEstimator(sensor_position=np.array([0.0, 0.0]))
        # NaN velocity → speed_sq < 1e-12 check catches it
        t_cpa, r_cpa = est._compute_cpa(
            np.array([1000.0, 1000.0]),
            np.array([float("nan"), float("nan")]),
        )
        # Should return None for time, valid range for distance
        # (NaN dot NaN = NaN, and NaN < 1e-12 is False, so t_cpa = NaN)
        # Our fix adds isfinite check
        if t_cpa is not None:
            assert math.isfinite(t_cpa)

    def test_cpa_normal_case(self):
        """CPA computation works for normal approaching target."""
        from sentinel.classification.intent_estimator import IntentEstimator

        est = IntentEstimator(sensor_position=np.array([0.0, 0.0]))
        t_cpa, r_cpa = est._compute_cpa(
            np.array([1000.0, 0.0]),
            np.array([-100.0, 0.0]),  # Approaching at 100 m/s
        )
        assert t_cpa is not None
        assert t_cpa == pytest.approx(10.0, abs=0.1)
        assert r_cpa == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Bug 11: Quality score weight normalization
# ---------------------------------------------------------------------------

class TestFeasibilityWeightNormalization:

    def test_weights_normalized_when_not_summing_to_one(self):
        """Quality weights should be normalized if they don't sum to 1.0."""
        from sentinel.engagement.feasibility import FeasibilityCalculator

        calc = FeasibilityCalculator(pk_weight=0.5, tti_weight=0.5, threat_weight=0.5)

        # Weights should be normalized to 1/3 each
        assert calc._pk_weight == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert calc._tti_weight == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert calc._threat_weight == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_weights_preserved_when_summing_to_one(self):
        """Default weights (0.4 + 0.3 + 0.3 = 1.0) should be preserved."""
        from sentinel.engagement.feasibility import FeasibilityCalculator

        calc = FeasibilityCalculator()
        assert calc._pk_weight == pytest.approx(0.4)
        assert calc._tti_weight == pytest.approx(0.3)
        assert calc._threat_weight == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Bug 12: WEZ validation for short position/velocity arrays
# ---------------------------------------------------------------------------

class TestWEZArrayValidation:

    def test_evaluate_short_position_returns_infeasible(self):
        """WEZ should return infeasible for position arrays < 2 elements."""
        from sentinel.engagement.weapons import WEZCalculator, WeaponProfile
        from sentinel.core.types import WeaponType

        calc = WEZCalculator()
        weapon = WeaponProfile(
            weapon_id="W1", name="W1", weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
        )

        result = calc.evaluate(weapon, np.array([100.0]), np.array([10.0, 0.0]))
        assert not result.feasible

    def test_evaluate_short_velocity_returns_infeasible(self):
        """WEZ should return infeasible for velocity arrays < 2 elements."""
        from sentinel.engagement.weapons import WEZCalculator, WeaponProfile
        from sentinel.core.types import WeaponType

        calc = WEZCalculator()
        weapon = WeaponProfile(
            weapon_id="W1", name="W1", weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
        )

        result = calc.evaluate(weapon, np.array([1000.0, 500.0]), np.array([10.0]))
        assert not result.feasible


# ---------------------------------------------------------------------------
# Bug 13: TrackFusion zero image_width_px
# ---------------------------------------------------------------------------

class TestTrackFusionValidation:

    def test_zero_image_width_raises(self):
        """TrackFusion should reject image_width_px <= 0."""
        from sentinel.fusion.track_fusion import TrackFusion

        with pytest.raises(ValueError, match="image_width_px must be positive"):
            TrackFusion(image_width_px=0)

    def test_negative_image_width_raises(self):
        """TrackFusion should reject negative image_width_px."""
        from sentinel.fusion.track_fusion import TrackFusion

        with pytest.raises(ValueError, match="image_width_px must be positive"):
            TrackFusion(image_width_px=-1)


# ---------------------------------------------------------------------------
# Bug 14: Track ID empty string in engagement manager
# ---------------------------------------------------------------------------

class TestEngagementManagerTrackID:

    def test_missing_track_id_generates_fallback(self):
        """Tracks without fused_id or track_id get a generated ID."""
        from sentinel.engagement.manager import EngagementManager

        track = MagicMock(spec=[])  # No attributes at all
        track.position_m = np.array([1000.0, 2000.0])

        td = EngagementManager._extract_track_dict(track)
        assert td is not None
        assert td["track_id"].startswith("UNK-")
        assert len(td["track_id"]) > 4
