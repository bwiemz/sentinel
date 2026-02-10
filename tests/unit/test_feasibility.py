"""Tests for engagement feasibility calculator — Pk, TTI, quality, permissions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.weapons import WeaponProfile, WEZResult
from sentinel.engagement.feasibility import FeasibilityCalculator, EngagementFeasibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weapon(**overrides) -> WeaponProfile:
    """Build a WeaponProfile with sensible defaults, accepting overrides."""
    defaults = dict(
        weapon_id="W1",
        name="Test SAM",
        weapon_type=WeaponType.SAM_MEDIUM,
        position_xy=np.array([0.0, 0.0]),
        min_range_m=500.0,
        max_range_m=20000.0,
        optimal_range_m=10000.0,
        weapon_speed_mps=1200.0,
        pk_base=0.85,
        pk_range_falloff=2.0,
        pk_speed_penalty=0.3,
        pk_ecm_penalty=0.3,
        max_target_speed_mps=800.0,
        rounds_remaining=10,
        max_simultaneous_engagements=2,
        salvo_size=1,
        reload_time_s=5.0,
        min_aspect_angle_deg=0.0,
        max_aspect_angle_deg=180.0,
        altitude_m=0.0,
        min_altitude_m=100.0,
        max_altitude_m=20000.0,
    )
    defaults.update(overrides)
    return WeaponProfile(**defaults)


def _wez(
    feasible: bool = True,
    slant_range: float = 10000.0,
    speed: float = 200.0,
    aspect: float = 0.0,
    closing: float = 100.0,
    **kw,
) -> WEZResult:
    """Build a WEZResult with sensible defaults, accepting overrides."""
    defaults = dict(
        weapon_id="W1",
        track_id="T1",
        feasible=feasible,
        in_range=True,
        in_altitude=True,
        in_speed=True,
        in_aspect=True,
        slant_range_m=slant_range,
        target_speed_mps=speed,
        aspect_angle_deg=aspect,
        closing_speed_mps=closing,
    )
    defaults.update(kw)
    return WEZResult(**defaults)


# ---------------------------------------------------------------------------
# TestPkCalculation
# ---------------------------------------------------------------------------

class TestPkCalculation:
    """Tests for FeasibilityCalculator.compute_pk."""

    def test_pk_at_optimal_range_low_speed(self):
        """Pk at optimal range with low speed should approach pk_base."""
        calc = FeasibilityCalculator()
        w = _weapon()
        wez = _wez(slant_range=10000.0, speed=0.0)
        pk = calc.compute_pk(w, wez)
        # range_factor=1.0, speed_factor=1.0, aspect_factor=1.0, ecm=1.0
        assert pk == pytest.approx(0.85, abs=1e-6)

    def test_pk_infeasible_wez_returns_zero(self):
        """Infeasible WEZ must produce Pk = 0."""
        calc = FeasibilityCalculator()
        pk = calc.compute_pk(_weapon(), _wez(feasible=False))
        assert pk == 0.0

    def test_pk_no_rounds_returns_zero(self):
        """Weapon with no rounds remaining must produce Pk = 0."""
        calc = FeasibilityCalculator()
        pk = calc.compute_pk(_weapon(rounds_remaining=0), _wez())
        assert pk == 0.0

    def test_pk_at_max_range_returns_zero(self):
        """At exactly max_range the range factor is 0, so Pk = 0."""
        calc = FeasibilityCalculator()
        pk = calc.compute_pk(_weapon(), _wez(slant_range=20000.0))
        assert pk == pytest.approx(0.0, abs=1e-6)

    def test_pk_beyond_optimal_degrades_quadratically(self):
        """Pk between optimal and max_range uses quadratic falloff."""
        calc = FeasibilityCalculator()
        w = _weapon()
        # Halfway between optimal (10000) and max (20000) → ratio 0.5
        pk_mid = calc.compute_pk(w, _wez(slant_range=15000.0, speed=0.0))
        # range_factor = 1 - (0.5)^2 = 0.75, pk = 0.85 * 0.75
        assert pk_mid == pytest.approx(0.85 * 0.75, abs=1e-3)

    def test_pk_inside_min_range_degrades_linearly(self):
        """Inside min_range the range factor degrades linearly."""
        calc = FeasibilityCalculator()
        w = _weapon(min_range_m=500.0)
        # At 250 m — half of min_range → factor = 0.5
        pk = calc.compute_pk(w, _wez(slant_range=250.0, speed=0.0))
        assert pk == pytest.approx(0.85 * 0.5, abs=1e-3)

    def test_pk_speed_penalty(self):
        """High target speed reduces Pk via quadratic speed penalty."""
        calc = FeasibilityCalculator()
        w = _weapon(pk_speed_penalty=0.3, max_target_speed_mps=800.0)
        # speed_ratio = 800/800 = 1.0, speed_factor = 1 - 0.3*1 = 0.7
        pk = calc.compute_pk(w, _wez(slant_range=10000.0, speed=800.0))
        assert pk == pytest.approx(0.85 * 0.7, abs=1e-3)

    def test_pk_ecm_penalty_when_jammed(self):
        """Jamming reduces Pk by ecm_factor = 1 - pk_ecm_penalty."""
        calc = FeasibilityCalculator()
        w = _weapon(pk_ecm_penalty=0.3)
        pk = calc.compute_pk(w, _wez(slant_range=10000.0, speed=0.0), is_jammed=True)
        assert pk == pytest.approx(0.85 * 0.7, abs=1e-3)

    def test_pk_no_ecm_penalty_when_not_jammed(self):
        """Without jamming, ecm_factor should be 1.0."""
        calc = FeasibilityCalculator()
        pk_jammed = calc.compute_pk(_weapon(), _wez(speed=0.0), is_jammed=True)
        pk_clear = calc.compute_pk(_weapon(), _wez(speed=0.0), is_jammed=False)
        assert pk_clear > pk_jammed

    def test_pk_clamped_between_zero_and_one(self):
        """Pk must always be in [0, 1]."""
        calc = FeasibilityCalculator()
        # Even with absurdly high pk_base, clamp to 1.0
        w = _weapon(pk_base=5.0)
        pk = calc.compute_pk(w, _wez(slant_range=10000.0, speed=0.0))
        assert 0.0 <= pk <= 1.0


# ---------------------------------------------------------------------------
# TestTTICalculation
# ---------------------------------------------------------------------------

class TestTTICalculation:
    """Tests for FeasibilityCalculator.compute_tti."""

    def test_tti_head_on(self):
        """Head-on (closing > 0): TTI = range / (weapon_speed + closing)."""
        calc = FeasibilityCalculator()
        w = _weapon(weapon_speed_mps=1200.0)
        wez = _wez(slant_range=13000.0, closing=100.0)
        tti = calc.compute_tti(w, wez)
        expected = 13000.0 / (1200.0 + 100.0)
        assert tti == pytest.approx(expected, abs=1e-3)

    def test_tti_pursuit(self):
        """Pursuit (closing < 0): TTI = range / (weapon_speed - |closing|)."""
        calc = FeasibilityCalculator()
        w = _weapon(weapon_speed_mps=1200.0)
        wez = _wez(slant_range=10000.0, closing=-200.0)
        tti = calc.compute_tti(w, wez)
        expected = 10000.0 / (1200.0 - 200.0)
        assert tti == pytest.approx(expected, abs=1e-3)

    def test_tti_cant_catch_returns_inf(self):
        """If weapon can't catch target, TTI = inf."""
        calc = FeasibilityCalculator()
        w = _weapon(weapon_speed_mps=500.0)
        # Target receding faster than weapon can fly
        wez = _wez(slant_range=10000.0, closing=-600.0)
        tti = calc.compute_tti(w, wez)
        assert math.isinf(tti)

    def test_tti_infeasible_returns_inf(self):
        """Infeasible WEZ → TTI = inf."""
        calc = FeasibilityCalculator()
        tti = calc.compute_tti(_weapon(), _wez(feasible=False))
        assert math.isinf(tti)

    def test_tti_zero_range_feasible_returns_zero(self):
        """Feasible target at zero range → TTI = 0."""
        calc = FeasibilityCalculator()
        tti = calc.compute_tti(_weapon(), _wez(slant_range=0.5, closing=0.0))
        assert tti == pytest.approx(0.0, abs=1e-6)

    def test_tti_effective_speed_zero_returns_inf(self):
        """Exact speed match in pursuit → effective speed 0 → inf."""
        calc = FeasibilityCalculator()
        w = _weapon(weapon_speed_mps=500.0)
        wez = _wez(slant_range=10000.0, closing=-500.0)
        tti = calc.compute_tti(w, wez)
        assert math.isinf(tti)


# ---------------------------------------------------------------------------
# TestQualityScore
# ---------------------------------------------------------------------------

class TestQualityScore:
    """Tests for FeasibilityCalculator.compute_quality_score."""

    def test_perfect_engagement_critical_threat(self):
        """High Pk, low TTI, CRITICAL threat → near maximum quality."""
        calc = FeasibilityCalculator(pk_weight=0.4, tti_weight=0.3, threat_weight=0.3)
        q = calc.compute_quality_score(
            pk=1.0, tti_s=0.0, threat_level="CRITICAL",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        # 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert q == pytest.approx(1.0, abs=1e-3)

    def test_no_fire_zone_returns_zero(self):
        """NO_FIRE zone always produces quality 0."""
        calc = FeasibilityCalculator()
        q = calc.compute_quality_score(
            pk=0.9, tti_s=5.0, threat_level="CRITICAL",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.NO_FIRE,
        )
        assert q == 0.0

    def test_hold_fire_roe_returns_zero(self):
        """HOLD_FIRE ROE always produces quality 0."""
        calc = FeasibilityCalculator()
        q = calc.compute_quality_score(
            pk=0.9, tti_s=5.0, threat_level="HIGH",
            roe_auth=EngagementAuth.HOLD_FIRE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        assert q == 0.0

    def test_restricted_fire_zone_reduces_quality(self):
        """RESTRICTED_FIRE zone multiplies quality by 0.3."""
        calc = FeasibilityCalculator()
        q_free = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        q_restricted = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.RESTRICTED_FIRE,
        )
        assert q_restricted == pytest.approx(q_free * 0.3, abs=1e-3)

    def test_self_defense_only_zone_reduces_quality(self):
        """SELF_DEFENSE_ONLY zone multiplies quality by 0.5."""
        calc = FeasibilityCalculator()
        q_free = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        q_self = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.SELF_DEFENSE_ONLY,
        )
        assert q_self == pytest.approx(q_free * 0.5, abs=1e-3)

    def test_weapons_hold_roe_reduces_quality(self):
        """WEAPONS_HOLD ROE multiplies quality by 0.3."""
        calc = FeasibilityCalculator()
        q_free = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        q_hold = calc.compute_quality_score(
            pk=0.8, tti_s=10.0, threat_level="HIGH",
            roe_auth=EngagementAuth.WEAPONS_HOLD,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        assert q_hold == pytest.approx(q_free * 0.3, abs=1e-3)

    def test_threat_weight_mapping(self):
        """Different threat levels produce different quality scores."""
        calc = FeasibilityCalculator()
        kwargs = dict(
            pk=0.5, tti_s=60.0,
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        q_crit = calc.compute_quality_score(threat_level="CRITICAL", **kwargs)
        q_high = calc.compute_quality_score(threat_level="HIGH", **kwargs)
        q_med = calc.compute_quality_score(threat_level="MEDIUM", **kwargs)
        q_low = calc.compute_quality_score(threat_level="LOW", **kwargs)
        assert q_crit > q_high > q_med > q_low

    def test_tti_beyond_max_gives_zero_tti_factor(self):
        """TTI >= max_tti_s → tti_factor = 0, only Pk and threat contribute."""
        calc = FeasibilityCalculator(max_tti_s=120.0)
        q = calc.compute_quality_score(
            pk=0.5, tti_s=200.0, threat_level="MEDIUM",
            roe_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        # 0.4*0.5 + 0.3*0 + 0.3*0.5 = 0.35
        assert q == pytest.approx(0.35, abs=1e-3)


# ---------------------------------------------------------------------------
# TestEngagementPermitted
# ---------------------------------------------------------------------------

class TestEngagementPermitted:
    """Tests for FeasibilityCalculator.check_engagement_permitted."""

    def test_no_fire_zone_always_blocked(self):
        assert not FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.NO_FIRE, EngagementAuth.WEAPONS_FREE,
        )

    def test_hold_fire_roe_always_blocked(self):
        assert not FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.WEAPONS_FREE, EngagementAuth.HOLD_FIRE,
        )

    def test_weapons_free_zone_and_roe_permitted(self):
        assert FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.WEAPONS_FREE, EngagementAuth.WEAPONS_FREE,
        )

    def test_weapons_free_zone_with_weapons_tight_roe_permitted(self):
        assert FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.WEAPONS_FREE, EngagementAuth.WEAPONS_TIGHT,
        )

    def test_self_defense_zone_with_weapons_free_roe_permitted(self):
        assert FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.SELF_DEFENSE_ONLY, EngagementAuth.WEAPONS_FREE,
        )

    def test_self_defense_zone_with_weapons_hold_roe_blocked(self):
        assert not FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.SELF_DEFENSE_ONLY, EngagementAuth.WEAPONS_HOLD,
        )

    def test_restricted_fire_zone_with_weapons_free_roe_permitted(self):
        assert FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.RESTRICTED_FIRE, EngagementAuth.WEAPONS_FREE,
        )

    def test_restricted_fire_zone_with_weapons_tight_roe_blocked(self):
        assert not FeasibilityCalculator.check_engagement_permitted(
            ZoneAuth.RESTRICTED_FIRE, EngagementAuth.WEAPONS_TIGHT,
        )


# ---------------------------------------------------------------------------
# TestFeasibilityEvaluate
# ---------------------------------------------------------------------------

class TestFeasibilityEvaluate:
    """Tests for the top-level FeasibilityCalculator.evaluate method."""

    def test_evaluate_returns_engagement_feasibility(self):
        """evaluate() must return an EngagementFeasibility dataclass."""
        calc = FeasibilityCalculator()
        result = calc.evaluate(
            weapon=_weapon(),
            wez=_wez(),
            threat_level="HIGH",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        assert isinstance(result, EngagementFeasibility)

    def test_evaluate_ids_match(self):
        """weapon_id and track_id in result must match inputs."""
        calc = FeasibilityCalculator()
        result = calc.evaluate(
            weapon=_weapon(weapon_id="WPN-7"),
            wez=_wez(weapon_id="WPN-7", track_id="TRK-42"),
            threat_level="MEDIUM",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        assert result.weapon_id == "WPN-7"
        assert result.track_id == "TRK-42"

    def test_evaluate_permitted_when_free(self):
        """WEAPONS_FREE zone + ROE should set engagement_permitted = True."""
        calc = FeasibilityCalculator()
        result = calc.evaluate(
            weapon=_weapon(),
            wez=_wez(),
            threat_level="HIGH",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        assert result.engagement_permitted is True
        assert result.pk > 0
        assert result.quality_score > 0

    def test_evaluate_not_permitted_no_fire(self):
        """NO_FIRE zone should block engagement and zero quality."""
        calc = FeasibilityCalculator()
        result = calc.evaluate(
            weapon=_weapon(),
            wez=_wez(),
            threat_level="CRITICAL",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.NO_FIRE,
        )
        assert result.engagement_permitted is False
        assert result.quality_score == 0.0

    def test_evaluate_to_dict(self):
        """to_dict() must produce expected keys."""
        calc = FeasibilityCalculator()
        result = calc.evaluate(
            weapon=_weapon(),
            wez=_wez(),
            threat_level="HIGH",
            engagement_auth=EngagementAuth.WEAPONS_FREE,
            zone_auth=ZoneAuth.WEAPONS_FREE,
        )
        d = result.to_dict()
        for key in ("weapon_id", "track_id", "pk", "tti_s", "quality_score",
                     "zone_authorization", "roe_authorization",
                     "engagement_permitted", "wez"):
            assert key in d

    def test_evaluate_jammed_reduces_pk(self):
        """Jammed engagement should have lower Pk than clear engagement."""
        calc = FeasibilityCalculator()
        w = _weapon()
        wez = _wez(speed=0.0)
        result_clear = calc.evaluate(w, wez, "HIGH",
                                     EngagementAuth.WEAPONS_FREE,
                                     ZoneAuth.WEAPONS_FREE, is_jammed=False)
        result_jammed = calc.evaluate(w, wez, "HIGH",
                                      EngagementAuth.WEAPONS_FREE,
                                      ZoneAuth.WEAPONS_FREE, is_jammed=True)
        assert result_jammed.pk < result_clear.pk
