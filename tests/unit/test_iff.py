"""Unit tests for IFF interrogation simulation."""

import numpy as np
import pytest

from sentinel.core.types import IFFCode, IFFMode
from sentinel.sensors.iff import (
    IFFConfig,
    IFFInterrogator,
    IFFResponse,
    IFFResult,
    IFFTransponder,
    iff_altitude_check,
    iff_interrogation_probability,
    iff_spoof_detection_score,
)


# ===================================================================
# Physics functions
# ===================================================================


class TestIFFInterrogationProbability:
    def test_zero_range(self):
        assert iff_interrogation_probability(0, 400_000) == 0.0

    def test_negative_range(self):
        assert iff_interrogation_probability(-100, 400_000) == 0.0

    def test_at_max_range(self):
        assert iff_interrogation_probability(400_000, 400_000) == 0.0

    def test_beyond_max_range(self):
        assert iff_interrogation_probability(500_000, 400_000) == 0.0

    def test_close_range_high_probability(self):
        p = iff_interrogation_probability(1000, 400_000)
        assert p > 0.97

    def test_mid_range(self):
        p = iff_interrogation_probability(200_000, 400_000)
        # 0.98 * (1 - 0.5^2) = 0.98 * 0.75 = 0.735
        assert 0.70 < p < 0.80

    def test_near_max_range_low_probability(self):
        p = iff_interrogation_probability(380_000, 400_000)
        assert p < 0.10

    def test_custom_base_probability(self):
        p = iff_interrogation_probability(1000, 400_000, base_probability=0.50)
        assert 0.49 < p < 0.51

    def test_zero_max_range(self):
        assert iff_interrogation_probability(100, 0) == 0.0


class TestIFFSpoofDetectionScore:
    def test_all_valid(self):
        score = iff_spoof_detection_score(
            crypto_valid=True, code_consistent=True, kinematic_plausible=True
        )
        assert score == 0.0

    def test_crypto_invalid(self):
        score = iff_spoof_detection_score(
            crypto_valid=False, code_consistent=True, kinematic_plausible=True
        )
        assert score == pytest.approx(0.40)

    def test_code_inconsistent(self):
        score = iff_spoof_detection_score(
            crypto_valid=True, code_consistent=False, kinematic_plausible=True
        )
        assert score == pytest.approx(0.30)

    def test_kinematic_implausible(self):
        score = iff_spoof_detection_score(
            crypto_valid=True, code_consistent=True, kinematic_plausible=False
        )
        assert score == pytest.approx(0.30)

    def test_all_invalid(self):
        score = iff_spoof_detection_score(
            crypto_valid=False, code_consistent=False, kinematic_plausible=False
        )
        assert score == 1.0

    def test_crypto_and_code_invalid(self):
        score = iff_spoof_detection_score(
            crypto_valid=False, code_consistent=False, kinematic_plausible=True
        )
        assert score == pytest.approx(0.70)


class TestIFFAltitudeCheck:
    def test_no_mode_c(self):
        assert iff_altitude_check(None, 10000, 5.0) is True

    def test_no_elevation(self):
        assert iff_altitude_check(10000, 10000, None) is True

    def test_consistent_altitude(self):
        # range=10km, elev=5.73deg -> alt ~1000m ~3281ft
        assert iff_altitude_check(3300, 10000, 5.73) is True

    def test_inconsistent_altitude(self):
        # range=10km, elev=5.73deg -> alt ~1000m ~3281ft, reported 20000ft
        assert iff_altitude_check(20000, 10000, 5.73) is False

    def test_zero_elevation(self):
        assert iff_altitude_check(0, 10000, 0.0) is True


# ===================================================================
# IFFTransponder
# ===================================================================


class TestIFFTransponder:
    def test_default_disabled(self):
        tp = IFFTransponder()
        assert tp.enabled is False
        assert tp.modes == []
        assert tp.mode_3a_code == "0000"

    def test_friendly_transponder(self):
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
        )
        assert tp.enabled is True
        assert IFFMode.MODE_4 in tp.modes
        assert tp.mode_4_valid is True
        assert tp.is_spoofed is False

    def test_spoofed_transponder(self):
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
            is_spoofed=True,
        )
        assert tp.is_spoofed is True

    def test_from_omegaconf(self):
        cfg = {
            "enabled": True,
            "modes": ["mode_3a", "mode_c", "mode_4"],
            "mode_3a_code": "4521",
            "mode_4_valid": True,
            "reliability": 0.95,
        }
        tp = IFFTransponder.from_omegaconf(cfg)
        assert tp.enabled is True
        assert len(tp.modes) == 3
        assert tp.mode_3a_code == "4521"
        assert tp.mode_4_valid is True
        assert tp.reliability == 0.95

    def test_from_omegaconf_minimal(self):
        cfg = {"enabled": True}
        tp = IFFTransponder.from_omegaconf(cfg)
        assert tp.enabled is True
        assert tp.modes == []


# ===================================================================
# IFFConfig
# ===================================================================


class TestIFFConfig:
    def test_default_disabled(self):
        cfg = IFFConfig()
        assert cfg.enabled is False
        assert cfg.max_interrogation_range_m == 400_000.0

    def test_from_omegaconf(self):
        cfg_dict = {
            "enabled": True,
            "max_interrogation_range_m": 200_000,
            "modes": ["mode_3a", "mode_4", "mode_5"],
            "require_mode_4": True,
            "controlled_airspace": True,
            "friendly_codes": ["1200", "4521"],
        }
        cfg = IFFConfig.from_omegaconf(cfg_dict)
        assert cfg.enabled is True
        assert cfg.max_interrogation_range_m == 200_000
        assert len(cfg.modes) == 3
        assert cfg.require_mode_4 is True
        assert cfg.controlled_airspace is True
        assert cfg.friendly_codes == ["1200", "4521"]

    def test_from_omegaconf_defaults(self):
        cfg = IFFConfig.from_omegaconf({})
        assert cfg.enabled is False
        assert len(cfg.modes) == 2  # mode_3a, mode_c


# ===================================================================
# IFFInterrogator — single target interrogation
# ===================================================================


class TestIFFInterrogatorSingle:
    def _make_interrogator(self, **kwargs):
        cfg = IFFConfig(enabled=True, **kwargs)
        return IFFInterrogator(cfg, seed=42)

    def test_no_transponder_no_response(self):
        interrog = self._make_interrogator()
        resp = interrog.interrogate_target("TGT-1", None, 5000, 0.0)
        assert resp.responded is False

    def test_disabled_transponder_no_response(self):
        interrog = self._make_interrogator()
        tp = IFFTransponder(enabled=False)
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is False

    def test_friendly_transponder_responds(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
            friendly_codes=["1200"],
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_C, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is True
        assert resp.mode_3a_code == "1200"
        assert resp.mode_4_authenticated is True
        assert resp.identification == IFFCode.FRIENDLY
        assert resp.confidence > 0.85

    def test_crypto_mode_5_highest_confidence(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A, IFFMode.MODE_5],
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_5],
            mode_3a_code="1200",
            mode_5_valid=True,
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is True
        assert resp.mode_5_authenticated is True
        assert resp.confidence >= 0.95

    def test_unknown_code_low_confidence(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A],
            friendly_codes=["1200"],
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A],
            mode_3a_code="7777",
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is True
        assert resp.identification == IFFCode.UNKNOWN
        assert resp.confidence < 0.5

    def test_spoofed_transponder_detected(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            require_mode_4=True,
            spoof_detection_threshold=0.3,
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
            is_spoofed=True,
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is True
        assert resp.mode_4_authenticated is False  # Spoofed → crypto fails
        assert resp.is_spoof_suspect is True
        assert resp.identification == IFFCode.SPOOF_SUSPECT

    def test_out_of_range_no_response(self):
        interrog = self._make_interrogator(max_interrogation_range_m=10_000)
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A],
            mode_3a_code="1200",
        )
        resp = interrog.interrogate_target("TGT-1", tp, 15_000, 0.0)
        assert resp.responded is False

    def test_unreliable_transponder(self):
        interrog = self._make_interrogator()
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A],
            mode_3a_code="1200",
            reliability=0.0,  # Never responds
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is False

    def test_mode_mismatch_no_response(self):
        interrog = self._make_interrogator(modes=[IFFMode.MODE_4])
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A],  # Only supports 3A, not Mode 4
            mode_3a_code="1200",
        )
        resp = interrog.interrogate_target("TGT-1", tp, 5000, 0.0)
        assert resp.responded is False


# ===================================================================
# IFFInterrogator — track identification accumulation
# ===================================================================


class TestIFFTrackIdentification:
    def _make_interrogator(self, **kwargs):
        cfg = IFFConfig(enabled=True, **kwargs)
        return IFFInterrogator(cfg, seed=42)

    def test_friendly_identification_accumulates(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
        )
        for i in range(3):
            resp = interrog.interrogate_target("TGT-1", tp, 5000, float(i))
            result = interrog.update_track_identification("TGT-1", resp)

        assert result.identification == IFFCode.FRIENDLY
        assert result.confidence > 0.90
        assert result.consecutive_responses == 3

    def test_no_response_escalates_to_assumed_hostile(self):
        interrog = self._make_interrogator(no_response_hostile_threshold=3)
        for i in range(4):
            resp = interrog.interrogate_target("TGT-1", None, 5000, float(i))
            result = interrog.update_track_identification("TGT-1", resp)

        assert result.identification == IFFCode.ASSUMED_HOSTILE
        assert result.confidence > 0.50
        assert result.consecutive_no_responses >= 3

    def test_assumed_friendly_after_threshold(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A],
            friendly_response_threshold=2,
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A],
            mode_3a_code="7777",  # Unknown code (not in friendly_codes)
        )
        for i in range(3):
            resp = interrog.interrogate_target("TGT-1", tp, 5000, float(i))
            result = interrog.update_track_identification("TGT-1", resp)

        assert result.identification == IFFCode.ASSUMED_FRIENDLY
        assert result.consecutive_responses >= 2

    def test_spoof_indicators_accumulate(self):
        interrog = self._make_interrogator(
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            require_mode_4=True,
            spoof_detection_threshold=0.3,
        )
        tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            mode_4_valid=True,
            is_spoofed=True,
        )
        for i in range(3):
            resp = interrog.interrogate_target("TGT-1", tp, 5000, float(i))
            result = interrog.update_track_identification("TGT-1", resp)

        assert result.identification == IFFCode.SPOOF_SUSPECT
        assert result.spoof_indicators >= 3

    def test_pending_before_threshold(self):
        interrog = self._make_interrogator(no_response_hostile_threshold=5)
        resp = interrog.interrogate_target("TGT-1", None, 5000, 0.0)
        result = interrog.update_track_identification("TGT-1", resp)
        assert result.identification == IFFCode.PENDING


# ===================================================================
# IFFInterrogator — batch interrogation
# ===================================================================


class TestIFFInterrogateAll:
    def test_interrogate_all_mixed_targets(self):
        cfg = IFFConfig(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            no_response_hostile_threshold=1,
        )
        interrog = IFFInterrogator(cfg, seed=42)

        friendly_tp = IFFTransponder(
            enabled=True,
            modes=[IFFMode.MODE_3A, IFFMode.MODE_4],
            mode_3a_code="1200",
            mode_4_valid=True,
        )
        targets = [
            ("FRIEND-1", friendly_tp, 5000.0),
            ("HOSTILE-1", None, 8000.0),
            ("HOSTILE-2", IFFTransponder(enabled=False), 6000.0),
        ]

        results = interrog.interrogate_all(targets, timestamp=0.0)
        assert "FRIEND-1" in results
        assert "HOSTILE-1" in results
        assert "HOSTILE-2" in results

        assert results["FRIEND-1"].identification == IFFCode.FRIENDLY
        assert results["HOSTILE-1"].identification in (
            IFFCode.ASSUMED_HOSTILE, IFFCode.PENDING
        )

    def test_get_result(self):
        cfg = IFFConfig(enabled=True, modes=[IFFMode.MODE_3A])
        interrog = IFFInterrogator(cfg, seed=42)

        tp = IFFTransponder(
            enabled=True, modes=[IFFMode.MODE_3A], mode_3a_code="1200"
        )
        interrog.interrogate_all([("TGT-1", tp, 5000.0)], timestamp=0.0)

        result = interrog.get_result("TGT-1")
        assert result is not None
        assert result.total_interrogations == 1

        assert interrog.get_result("NONEXISTENT") is None

    def test_reset(self):
        cfg = IFFConfig(enabled=True, modes=[IFFMode.MODE_3A])
        interrog = IFFInterrogator(cfg, seed=42)

        tp = IFFTransponder(
            enabled=True, modes=[IFFMode.MODE_3A], mode_3a_code="1200"
        )
        interrog.interrogate_all([("TGT-1", tp, 5000.0)], timestamp=0.0)
        assert len(interrog.track_results) == 1

        interrog.reset()
        assert len(interrog.track_results) == 0

    def test_track_results_returns_copy(self):
        cfg = IFFConfig(enabled=True, modes=[IFFMode.MODE_3A])
        interrog = IFFInterrogator(cfg, seed=42)

        r1 = interrog.track_results
        r1["fake"] = IFFResult()
        assert "fake" not in interrog.track_results
