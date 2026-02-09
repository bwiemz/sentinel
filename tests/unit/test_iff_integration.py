"""Integration tests for IFF + fusion pipeline."""

import numpy as np
import pytest

from sentinel.core.types import IFFCode, IFFMode, EngagementAuth
from sentinel.sensors.iff import IFFConfig, IFFInterrogator, IFFTransponder, IFFResult
from sentinel.classification.roe import ROEConfig, ROEEngine
from sentinel.fusion.multi_sensor_fusion import (
    EnhancedFusedTrack,
    MultiSensorFusion,
)
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.thermal_track import ThermalTrack


# ===================================================================
# Helpers
# ===================================================================


def _make_radar_track(track_id: str, range_m=5000.0, azimuth_deg=10.0, target_id=None):
    """Create a minimal RadarTrack for testing."""
    from unittest.mock import MagicMock

    t = MagicMock(spec=RadarTrack)
    t.track_id = track_id
    t.range_m = range_m
    t.azimuth_deg = azimuth_deg
    t.is_alive = True
    t.score = 5.0
    t.state = MagicMock()
    t.state.value = "confirmed"
    t.last_detection = MagicMock()
    t.last_detection.target_id = target_id
    t.position = np.array([range_m * np.cos(np.radians(azimuth_deg)),
                           range_m * np.sin(np.radians(azimuth_deg))])
    t.velocity = np.array([-50.0, 0.0])
    t.position_geo = None
    t.velocity_mps = 100.0
    return t


def _make_iff_result(
    identification=IFFCode.FRIENDLY,
    confidence=0.9,
    mode_3a_code="1200",
    mode_s_address=None,
    last_auth_mode=IFFMode.MODE_4,
    spoof_indicators=0,
):
    return IFFResult(
        identification=identification,
        confidence=confidence,
        mode_3a_code=mode_3a_code,
        mode_s_address=mode_s_address,
        last_authenticated_mode=last_auth_mode,
        spoof_indicators=spoof_indicators,
    )


# ===================================================================
# EnhancedFusedTrack IFF fields
# ===================================================================


class TestEnhancedFusedTrackIFF:
    def test_default_iff_fields(self):
        eft = EnhancedFusedTrack(fused_id="test")
        assert eft.iff_identification == "unknown"
        assert eft.iff_confidence == 0.0
        assert eft.iff_spoof_suspect is False
        assert eft.engagement_auth == "weapons_hold"

    def test_to_dict_includes_iff(self):
        eft = EnhancedFusedTrack(
            fused_id="test",
            iff_identification="friendly",
            iff_confidence=0.95,
            iff_spoof_suspect=False,
            engagement_auth="hold_fire",
        )
        d = eft.to_dict()
        assert d["iff_identification"] == "friendly"
        assert d["iff_confidence"] == 0.95
        assert d["iff_spoof_suspect"] is False
        assert d["engagement_auth"] == "hold_fire"


# ===================================================================
# MultiSensorFusion with IFF
# ===================================================================


class TestFusionWithIFF:
    def _make_fusion(self, iff_config=None, roe_config=None, controlled=False):
        iff_int = IFFInterrogator(iff_config) if iff_config else None
        roe_eng = ROEEngine(roe_config) if roe_config else None
        return MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            iff_interrogator=iff_int,
            roe_engine=roe_eng,
            controlled_airspace=controlled,
        )

    def test_fusion_without_iff_unchanged(self):
        """Without IFF config, fused tracks have default IFF fields."""
        fusion = self._make_fusion()
        rt = _make_radar_track("R1", target_id="TGT-01")
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
        )
        assert len(fused) >= 1
        eft = fused[0]
        assert eft.iff_identification == "unknown"
        assert eft.engagement_auth == "weapons_hold"

    def test_fusion_with_friendly_iff(self):
        """Friendly IFF result is applied to fused track."""
        fusion = self._make_fusion(
            iff_config=IFFConfig(enabled=True),
            roe_config=ROEConfig(enabled=True),
        )
        rt = _make_radar_track("R1", target_id="TGT-01")
        iff_results = {
            "TGT-01": _make_iff_result(
                identification=IFFCode.FRIENDLY,
                confidence=0.95,
                last_auth_mode=IFFMode.MODE_4,
            ),
        }
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            iff_results=iff_results,
        )
        eft = fused[0]
        assert eft.iff_identification == "friendly"
        assert eft.iff_confidence == 0.95
        assert eft.engagement_auth == "hold_fire"

    def test_fusion_with_hostile_iff(self):
        """Hostile IFF + attack intent = WEAPONS_FREE."""
        fusion = self._make_fusion(
            iff_config=IFFConfig(enabled=True),
            roe_config=ROEConfig(enabled=True),
        )
        rt = _make_radar_track("R1", target_id="TGT-02")
        iff_results = {
            "TGT-02": _make_iff_result(
                identification=IFFCode.HOSTILE,
                confidence=0.8,
                last_auth_mode=None,
                spoof_indicators=0,
            ),
        }
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            iff_results=iff_results,
        )
        eft = fused[0]
        assert eft.iff_identification == "hostile"
        # ROE: hostile + default intent "unknown" → hostile_no_attack → WEAPONS_TIGHT
        assert eft.engagement_auth == "weapons_tight"

    def test_fusion_spoof_boosts_threat(self):
        """SPOOF_SUSPECT should boost threat to at least HIGH."""
        fusion = self._make_fusion(
            iff_config=IFFConfig(enabled=True),
            roe_config=ROEConfig(enabled=True),
        )
        rt = _make_radar_track("R1", target_id="TGT-03")
        iff_results = {
            "TGT-03": _make_iff_result(
                identification=IFFCode.SPOOF_SUSPECT,
                confidence=0.7,
                spoof_indicators=2,
            ),
        }
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            iff_results=iff_results,
        )
        eft = fused[0]
        assert eft.iff_spoof_suspect is True
        assert eft.engagement_auth == "weapons_free"  # spoof → WEAPONS_FREE

    def test_fusion_friendly_caps_threat(self):
        """Friendly IFF should cap threat at MEDIUM even for high threat targets."""
        fusion = self._make_fusion(
            iff_config=IFFConfig(enabled=True),
        )
        rt = _make_radar_track("R1", target_id="TGT-04")
        iff_results = {
            "TGT-04": _make_iff_result(
                identification=IFFCode.FRIENDLY,
                confidence=0.9,
            ),
        }
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            iff_results=iff_results,
        )
        eft = fused[0]
        # Threat should be at most MEDIUM for friendly
        assert eft.threat_level in ("LOW", "MEDIUM", "UNKNOWN")

    def test_no_matching_iff_result(self):
        """If target_id doesn't match any IFF result, defaults are kept."""
        fusion = self._make_fusion(iff_config=IFFConfig(enabled=True))
        rt = _make_radar_track("R1", target_id="NO-MATCH")
        iff_results = {
            "TGT-01": _make_iff_result(),
        }
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            iff_results=iff_results,
        )
        eft = fused[0]
        assert eft.iff_identification == "unknown"


# ===================================================================
# Feature extraction with IFF
# ===================================================================


class TestFeatureExtractionIFF:
    def test_iff_features_friendly(self):
        from sentinel.classification.features import FeatureExtractor

        extractor = FeatureExtractor()
        eft = EnhancedFusedTrack(
            fused_id="test",
            iff_identification="friendly",
            iff_last_auth_mode="mode_4",
        )
        features = extractor.extract(eft)
        assert features[28] == 1.0  # iff_is_friendly
        assert features[29] == 0.0  # iff_is_hostile
        assert features[30] == 1.0  # iff_has_crypto_auth
        assert features[31] == 0.0  # iff_spoof_suspect

    def test_iff_features_hostile_spoof(self):
        from sentinel.classification.features import FeatureExtractor

        extractor = FeatureExtractor()
        eft = EnhancedFusedTrack(
            fused_id="test",
            iff_identification="hostile",
            iff_spoof_suspect=True,
        )
        features = extractor.extract(eft)
        assert features[28] == 0.0  # iff_is_friendly
        assert features[29] == 1.0  # iff_is_hostile
        assert features[30] == 0.0  # iff_has_crypto_auth
        assert features[31] == 1.0  # iff_spoof_suspect

    def test_feature_vector_length(self):
        from sentinel.classification.features import FeatureExtractor, FEATURE_NAMES

        extractor = FeatureExtractor()
        eft = EnhancedFusedTrack(fused_id="test")
        features = extractor.extract(eft)
        assert len(features) == len(FEATURE_NAMES)
        assert len(features) == 32


# ===================================================================
# Config schema validation
# ===================================================================


class TestConfigSchema:
    def test_iff_config_schema_validates(self):
        from sentinel.core.config_schema import IFFConfigSchema
        cfg = IFFConfigSchema()
        assert cfg.enabled is False
        assert cfg.max_interrogation_range_m == 400_000.0

    def test_roe_config_schema_validates(self):
        from sentinel.core.config_schema import ROEConfigSchema
        cfg = ROEConfigSchema()
        assert cfg.enabled is False
        assert cfg.default_posture == "weapons_hold"

    def test_root_config_includes_iff_roe(self):
        from sentinel.core.config_schema import SentinelRootConfig
        cfg = SentinelRootConfig()
        assert hasattr(cfg, "iff")
        assert hasattr(cfg, "roe")
        assert cfg.iff.enabled is False
        assert cfg.roe.enabled is False
