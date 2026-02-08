"""Tests for Pydantic config schema validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from sentinel.core.config_schema import (
    SentinelConfigSchema,
    validate_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_default_dict():
    """Load config/default.yaml as a plain dict."""
    cfg = OmegaConf.load("config/default.yaml")
    return OmegaConf.to_container(cfg, resolve=True)


# ---------------------------------------------------------------------------
# Valid config passes
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_default_yaml_passes(self):
        """The shipped default.yaml should validate without errors."""
        d = _load_default_dict()
        schema = validate_config(d)
        assert isinstance(schema, SentinelConfigSchema)
        assert schema.sentinel.system.name == "SENTINEL"

    def test_minimal_config_passes(self):
        """Minimal config with just the required structure."""
        d = {"sentinel": {}}
        schema = validate_config(d)
        assert schema.sentinel.system.name == "SENTINEL"

    def test_all_defaults_populated(self):
        """All default values should be present after validation."""
        d = {"sentinel": {}}
        schema = validate_config(d)
        assert schema.sentinel.sensors.camera.width == 1280
        assert schema.sentinel.detection.confidence == 0.25
        assert schema.sentinel.tracking.filter.type == "kf"

    def test_overridden_values_preserved(self):
        d = {"sentinel": {"system": {"name": "CUSTOM"}, "detection": {"confidence": 0.5}}}
        schema = validate_config(d)
        assert schema.sentinel.system.name == "CUSTOM"
        assert schema.sentinel.detection.confidence == 0.5

    def test_extra_keys_allowed(self):
        """Unknown keys should not cause validation failure (forward compat)."""
        d = {"sentinel": {"future_feature": {"setting": 42}}}
        schema = validate_config(d)
        assert schema.sentinel.system.name == "SENTINEL"


# ---------------------------------------------------------------------------
# System config validation
# ---------------------------------------------------------------------------


class TestSystemValidation:
    def test_invalid_log_level_rejected(self):
        d = {"sentinel": {"system": {"log_level": "VERBOSE"}}}
        with pytest.raises(ValidationError, match="log_level"):
            validate_config(d)

    def test_max_sensor_errors_must_be_positive(self):
        d = {"sentinel": {"system": {"max_sensor_errors": 0}}}
        with pytest.raises(ValidationError, match="max_sensor_errors"):
            validate_config(d)

    def test_negative_max_sensor_errors_rejected(self):
        d = {"sentinel": {"system": {"max_sensor_errors": -5}}}
        with pytest.raises(ValidationError, match="max_sensor_errors"):
            validate_config(d)


# ---------------------------------------------------------------------------
# Sensor config validation
# ---------------------------------------------------------------------------


class TestSensorValidation:
    def test_negative_camera_width_rejected(self):
        d = {"sentinel": {"sensors": {"camera": {"width": -100}}}}
        with pytest.raises(ValidationError, match="width"):
            validate_config(d)

    def test_zero_fps_rejected(self):
        d = {"sentinel": {"sensors": {"camera": {"fps": 0}}}}
        with pytest.raises(ValidationError, match="fps"):
            validate_config(d)

    def test_negative_scan_rate_rejected(self):
        d = {"sentinel": {"sensors": {"radar": {"scan_rate_hz": -1}}}}
        with pytest.raises(ValidationError, match="scan_rate_hz"):
            validate_config(d)

    def test_fov_over_360_rejected(self):
        d = {"sentinel": {"sensors": {"radar": {"fov_deg": 400}}}}
        with pytest.raises(ValidationError, match="fov_deg"):
            validate_config(d)

    def test_detection_prob_out_of_range(self):
        d = {"sentinel": {"sensors": {"radar": {"noise": {"detection_probability": 1.5}}}}}
        with pytest.raises(ValidationError, match="detection_probability"):
            validate_config(d)

    def test_false_alarm_rate_negative(self):
        d = {"sentinel": {"sensors": {"radar": {"noise": {"false_alarm_rate": -0.1}}}}}
        with pytest.raises(ValidationError, match="false_alarm_rate"):
            validate_config(d)

    def test_quantum_receiver_invalid(self):
        d = {"sentinel": {"sensors": {"quantum_radar": {"receiver_type": "teleportation"}}}}
        with pytest.raises(ValidationError, match="receiver_type"):
            validate_config(d)


# ---------------------------------------------------------------------------
# Detection config validation
# ---------------------------------------------------------------------------


class TestDetectionValidation:
    def test_confidence_above_1_rejected(self):
        d = {"sentinel": {"detection": {"confidence": 2.0}}}
        with pytest.raises(ValidationError, match="confidence"):
            validate_config(d)

    def test_confidence_negative_rejected(self):
        d = {"sentinel": {"detection": {"confidence": -0.5}}}
        with pytest.raises(ValidationError, match="confidence"):
            validate_config(d)

    def test_iou_above_1_rejected(self):
        d = {"sentinel": {"detection": {"iou_threshold": 1.5}}}
        with pytest.raises(ValidationError, match="iou_threshold"):
            validate_config(d)

    def test_max_detections_zero_rejected(self):
        d = {"sentinel": {"detection": {"max_detections": 0}}}
        with pytest.raises(ValidationError, match="max_detections"):
            validate_config(d)

    def test_image_size_zero_rejected(self):
        d = {"sentinel": {"detection": {"image_size": 0}}}
        with pytest.raises(ValidationError, match="image_size"):
            validate_config(d)


# ---------------------------------------------------------------------------
# Tracking config validation
# ---------------------------------------------------------------------------


class TestTrackingValidation:
    def test_invalid_filter_type_rejected(self):
        d = {"sentinel": {"tracking": {"filter": {"type": "particle"}}}}
        with pytest.raises(ValidationError, match="type"):
            validate_config(d)

    def test_valid_filter_types_accepted(self):
        for ft in ("kf", "ekf", "ca", "imm"):
            d = {"sentinel": {"tracking": {"filter": {"type": ft}}}}
            schema = validate_config(d)
            assert schema.sentinel.tracking.filter.type == ft

    def test_zero_dt_rejected(self):
        d = {"sentinel": {"tracking": {"filter": {"dt": 0}}}}
        with pytest.raises(ValidationError, match="dt"):
            validate_config(d)

    def test_negative_process_noise_rejected(self):
        d = {"sentinel": {"tracking": {"filter": {"process_noise_std": -1.0}}}}
        with pytest.raises(ValidationError, match="process_noise_std"):
            validate_config(d)

    def test_gate_threshold_zero_rejected(self):
        d = {"sentinel": {"tracking": {"association": {"gate_threshold": 0}}}}
        with pytest.raises(ValidationError, match="gate_threshold"):
            validate_config(d)

    def test_imm_transition_prob_over_1_rejected(self):
        d = {"sentinel": {"tracking": {"filter": {"imm": {"transition_prob": 1.5}}}}}
        with pytest.raises(ValidationError, match="transition_prob"):
            validate_config(d)

    def test_max_tracks_zero_rejected(self):
        d = {"sentinel": {"tracking": {"track_management": {"max_tracks": 0}}}}
        with pytest.raises(ValidationError, match="max_tracks"):
            validate_config(d)


# ---------------------------------------------------------------------------
# Fusion config validation
# ---------------------------------------------------------------------------


class TestFusionValidation:
    def test_min_fusion_quality_out_of_range(self):
        d = {"sentinel": {"fusion": {"min_fusion_quality": 2.0}}}
        with pytest.raises(ValidationError, match="min_fusion_quality"):
            validate_config(d)


# ---------------------------------------------------------------------------
# UI config validation
# ---------------------------------------------------------------------------


class TestUIValidation:
    def test_overlay_alpha_out_of_range(self):
        d = {"sentinel": {"ui": {"hud": {"overlay_alpha": 5.0}}}}
        with pytest.raises(ValidationError, match="overlay_alpha"):
            validate_config(d)

    def test_port_out_of_range(self):
        d = {"sentinel": {"ui": {"web": {"port": 99999}}}}
        with pytest.raises(ValidationError, match="port"):
            validate_config(d)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_validate_false_is_noop(self):
        """When validate=False (default), invalid config should load fine."""
        from sentinel.core.config import SentinelConfig

        config = SentinelConfig("config/default.yaml")
        cfg = config.load(validate=False)
        assert cfg.sentinel.system.name == "SENTINEL"

    def test_validate_true_passes_for_default(self):
        """validate=True should pass for the default config."""
        from sentinel.core.config import SentinelConfig

        config = SentinelConfig("config/default.yaml")
        cfg = config.load(validate=True)
        assert cfg.sentinel.system.name == "SENTINEL"
