"""Shared fixtures for integration tests."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from sentinel.core.types import SensorType
from sentinel.sensors.frame import SensorFrame


def _minimal_pipeline_config(**overrides):
    """Build a config suitable for integration testing with mocked camera."""
    cfg = OmegaConf.create(
        {
            "sentinel": {
                "system": {
                    "name": "SENTINEL-TEST",
                    "version": "0.0.1",
                    "log_level": "WARNING",
                    "graceful_degradation": True,
                    "max_sensor_errors": 5,
                },
                "sensors": {
                    "camera": {
                        "enabled": True,
                        "source": 0,
                        "width": 320,
                        "height": 240,
                        "fps": 30,
                        "buffer_size": 1,
                        "backend": "auto",
                    },
                    "radar": {
                        "enabled": False,
                        "mode": "simulator",
                        "scan_rate_hz": 10,
                        "max_range_m": 10000,
                        "fov_deg": 120,
                        "noise": {
                            "range_m": 5.0,
                            "azimuth_deg": 1.0,
                            "velocity_mps": 0.5,
                            "rcs_dbsm": 2.0,
                            "false_alarm_rate": 0.0,
                            "detection_probability": 1.0,
                        },
                        "scenario": {"targets": []},
                    },
                },
                "detection": {
                    "model": "yolov8n.pt",
                    "confidence": 0.25,
                    "iou_threshold": 0.45,
                    "device": "cpu",
                    "classes": None,
                    "max_detections": 100,
                    "image_size": 640,
                },
                "tracking": {
                    "filter": {"type": "kf", "dt": 0.033, "process_noise_std": 1.0, "measurement_noise_std": 10.0},
                    "association": {
                        "method": "hungarian",
                        "gate_threshold": 9.21,
                        "iou_weight": 0.5,
                        "mahalanobis_weight": 0.5,
                    },
                    "track_management": {
                        "confirm_hits": 3,
                        "confirm_window": 5,
                        "max_coast_frames": 15,
                        "max_tracks": 100,
                        "tentative_delete_misses": 3,
                    },
                    "radar": {
                        "filter": {"dt": 0.1, "type": "ekf"},
                        "association": {"gate_threshold": 9.21},
                        "track_management": {"confirm_hits": 3, "max_coast_frames": 5, "max_tracks": 50},
                    },
                },
                "fusion": {"enabled": False},
                "ui": {"hud": {"enabled": False, "display": False}},
            }
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def make_camera_frame(timestamp: float = 1.0, frame_number: int = 1) -> SensorFrame:
    """Create a synthetic camera SensorFrame."""
    return SensorFrame(
        data=np.zeros((240, 320, 3), dtype=np.uint8),
        timestamp=timestamp,
        sensor_type=SensorType.CAMERA,
        frame_number=frame_number,
    )
