"""Core data types for the SENTINEL tracking system."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


class SensorType(enum.Enum):
    CAMERA = "camera"
    RADAR = "radar"
    THERMAL = "thermal"
    QUANTUM_RADAR = "quantum_radar"


class RadarBand(enum.Enum):
    """Radar frequency bands."""

    VHF = "vhf"  # 30-300 MHz
    UHF = "uhf"  # 300 MHz - 1 GHz
    L_BAND = "l_band"  # 1-2 GHz
    S_BAND = "s_band"  # 2-4 GHz
    X_BAND = "x_band"  # 8-12 GHz


class ThermalBand(enum.Enum):
    """Infrared wavelength bands."""

    SWIR = "swir"  # 0.9-1.7 um
    MWIR = "mwir"  # 3-5 um
    LWIR = "lwir"  # 8-12 um


class TargetType(enum.Enum):
    """Target classification for simulation."""

    CONVENTIONAL = "conventional"
    STEALTH = "stealth"
    HYPERSONIC = "hypersonic"


class TrackState(enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class Detection:
    """Unified detection from any sensor modality."""

    sensor_type: SensorType
    timestamp: float  # epoch seconds

    # Camera fields
    bbox: np.ndarray | None = None  # [x1, y1, x2, y2] pixels
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None

    # Radar fields
    range_m: float | None = None
    azimuth_deg: float | None = None
    velocity_mps: float | None = None
    rcs_dbsm: float | None = None

    # Thermal fields
    elevation_deg: float | None = None
    temperature_k: float | None = None
    thermal_band: str | None = None
    intensity: float | None = None

    # Multi-frequency radar field
    radar_band: str | None = None

    # Quantum illumination radar fields
    qi_advantage_db: float | None = None
    entanglement_fidelity: float | None = None
    n_signal_photons: float | None = None
    receiver_type: str | None = None

    # Electronic warfare
    is_ew_generated: bool = False  # True for chaff/decoy/deceptive detections
    ew_source_id: str | None = None  # Identifies the EW source

    # Shared / fused
    position_3d: np.ndarray | None = None  # [x, y, z] world frame

    @property
    def bbox_center(self) -> np.ndarray | None:
        """Return center [cx, cy] of bounding box."""
        if self.bbox is None:
            return None
        return np.array(
            [
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2,
            ]
        )

    @property
    def bbox_area(self) -> float:
        if self.bbox is None:
            return 0.0
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(max(w, 0) * max(h, 0))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sensor_type": self.sensor_type.value,
            "timestamp": self.timestamp,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox.tolist()
        if self.range_m is not None:
            d["range_m"] = self.range_m
            d["azimuth_deg"] = self.azimuth_deg
            d["velocity_mps"] = self.velocity_mps
        if self.radar_band is not None:
            d["radar_band"] = self.radar_band
        if self.temperature_k is not None:
            d["temperature_k"] = self.temperature_k
            d["thermal_band"] = self.thermal_band
        if self.elevation_deg is not None:
            d["elevation_deg"] = self.elevation_deg
        if self.qi_advantage_db is not None:
            d["qi_advantage_db"] = self.qi_advantage_db
            d["entanglement_fidelity"] = self.entanglement_fidelity
            d["n_signal_photons"] = self.n_signal_photons
            d["receiver_type"] = self.receiver_type
        return d


def generate_track_id() -> str:
    return uuid.uuid4().hex[:8].upper()
