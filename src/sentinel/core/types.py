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


class ThreatLevel(enum.Enum):
    """Threat classification level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IntentType(enum.Enum):
    """Estimated target intent/behavior pattern."""

    TRANSIT = "transit"
    PATROL = "patrol"
    APPROACH = "approach"
    ATTACK = "attack"
    EVASION = "evasion"
    UNKNOWN = "unknown"


class IFFCode(enum.Enum):
    """IFF identification result."""

    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    UNKNOWN = "unknown"
    PENDING = "pending"
    ASSUMED_FRIENDLY = "assumed_friendly"
    ASSUMED_HOSTILE = "assumed_hostile"
    SPOOF_SUSPECT = "spoof_suspect"


class IFFMode(enum.Enum):
    """IFF interrogation modes."""

    MODE_1 = "mode_1"  # Military mission code (2-digit octal)
    MODE_2 = "mode_2"  # Military unit code (4-digit octal)
    MODE_3A = "mode_3a"  # Civil/military squawk (4-digit octal)
    MODE_C = "mode_c"  # Altitude reporting (100ft increments)
    MODE_S = "mode_s"  # 24-bit ICAO address + selective interrogation
    MODE_4 = "mode_4"  # Cryptographic military (challenge-response)
    MODE_5 = "mode_5"  # Enhanced crypto military (AES-128)


class EngagementAuth(enum.Enum):
    """Rules of Engagement authorization status."""

    WEAPONS_FREE = "weapons_free"
    WEAPONS_TIGHT = "weapons_tight"
    WEAPONS_HOLD = "weapons_hold"
    HOLD_FIRE = "hold_fire"


class NodeRole(enum.Enum):
    """Role of a node in the tactical network."""

    SENSOR = "sensor"
    FUSION_CENTER = "fusion_center"
    COMMAND = "command"
    RELAY = "relay"


class NodeState(enum.Enum):
    """State of a network node."""

    JOINING = "joining"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class MessageType(enum.Enum):
    """Tactical network message types (Link 16 J-series inspired)."""

    TRACK_REPORT = "track_report"
    DETECTION_REPORT = "detection_report"
    IFF_REPORT = "iff_report"
    ENGAGEMENT_STATUS = "engagement_status"
    HEARTBEAT = "heartbeat"
    SENSOR_STATUS = "sensor_status"


class ZoneAuth(enum.Enum):
    """Geographic zone engagement authorization."""

    NO_FIRE = "no_fire"
    RESTRICTED_FIRE = "restricted_fire"
    SELF_DEFENSE_ONLY = "self_defense_only"
    WEAPONS_FREE = "weapons_free"


class WeaponType(enum.Enum):
    """Weapon system categories."""

    SAM_SHORT = "sam_short"
    SAM_MEDIUM = "sam_medium"
    SAM_LONG = "sam_long"
    AAM_SHORT = "aam_short"
    AAM_MEDIUM = "aam_medium"
    CIWS = "ciws"
    GUN = "gun"


class EngagementStatus(enum.Enum):
    """Status of an engagement assignment."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ABORTED = "aborted"


class RecordingState(enum.Enum):
    """State of the history recorder."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class PlaybackState(enum.Enum):
    """State of the replay controller."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    STEPPING = "stepping"


class L16Identity(enum.Enum):
    """Link 16 track identity (3-bit, STANAG 5516)."""

    PENDING = "pending"
    UNKNOWN = "unknown"
    ASSUMED_FRIEND = "assumed_friend"
    FRIEND = "friend"
    NEUTRAL = "neutral"
    SUSPECT = "suspect"
    HOSTILE = "hostile"
    JOKER = "joker"


class L16MessageType(enum.Enum):
    """Link 16 J-series message types."""

    J2_2 = "J2.2"
    J3_2 = "J3.2"
    J3_5 = "J3.5"
    J7_0 = "J7.0"


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

    # Source target ID (propagated from simulator for IFF matching)
    target_id: str | None = None

    # Shared / fused
    position_3d: np.ndarray | None = None  # [x, y, z] world frame

    @property
    def bbox_center(self) -> np.ndarray | None:
        """Return center [cx, cy] of bounding box (cached after first call)."""
        if self.bbox is None:
            return None
        # Cache to avoid repeated allocation in association loops
        cached = getattr(self, '_bbox_center_cache', None)
        if cached is not None:
            return cached
        center = np.array(
            [
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2,
            ]
        )
        object.__setattr__(self, '_bbox_center_cache', center)
        return center

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
