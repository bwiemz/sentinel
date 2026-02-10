"""Tactical network message catalog — Link 16 J-series inspired.

All messages use msgpack binary serialization for compact, fast wire format.
Falls back to JSON serialization if msgpack is not installed.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sentinel.core.types import MessageType

# Optional msgpack — fall back to JSON
try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:  # pragma: no cover
    _HAS_MSGPACK = False


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _default_serializer(obj: Any) -> Any:
    """Convert non-serializable objects for msgpack/JSON."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, MessageType):
        return obj.value
    raise TypeError(f"Cannot serialize {type(obj)}")


def _object_hook(obj: Any) -> Any:
    """Reconstruct numpy arrays from deserialized dicts."""
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj


def _serialize(data: dict) -> bytes:
    """Serialize dict to bytes (msgpack if available, else JSON)."""
    if _HAS_MSGPACK:
        return msgpack.packb(data, default=_default_serializer, use_bin_type=True)
    return json.dumps(data, default=_default_serializer).encode("utf-8")


def _deserialize(raw: bytes) -> dict:
    """Deserialize bytes to dict."""
    if _HAS_MSGPACK:
        d = msgpack.unpackb(raw, raw=False)
    else:
        d = json.loads(raw.decode("utf-8"))
    return _walk_object_hook(d)


def _walk_object_hook(obj: Any) -> Any:
    """Recursively apply object hook to nested dicts."""
    if isinstance(obj, dict):
        obj = {k: _walk_object_hook(v) for k, v in obj.items()}
        return _object_hook(obj)
    if isinstance(obj, list):
        return [_walk_object_hook(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# NetworkMessage
# ---------------------------------------------------------------------------


@dataclass
class NetworkMessage:
    """A tactical network message with header + payload.

    Inspired by Link 16 J-series messages with CEC-level detail.
    Priority levels: 0=routine, 1=priority, 2=immediate, 3=flash.
    """

    msg_type: MessageType
    source_node: str
    timestamp: float
    payload: dict = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    priority: int = 0
    ttl: int = 3
    sequence_num: int = 0

    def serialize(self) -> bytes:
        """Serialize to bytes for network transmission."""
        data = {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "source_node": self.source_node,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "ttl": self.ttl,
            "sequence_num": self.sequence_num,
            "payload": self.payload,
        }
        return _serialize(data)

    @classmethod
    def deserialize(cls, raw: bytes) -> NetworkMessage:
        """Deserialize from bytes."""
        data = _deserialize(raw)
        return cls(
            msg_id=data["msg_id"],
            msg_type=MessageType(data["msg_type"]),
            source_node=data["source_node"],
            timestamp=data["timestamp"],
            priority=data.get("priority", 0),
            ttl=data.get("ttl", 3),
            sequence_num=data.get("sequence_num", 0),
            payload=data.get("payload", {}),
        )

    def decrement_ttl(self) -> NetworkMessage:
        """Return copy with TTL decremented (for forwarding)."""
        return NetworkMessage(
            msg_id=self.msg_id,
            msg_type=self.msg_type,
            source_node=self.source_node,
            timestamp=self.timestamp,
            payload=self.payload,
            priority=self.priority,
            ttl=self.ttl - 1,
            sequence_num=self.sequence_num,
        )

    @property
    def is_expired(self) -> bool:
        """True if TTL has reached zero."""
        return self.ttl <= 0

    @property
    def size_bytes(self) -> int:
        """Approximate serialized size (cached after first call)."""
        cached = getattr(self, '_cached_size', None)
        if cached is not None:
            return cached
        size = len(self.serialize())
        object.__setattr__(self, '_cached_size', size)
        return size


# ---------------------------------------------------------------------------
# Payload builders (convenience factories)
# ---------------------------------------------------------------------------


def make_track_report(
    source_node: str,
    timestamp: float,
    track_id: str,
    position: list[float] | np.ndarray,
    velocity: list[float] | np.ndarray,
    covariance: list[list[float]] | np.ndarray | None = None,
    position_geo: tuple[float, float, float] | None = None,
    sensor_types: list[str] | None = None,
    threat_level: str = "UNKNOWN",
    iff_identification: str = "unknown",
    engagement_auth: str = "weapons_hold",
    confidence: float = 0.0,
    update_time: float | None = None,
) -> NetworkMessage:
    """Build a TRACK_REPORT message (CEC-inspired composite tracking)."""
    pos = position.tolist() if isinstance(position, np.ndarray) else list(position)
    vel = velocity.tolist() if isinstance(velocity, np.ndarray) else list(velocity)
    cov = None
    if covariance is not None:
        cov = (
            covariance.tolist()
            if isinstance(covariance, np.ndarray)
            else covariance
        )
    payload = {
        "track_id": track_id,
        "position": pos,
        "velocity": vel,
        "covariance": cov,
        "position_geo": list(position_geo) if position_geo else None,
        "sensor_types": sensor_types or [],
        "threat_level": threat_level,
        "iff_identification": iff_identification,
        "engagement_auth": engagement_auth,
        "confidence": confidence,
        "update_time": update_time or timestamp,
    }
    return NetworkMessage(
        msg_type=MessageType.TRACK_REPORT,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=1,  # priority
    )


def make_detection_report(
    source_node: str,
    timestamp: float,
    sensor_type: str,
    range_m: float | None = None,
    azimuth_deg: float | None = None,
    elevation_deg: float | None = None,
    velocity_mps: float | None = None,
    rcs_dbsm: float | None = None,
    position: list[float] | None = None,
) -> NetworkMessage:
    """Build a DETECTION_REPORT message."""
    payload = {
        "sensor_type": sensor_type,
        "range_m": range_m,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "velocity_mps": velocity_mps,
        "rcs_dbsm": rcs_dbsm,
        "position": position,
    }
    return NetworkMessage(
        msg_type=MessageType.DETECTION_REPORT,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=0,  # routine
    )


def make_iff_report(
    source_node: str,
    timestamp: float,
    target_id: str,
    identification: str,
    confidence: float,
    mode_3a_code: str | None = None,
    mode_s_address: str | None = None,
    spoof_suspect: bool = False,
) -> NetworkMessage:
    """Build an IFF_REPORT message."""
    payload = {
        "target_id": target_id,
        "identification": identification,
        "confidence": confidence,
        "mode_3a_code": mode_3a_code,
        "mode_s_address": mode_s_address,
        "spoof_suspect": spoof_suspect,
    }
    return NetworkMessage(
        msg_type=MessageType.IFF_REPORT,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=2,  # immediate
    )


def make_engagement_status(
    source_node: str,
    timestamp: float,
    track_id: str,
    engagement_auth: str,
    reason: str = "",
) -> NetworkMessage:
    """Build an ENGAGEMENT_STATUS message."""
    payload = {
        "track_id": track_id,
        "engagement_auth": engagement_auth,
        "reason": reason,
    }
    return NetworkMessage(
        msg_type=MessageType.ENGAGEMENT_STATUS,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=3,  # flash
    )


def make_heartbeat(
    source_node: str,
    timestamp: float,
    state: str = "active",
    capabilities: list[str] | None = None,
    track_count: int = 0,
    uptime_s: float = 0.0,
) -> NetworkMessage:
    """Build a HEARTBEAT message."""
    payload = {
        "state": state,
        "capabilities": capabilities or [],
        "track_count": track_count,
        "uptime_s": uptime_s,
    }
    return NetworkMessage(
        msg_type=MessageType.HEARTBEAT,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=0,  # routine
    )


def make_sensor_status(
    source_node: str,
    timestamp: float,
    sensor_type: str,
    operational: bool = True,
    mode: str = "normal",
    coverage_deg: float = 120.0,
    max_range_m: float = 50000.0,
) -> NetworkMessage:
    """Build a SENSOR_STATUS message."""
    payload = {
        "sensor_type": sensor_type,
        "operational": operational,
        "mode": mode,
        "coverage_deg": coverage_deg,
        "max_range_m": max_range_m,
    }
    return NetworkMessage(
        msg_type=MessageType.SENSOR_STATUS,
        source_node=source_node,
        timestamp=timestamp,
        payload=payload,
        priority=1,  # priority
    )
