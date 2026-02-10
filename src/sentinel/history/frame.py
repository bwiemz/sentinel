"""HistoryFrame — one point-in-time snapshot of pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HistoryFrame:
    """Immutable snapshot of pipeline state for recording/replay.

    All fields are plain Python types (dicts, lists, floats) — no numpy
    arrays or live track objects.  This ensures the frame is safely
    serializable and thread-safe to read without locks.

    NOTE: Does NOT include HUD JPEG frames (too memory-intensive).
    """

    frame_number: int
    timestamp: float  # epoch seconds from clock.now()
    elapsed: float  # seconds since pipeline start
    camera_tracks: list[dict[str, Any]] = field(default_factory=list)
    radar_tracks: list[dict[str, Any]] = field(default_factory=list)
    thermal_tracks: list[dict[str, Any]] = field(default_factory=list)
    fused_tracks: list[dict[str, Any]] = field(default_factory=list)
    enhanced_fused_tracks: list[dict[str, Any]] = field(default_factory=list)
    system_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for msgpack/json export."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "elapsed": self.elapsed,
            "camera_tracks": self.camera_tracks,
            "radar_tracks": self.radar_tracks,
            "thermal_tracks": self.thermal_tracks,
            "fused_tracks": self.fused_tracks,
            "enhanced_fused_tracks": self.enhanced_fused_tracks,
            "system_status": self.system_status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HistoryFrame:
        """Reconstruct from a deserialized dict."""
        return cls(
            frame_number=d["frame_number"],
            timestamp=d["timestamp"],
            elapsed=d.get("elapsed", 0.0),
            camera_tracks=d.get("camera_tracks", []),
            radar_tracks=d.get("radar_tracks", []),
            thermal_tracks=d.get("thermal_tracks", []),
            fused_tracks=d.get("fused_tracks", []),
            enhanced_fused_tracks=d.get("enhanced_fused_tracks", []),
            system_status=d.get("system_status", {}),
        )

    @property
    def track_count(self) -> int:
        """Total number of tracks across all sensor modalities."""
        return (
            len(self.camera_tracks)
            + len(self.radar_tracks)
            + len(self.thermal_tracks)
            + len(self.fused_tracks)
            + len(self.enhanced_fused_tracks)
        )

    @property
    def estimated_size_bytes(self) -> int:
        """Rough memory estimate for monitoring."""
        return self.track_count * 200 + 500
