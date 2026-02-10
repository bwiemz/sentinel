"""Gateway statistics and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GatewayStats:
    """Track gateway performance and message counts."""

    messages_sent: int = 0
    messages_received: int = 0
    messages_invalid: int = 0
    encode_errors: int = 0
    decode_errors: int = 0
    tracks_published: int = 0
    tracks_received: int = 0
    tracks_skipped: int = 0
    iff_published: int = 0
    iff_received: int = 0
    engagement_published: int = 0
    engagement_received: int = 0

    def to_dict(self) -> dict:
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_invalid": self.messages_invalid,
            "encode_errors": self.encode_errors,
            "decode_errors": self.decode_errors,
            "tracks_published": self.tracks_published,
            "tracks_received": self.tracks_received,
            "tracks_skipped": self.tracks_skipped,
            "iff_published": self.iff_published,
            "iff_received": self.iff_received,
            "engagement_published": self.engagement_published,
            "engagement_received": self.engagement_received,
        }

    def reset(self) -> None:
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_invalid = 0
        self.encode_errors = 0
        self.decode_errors = 0
        self.tracks_published = 0
        self.tracks_received = 0
        self.tracks_skipped = 0
        self.iff_published = 0
        self.iff_received = 0
        self.engagement_published = 0
        self.engagement_received = 0
