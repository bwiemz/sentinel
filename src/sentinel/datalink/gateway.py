"""DataLinkGateway — Link 16 interoperability gateway for SENTINEL.

Sits alongside (not replacing) the existing NetworkBridge.
Converts between SENTINEL EnhancedFusedTrack objects and Link 16
J-series binary messages for external C2 interoperability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sentinel.datalink.adapter import DataLinkAdapter
from sentinel.datalink.config import DataLinkConfig
from sentinel.datalink.encoding import (
    J2_2Codec,
    J3_2Codec,
    J3_5Codec,
    J7_0Codec,
    decode_message,
    encode_message,
    peek_message_type,
)
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.stats import GatewayStats
from sentinel.datalink.track_mapping import TrackNumberAllocator
from sentinel.datalink.transport import DataLinkTransport, InMemoryDataLinkTransport
from sentinel.datalink.validator import L16Validator
from sentinel.core.types import L16MessageType

logger = logging.getLogger(__name__)


class DataLinkGateway:
    """Link 16 interoperability gateway."""

    def __init__(
        self,
        config: DataLinkConfig,
        transport: DataLinkTransport | None = None,
        geo_context: Any | None = None,
    ) -> None:
        self._config = config
        self._allocator = TrackNumberAllocator(max_entries=config.max_track_numbers)
        self._adapter = DataLinkAdapter(
            geo_context=geo_context,
            track_allocator=self._allocator,
        )
        self._transport = transport or InMemoryDataLinkTransport("sentinel-gateway")
        self._stats = GatewayStats()

        # Rate limiting
        self._min_interval = 1.0 / max(config.publish_rate_hz, 0.01)
        self._last_publish: float = 0.0

        # Inbound buffers
        self._received_tracks: dict[int, dict] = {}
        self._received_iff: dict[int, dict] = {}
        self._received_engagement: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Outbound: SENTINEL -> Link 16
    # ------------------------------------------------------------------

    def publish_tracks(self, tracks: list, current_time: float) -> int:
        """Convert SENTINEL tracks to J2.2 and transmit."""
        now = time.monotonic()
        if now - self._last_publish < self._min_interval:
            return 0
        self._last_publish = now

        sent = 0
        for track in tracks:
            msg = self._adapter.track_to_j2_2(track, current_time)
            if msg is None:
                self._stats.tracks_skipped += 1
                continue

            if self._config.validate_outbound:
                errors = L16Validator.validate_j2_2(msg)
                if errors:
                    self._stats.messages_invalid += 1
                    logger.debug("Outbound J2.2 validation failed: %s", errors[0].message)
                    continue

            try:
                data = J2_2Codec.encode(msg)
                self._transport.send(data)
                self._stats.messages_sent += 1
                self._stats.tracks_published += 1
                sent += 1
            except Exception:
                self._stats.encode_errors += 1
                logger.debug("J2.2 encode failed", exc_info=True)

        return sent

    def publish_iff(self, iff_results: dict, current_time: float) -> int:
        """Convert IFF results to J7.0 and transmit."""
        if not self._config.publish_iff:
            return 0

        sent = 0
        for track_id, result in iff_results.items():
            msg = self._adapter.iff_result_to_j7_0(track_id, result, current_time)
            if msg is None:
                continue
            try:
                data = J7_0Codec.encode(msg)
                self._transport.send(data)
                self._stats.messages_sent += 1
                self._stats.iff_published += 1
                sent += 1
            except Exception:
                self._stats.encode_errors += 1

        return sent

    def publish_engagement(
        self,
        track_id: str,
        auth: str,
        current_time: float,
        weapon_type: int = 0,
        engagement_status: int = 0,
    ) -> None:
        """Convert engagement status to J3.5 and transmit."""
        if not self._config.publish_engagement:
            return

        msg = self._adapter.engagement_to_j3_5(
            track_id, auth, weapon_type, engagement_status, current_time
        )
        if msg is None:
            return

        try:
            data = J3_5Codec.encode(msg)
            self._transport.send(data)
            self._stats.messages_sent += 1
            self._stats.engagement_published += 1
        except Exception:
            self._stats.encode_errors += 1

    def publish_track_drop(self, track_id: str, current_time: float) -> None:
        """Send J3.2 track drop and release track number."""
        # We need a dummy track object for the adapter — use a simple namespace
        class _Stub:
            pass

        stub = _Stub()
        stub.fused_id = track_id  # type: ignore[attr-defined]
        stub.iff_identification = "unknown"  # type: ignore[attr-defined]
        stub.sensor_count = 0  # type: ignore[attr-defined]

        msg = self._adapter.track_to_j3_2(stub, action=1, timestamp=current_time)
        if msg is not None:
            try:
                data = J3_2Codec.encode(msg)
                self._transport.send(data)
                self._stats.messages_sent += 1
            except Exception:
                self._stats.encode_errors += 1

        self._allocator.release(track_id)

    # ------------------------------------------------------------------
    # Inbound: Link 16 -> SENTINEL
    # ------------------------------------------------------------------

    def process_incoming(self) -> int:
        """Process all pending inbound Link 16 messages."""
        if not self._config.accept_inbound:
            return 0

        raw_messages = self._transport.recv_all()
        processed = 0

        for raw in raw_messages:
            try:
                msg = decode_message(raw)
            except Exception:
                self._stats.decode_errors += 1
                logger.debug("Failed to decode inbound L16 message", exc_info=True)
                continue

            if self._config.validate_inbound:
                errors = L16Validator.validate(msg)
                if errors:
                    self._stats.messages_invalid += 1
                    continue

            self._stats.messages_received += 1

            if isinstance(msg, J2_2AirTrack):
                remote = self._adapter.j2_2_to_remote_track(msg)
                self._received_tracks[msg.track_number] = remote
                self._stats.tracks_received += 1
            elif isinstance(msg, J7_0IFF):
                result = self._adapter.j7_0_to_iff_result(msg)
                self._received_iff[msg.track_number] = result
                self._stats.iff_received += 1
            elif isinstance(msg, J3_5EngagementStatus):
                eng = self._adapter.j3_5_to_engagement(msg)
                self._received_engagement[msg.track_number] = eng
                self._stats.engagement_received += 1
            elif isinstance(msg, J3_2TrackManagement):
                if msg.action == 1:  # drop
                    self._received_tracks.pop(msg.track_number, None)

            processed += 1

        return processed

    def get_received_tracks(self) -> list[dict]:
        return list(self._received_tracks.values())

    def get_received_iff(self) -> dict[int, dict]:
        return dict(self._received_iff)

    def get_received_engagement(self) -> dict[int, dict]:
        return dict(self._received_engagement)

    def clear_buffers(self) -> None:
        self._received_tracks.clear()
        self._received_iff.clear()
        self._received_engagement.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return self._stats.to_dict()

    @property
    def transport(self) -> DataLinkTransport:
        return self._transport

    @property
    def allocator(self) -> TrackNumberAllocator:
        return self._allocator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        geo_context: Any | None = None,
    ) -> DataLinkGateway | None:
        """Build from OmegaConf or dict. Returns None if disabled."""
        config = DataLinkConfig.from_omegaconf(cfg)
        if not config.enabled:
            return None
        return cls(config=config, geo_context=geo_context)
