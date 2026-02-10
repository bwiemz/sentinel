"""Network bridge — converts SENTINEL objects to/from network messages.

Handles publishing local tracks/IFF/engagement to the mesh and
receiving remote data from peer nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from sentinel.core.types import MessageType
from sentinel.network.messages import (
    NetworkMessage,
    make_engagement_status,
    make_heartbeat,
    make_iff_report,
    make_sensor_status,
    make_track_report,
)
from sentinel.network.node import NetworkNode
from sentinel.network.pubsub import PubSubBroker
from sentinel.network.transport import SimulatedTransport


# ---------------------------------------------------------------------------
# Remote track representation
# ---------------------------------------------------------------------------


@dataclass
class RemoteTrack:
    """A track received from a remote node."""

    track_id: str
    source_node: str
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray | None = None
    position_geo: tuple[float, float, float] | None = None
    sensor_types: list[str] = field(default_factory=list)
    threat_level: str = "UNKNOWN"
    iff_identification: str = "unknown"
    engagement_auth: str = "weapons_hold"
    confidence: float = 0.0
    update_time: float = 0.0

    @property
    def age(self) -> float:
        """Placeholder — computed externally with current time."""
        return 0.0


# ---------------------------------------------------------------------------
# NetworkBridge
# ---------------------------------------------------------------------------


class NetworkBridge:
    """Converts SENTINEL objects to/from network messages.

    Publishes local tracks, IFF results, and engagement changes to the
    mesh. Subscribes to remote data and buffers it for composite fusion.
    """

    def __init__(
        self,
        node: NetworkNode,
        transport: SimulatedTransport,
        broker: PubSubBroker,
    ):
        self._node = node
        self._transport = transport
        self._broker = broker

        # Buffers for received remote data
        self._remote_tracks: dict[str, list[RemoteTrack]] = {}  # node_id → tracks
        self._remote_iff: dict[str, dict] = {}  # target_id → iff_payload
        self._remote_engagement: dict[str, str] = {}  # track_id → auth

        # Subscribe to inbound topics
        self._broker.subscribe("tracks", self._on_track_report)
        self._broker.subscribe("iff", self._on_iff_report)
        self._broker.subscribe("engagement", self._on_engagement_status)

    @property
    def node(self) -> NetworkNode:
        return self._node

    # --- Publishing (local → network) ---

    def publish_tracks(self, tracks: list, current_time: float) -> int:
        """Publish local fused tracks to the network.

        Accepts EnhancedFusedTrack or any object with position_m, velocity_mps, etc.
        Returns number of messages published.
        """
        count = 0
        for track in tracks:
            msg = self._track_to_message(track, current_time)
            if msg is not None:
                raw = msg.serialize()
                self._transport.broadcast_sync(raw)
                self._broker.publish("tracks", msg)
                count += 1
        return count

    def publish_iff(self, iff_results: dict, current_time: float) -> int:
        """Publish IFF results to the network."""
        count = 0
        for target_id, result in iff_results.items():
            identification = (
                result.identification.value
                if hasattr(result.identification, "value")
                else str(result.identification)
            )
            msg = make_iff_report(
                source_node=self._node.node_id,
                timestamp=current_time,
                target_id=target_id,
                identification=identification,
                confidence=result.confidence,
                mode_3a_code=getattr(result, "mode_3a_code", None),
                mode_s_address=getattr(result, "mode_s_address", None),
                spoof_suspect=getattr(result, "spoof_indicators", 0) > 0,
            )
            raw = msg.serialize()
            self._transport.broadcast_sync(raw)
            self._broker.publish("iff", msg)
            count += 1
        return count

    def publish_engagement(
        self, track_id: str, auth: str, current_time: float, reason: str = ""
    ) -> None:
        """Publish an engagement authorization change."""
        auth_value = auth.value if hasattr(auth, "value") else str(auth)
        msg = make_engagement_status(
            source_node=self._node.node_id,
            timestamp=current_time,
            track_id=track_id,
            engagement_auth=auth_value,
            reason=reason,
        )
        raw = msg.serialize()
        self._transport.broadcast_sync(raw)
        self._broker.publish("engagement", msg)

    # --- Receiving (network → local) ---

    def process_incoming(self) -> None:
        """Process all pending messages from transport and route to broker."""
        messages = self._transport.recv_all_sync()
        for source, raw in messages:
            try:
                msg = NetworkMessage.deserialize(raw)
            except Exception:
                logger.debug("Failed to deserialize message from %s", source)
                continue

            # Route to appropriate topic based on message type
            topic = self._msg_type_to_topic(msg.msg_type)
            if topic:
                self._broker.publish(topic, msg)

    def get_remote_tracks(self) -> dict[str, list[RemoteTrack]]:
        """Get buffered remote tracks from all peer nodes."""
        return dict(self._remote_tracks)

    def get_remote_iff(self) -> dict[str, dict]:
        """Get buffered remote IFF results."""
        return dict(self._remote_iff)

    def get_remote_engagement(self) -> dict[str, str]:
        """Get buffered remote engagement authorizations."""
        return dict(self._remote_engagement)

    def clear_buffers(self) -> None:
        """Clear all remote data buffers."""
        self._remote_tracks.clear()
        self._remote_iff.clear()
        self._remote_engagement.clear()

    # --- Internal callbacks ---

    def _on_track_report(self, msg: NetworkMessage) -> None:
        """Handle incoming track report — buffer as RemoteTrack."""
        if msg.source_node == self._node.node_id:
            return  # Skip own messages

        payload = msg.payload
        position = np.array(payload.get("position", [0, 0]))
        velocity = np.array(payload.get("velocity", [0, 0]))
        cov = payload.get("covariance")
        if cov is not None and not isinstance(cov, np.ndarray):
            cov = np.array(cov)
        geo = payload.get("position_geo")
        if geo is not None:
            geo = tuple(geo)

        remote_track = RemoteTrack(
            track_id=payload.get("track_id", "unknown"),
            source_node=msg.source_node,
            position=position,
            velocity=velocity,
            covariance=cov,
            position_geo=geo,
            sensor_types=payload.get("sensor_types", []),
            threat_level=payload.get("threat_level", "UNKNOWN"),
            iff_identification=payload.get("iff_identification", "unknown"),
            engagement_auth=payload.get("engagement_auth", "weapons_hold"),
            confidence=payload.get("confidence", 0.0),
            update_time=payload.get("update_time", msg.timestamp),
        )

        if msg.source_node not in self._remote_tracks:
            self._remote_tracks[msg.source_node] = []
        # Replace existing track with same ID or append
        tracks = self._remote_tracks[msg.source_node]
        for i, t in enumerate(tracks):
            if t.track_id == remote_track.track_id:
                tracks[i] = remote_track
                return
        tracks.append(remote_track)

    def _on_iff_report(self, msg: NetworkMessage) -> None:
        """Handle incoming IFF report."""
        if msg.source_node == self._node.node_id:
            return
        payload = msg.payload
        target_id = payload.get("target_id", "")
        self._remote_iff[target_id] = payload

    def _on_engagement_status(self, msg: NetworkMessage) -> None:
        """Handle incoming engagement status."""
        if msg.source_node == self._node.node_id:
            return
        payload = msg.payload
        track_id = payload.get("track_id", "")
        self._remote_engagement[track_id] = payload.get(
            "engagement_auth", "weapons_hold"
        )

    # --- Helpers ---

    def _track_to_message(self, track: Any, current_time: float) -> NetworkMessage | None:
        """Convert a fused track to a network message."""
        # Extract position from various track types
        position = getattr(track, "position_m", None)
        if position is None:
            position = getattr(track, "position", None)
        if position is None:
            return None

        velocity_mps = getattr(track, "velocity_mps", 0.0)
        velocity = getattr(track, "velocity", None)
        if velocity is None or (hasattr(velocity, "__len__") and len(velocity) == 0):
            velocity = np.array([velocity_mps, 0.0])

        # Covariance for composite fusion
        covariance = getattr(track, "fused_covariance", None)

        # Geo position
        geo = getattr(track, "position_geo", None)
        if isinstance(geo, dict):
            geo = (geo.get("lat", 0), geo.get("lon", 0), geo.get("alt", 0))

        # Sensor types
        sensor_sources = getattr(track, "sensor_sources", set())
        sensor_types = [s.value if hasattr(s, "value") else str(s) for s in sensor_sources]

        # Prefix track ID with node ID for deconfliction
        fused_id = getattr(track, "fused_id", "unknown")
        track_id = f"{self._node.node_id}:{fused_id}"

        return make_track_report(
            source_node=self._node.node_id,
            timestamp=current_time,
            track_id=track_id,
            position=position,
            velocity=velocity,
            covariance=covariance,
            position_geo=geo,
            sensor_types=sensor_types,
            threat_level=getattr(track, "threat_level", "UNKNOWN"),
            iff_identification=getattr(track, "iff_identification", "unknown"),
            engagement_auth=getattr(track, "engagement_auth", "weapons_hold"),
            confidence=getattr(track, "confidence", 0.0),
            update_time=current_time,
        )

    @staticmethod
    def _msg_type_to_topic(msg_type: MessageType) -> str | None:
        """Map message type to pub/sub topic."""
        mapping = {
            MessageType.TRACK_REPORT: "tracks",
            MessageType.DETECTION_REPORT: "detections",
            MessageType.IFF_REPORT: "iff",
            MessageType.ENGAGEMENT_STATUS: "engagement",
            MessageType.HEARTBEAT: "heartbeat",
            MessageType.SENSOR_STATUS: "sensor_status",
        }
        return mapping.get(msg_type)
