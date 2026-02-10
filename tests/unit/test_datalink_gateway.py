"""Tests for DataLinkGateway — the main Link 16 interoperability hub."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import L16Identity
from sentinel.datalink.config import DataLinkConfig
from sentinel.datalink.encoding import J2_2Codec, J3_2Codec, J3_5Codec, J7_0Codec
from sentinel.datalink.gateway import DataLinkGateway
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.transport import InMemoryDataLinkTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTrack:
    fused_id: str = "abcd0001"
    track_id: str = ""
    position_geo: Any = (40.0, -74.0, 3000.0)
    velocity: Any = None
    iff_identification: str = "unknown"
    sensor_count: int = 2
    threat_level: str = "LOW"
    confidence: float | None = None
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False
    is_decoy_candidate: bool = False
    is_chaff_candidate: bool = False


def _default_config(**overrides) -> DataLinkConfig:
    d = {
        "enabled": True,
        "publish_rate_hz": 1000.0,  # high rate → no rate limiting in tests
        "validate_outbound": True,
        "validate_inbound": True,
        "accept_inbound": True,
        "publish_iff": True,
        "publish_engagement": True,
    }
    d.update(overrides)
    return DataLinkConfig(**d)


# ===========================================================================
# Outbound: publish_tracks
# ===========================================================================


class TestPublishTracks:
    def test_publish_single_track(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        tracks = [FakeTrack()]
        sent = gw.publish_tracks(tracks, 1000.0)
        assert sent == 1
        stats = gw.get_stats()
        assert stats["messages_sent"] == 1
        assert stats["tracks_published"] == 1

    def test_publish_multiple_tracks(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        tracks = [FakeTrack(fused_id=f"trk-{i:04d}") for i in range(5)]
        sent = gw.publish_tracks(tracks, 1000.0)
        assert sent == 5

    def test_track_without_position_skipped(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        track = FakeTrack(position_geo=None)
        sent = gw.publish_tracks([track], 1000.0)
        assert sent == 0
        assert gw.get_stats()["tracks_skipped"] == 1

    def test_rate_limiting(self):
        cfg = _default_config(publish_rate_hz=0.5)  # 1 msg per 2 sec
        gw = DataLinkGateway(cfg)
        tracks = [FakeTrack()]

        sent1 = gw.publish_tracks(tracks, 100.0)
        assert sent1 == 1

        # Immediate second call → rate-limited
        sent2 = gw.publish_tracks(tracks, 100.1)
        assert sent2 == 0

    def test_messages_received_by_peer(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("gw1")
        t2 = InMemoryDataLinkTransport("gw2")
        t1.connect_peer(t2)

        gw = DataLinkGateway(cfg, transport=t1)
        gw.publish_tracks([FakeTrack()], 1000.0)

        raw = t2.recv_all()
        assert len(raw) == 1


# ===========================================================================
# Outbound: publish_iff
# ===========================================================================


class TestPublishIFF:
    def test_publish_iff(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        iff = {
            "trk-001": {
                "identification": "friendly",
                "mode_3a_code": "1200",
            }
        }
        sent = gw.publish_iff(iff, 1000.0)
        assert sent == 1
        assert gw.get_stats()["iff_published"] == 1

    def test_iff_disabled(self):
        cfg = _default_config(publish_iff=False)
        gw = DataLinkGateway(cfg)
        sent = gw.publish_iff({"trk-001": {}}, 1000.0)
        assert sent == 0


# ===========================================================================
# Outbound: publish_engagement
# ===========================================================================


class TestPublishEngagement:
    def test_publish_engagement(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        gw.publish_engagement("trk-001", "weapons_free", 1000.0, weapon_type=2)
        assert gw.get_stats()["engagement_published"] == 1

    def test_engagement_disabled(self):
        cfg = _default_config(publish_engagement=False)
        gw = DataLinkGateway(cfg)
        gw.publish_engagement("trk-001", "weapons_hold", 1000.0)
        assert gw.get_stats()["engagement_published"] == 0


# ===========================================================================
# Outbound: publish_track_drop
# ===========================================================================


class TestPublishTrackDrop:
    def test_track_drop_sends_j3_2(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("gw1")
        t2 = InMemoryDataLinkTransport("gw2")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t1)

        # Allocate a track first
        gw.publish_tracks([FakeTrack(fused_id="drop-me")], 1000.0)
        gw.publish_track_drop("drop-me", 1001.0)

        raw = t2.recv_all()
        assert len(raw) == 2  # J2.2 + J3.2

    def test_track_drop_releases_number(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        gw.publish_tracks([FakeTrack(fused_id="release-me")], 1000.0)
        assert gw.allocator.active_count == 1
        gw.publish_track_drop("release-me", 1001.0)
        assert gw.allocator.active_count == 0


# ===========================================================================
# Inbound: process_incoming
# ===========================================================================


class TestProcessIncoming:
    def test_receive_j2_2(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        msg = J2_2AirTrack(
            track_number=42,
            identity=L16Identity.HOSTILE,
            latitude_deg=38.0,
            longitude_deg=-77.0,
            altitude_ft=20000,
            speed_knots=400,
            course_deg=270.0,
            track_quality=5,
            threat_level=2,
        )
        t1.send(J2_2Codec.encode(msg))

        processed = gw.process_incoming()
        assert processed == 1
        tracks = gw.get_received_tracks()
        assert len(tracks) == 1
        assert tracks[0]["iff_identification"] == "hostile"

    def test_receive_j7_0(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        msg = J7_0IFF(
            track_number=10,
            identity=L16Identity.FRIEND,
            mode_3a=0o1200,
            mode_4_valid=True,
        )
        t1.send(J7_0Codec.encode(msg))

        processed = gw.process_incoming()
        assert processed == 1
        iff = gw.get_received_iff()
        assert 10 in iff
        assert iff[10]["identification"] == "friendly"

    def test_receive_j3_5(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        msg = J3_5EngagementStatus(
            track_number=99,
            engagement_auth=0,
            weapon_type=2,
            engagement_status=1,
        )
        t1.send(J3_5Codec.encode(msg))

        processed = gw.process_incoming()
        assert processed == 1
        eng = gw.get_received_engagement()
        assert 99 in eng

    def test_receive_j3_2_drop(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        # First send a J2.2 so there's a track to drop
        air = J2_2AirTrack(
            track_number=55,
            identity=L16Identity.UNKNOWN,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_ft=0,
        )
        t1.send(J2_2Codec.encode(air))
        gw.process_incoming()
        assert len(gw.get_received_tracks()) == 1

        # Now send a J3.2 drop
        drop = J3_2TrackManagement(
            track_number=55,
            identity=L16Identity.UNKNOWN,
            action=1,
        )
        t1.send(J3_2Codec.encode(drop))
        gw.process_incoming()
        assert len(gw.get_received_tracks()) == 0

    def test_inbound_disabled(self):
        cfg = _default_config(accept_inbound=False)
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        msg = J2_2AirTrack(track_number=1, identity=L16Identity.UNKNOWN,
                           latitude_deg=0.0, longitude_deg=0.0, altitude_ft=0)
        t1.send(J2_2Codec.encode(msg))

        processed = gw.process_incoming()
        assert processed == 0

    def test_decode_error_counted(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        # Send garbage
        t1.send(b"\xff\xff\xff")
        gw.process_incoming()
        assert gw.get_stats()["decode_errors"] == 1

    def test_clear_buffers(self):
        cfg = _default_config()
        t1 = InMemoryDataLinkTransport("sender")
        t2 = InMemoryDataLinkTransport("receiver")
        t1.connect_peer(t2)
        gw = DataLinkGateway(cfg, transport=t2)

        msg = J2_2AirTrack(track_number=1, identity=L16Identity.UNKNOWN,
                           latitude_deg=0.0, longitude_deg=0.0, altitude_ft=0)
        t1.send(J2_2Codec.encode(msg))
        gw.process_incoming()
        assert len(gw.get_received_tracks()) == 1

        gw.clear_buffers()
        assert len(gw.get_received_tracks()) == 0


# ===========================================================================
# Factory: from_config
# ===========================================================================


class TestFromConfig:
    def test_disabled_returns_none(self):
        cfg = OmegaConf.create({"sentinel": {"datalink": {"enabled": False}}})
        gw = DataLinkGateway.from_config(cfg.sentinel.datalink)
        assert gw is None

    def test_enabled_returns_gateway(self):
        cfg = OmegaConf.create({"sentinel": {"datalink": {"enabled": True}}})
        gw = DataLinkGateway.from_config(cfg.sentinel.datalink)
        assert gw is not None
        assert isinstance(gw, DataLinkGateway)

    def test_custom_max_track_numbers(self):
        cfg = OmegaConf.create({"sentinel": {"datalink": {
            "enabled": True,
            "max_track_numbers": 256,
        }}})
        gw = DataLinkGateway.from_config(cfg.sentinel.datalink)
        assert gw is not None


# ===========================================================================
# Properties
# ===========================================================================


class TestGatewayProperties:
    def test_transport_property(self):
        cfg = _default_config()
        t = InMemoryDataLinkTransport("test")
        gw = DataLinkGateway(cfg, transport=t)
        assert gw.transport is t

    def test_allocator_property(self):
        cfg = _default_config()
        gw = DataLinkGateway(cfg)
        assert gw.allocator is not None
        assert gw.allocator.active_count == 0
