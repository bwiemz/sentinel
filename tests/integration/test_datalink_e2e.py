"""Integration tests for the Data Link / STANAG 5516 gateway.

Tests end-to-end workflows: two-gateway bidirectional communication,
full encode/decode roundtrips across transport, config loading from
OmegaConf, and pipeline status reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import L16Identity, L16MessageType
from sentinel.datalink.adapter import DataLinkAdapter
from sentinel.datalink.config import DataLinkConfig
from sentinel.datalink.encoding import (
    J2_2Codec,
    J3_2Codec,
    J3_5Codec,
    J7_0Codec,
    decode_message,
    encode_message,
)
from sentinel.datalink.gateway import DataLinkGateway
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.track_mapping import TrackNumberAllocator
from sentinel.datalink.transport import InMemoryDataLinkTransport
from sentinel.datalink.validator import L16Validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTrack:
    fused_id: str = "e2e-0001"
    track_id: str = ""
    position_geo: Any = (38.897, -77.036, 3000.0)
    velocity: Any = None
    iff_identification: str = "friendly"
    iff_mode_3a_code: str | None = "1200"
    sensor_count: int = 3
    confidence: float | None = 0.85
    threat_level: str = "MEDIUM"
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False
    is_decoy_candidate: bool = False
    is_chaff_candidate: bool = False


def _make_gateway_pair():
    """Create two linked gateways for bidirectional testing."""
    cfg = DataLinkConfig(
        enabled=True,
        publish_rate_hz=1000.0,
        validate_outbound=True,
        validate_inbound=True,
        accept_inbound=True,
        publish_iff=True,
        publish_engagement=True,
    )
    t1 = InMemoryDataLinkTransport("gw-alpha")
    t2 = InMemoryDataLinkTransport("gw-bravo")
    t1.connect_peer(t2)

    gw1 = DataLinkGateway(cfg, transport=t1)
    gw2 = DataLinkGateway(cfg, transport=t2)
    return gw1, gw2


# ===========================================================================
# Two-gateway bidirectional communication
# ===========================================================================


class TestTwoGatewayBidirectional:
    def test_publish_and_receive_tracks(self):
        """Gateway 1 publishes tracks, Gateway 2 receives them."""
        gw1, gw2 = _make_gateway_pair()

        tracks = [
            FakeTrack(fused_id="alpha-001", position_geo=(38.9, -77.0, 5000.0)),
            FakeTrack(fused_id="alpha-002", position_geo=(39.0, -76.5, 8000.0)),
        ]
        sent = gw1.publish_tracks(tracks, 1000.0)
        assert sent == 2

        processed = gw2.process_incoming()
        assert processed == 2

        received = gw2.get_received_tracks()
        assert len(received) == 2

    def test_publish_and_receive_iff(self):
        gw1, gw2 = _make_gateway_pair()

        iff = {
            "trk-a": {
                "identification": "friendly",
                "mode_3a_code": "1200",
                "mode_4_valid": True,
            }
        }
        gw1.publish_iff(iff, 1000.0)
        gw2.process_incoming()

        received_iff = gw2.get_received_iff()
        assert len(received_iff) == 1

    def test_publish_and_receive_engagement(self):
        gw1, gw2 = _make_gateway_pair()

        gw1.publish_engagement("trk-b", "weapons_free", 1000.0, weapon_type=3)
        gw2.process_incoming()

        received_eng = gw2.get_received_engagement()
        assert len(received_eng) == 1

    def test_bidirectional_exchange(self):
        """Both gateways publish and receive."""
        gw1, gw2 = _make_gateway_pair()

        # Gateway 1 publishes
        gw1.publish_tracks(
            [FakeTrack(fused_id="g1-001", position_geo=(40.0, -74.0, 1000.0))],
            1000.0,
        )

        # Gateway 2 publishes
        gw2.publish_tracks(
            [FakeTrack(fused_id="g2-001", position_geo=(41.0, -73.0, 2000.0))],
            1000.0,
        )

        # Both process
        gw1.process_incoming()
        gw2.process_incoming()

        assert len(gw1.get_received_tracks()) == 1
        assert len(gw2.get_received_tracks()) == 1


# ===========================================================================
# Full encode/decode roundtrip across transport
# ===========================================================================


class TestFullRoundtripAcrossTransport:
    def test_j2_2_position_precision(self):
        """Verify lat/lon survive encode->transport->decode within precision."""
        gw1, gw2 = _make_gateway_pair()

        lat_in, lon_in = 51.4775, -0.0015  # London
        track = FakeTrack(
            fused_id="london-01",
            position_geo=(lat_in, lon_in, 300.0),
            velocity=np.array([50.0, 100.0]),
        )
        gw1.publish_tracks([track], 1000.0)
        gw2.process_incoming()

        received = gw2.get_received_tracks()
        assert len(received) == 1

        pos = received[0]["position_geo"]
        assert abs(pos[0] - lat_in) < 0.005  # within ~500m (14-bit encoding)
        assert abs(pos[1] - lon_in) < 0.005

    def test_j2_2_identity_preserved(self):
        gw1, gw2 = _make_gateway_pair()

        track = FakeTrack(
            fused_id="hostile-01",
            position_geo=(0.0, 0.0, 0.0),
            iff_identification="hostile",
        )
        gw1.publish_tracks([track], 1000.0)
        gw2.process_incoming()

        received = gw2.get_received_tracks()
        assert received[0]["iff_identification"] == "hostile"

    def test_j7_0_mode_codes_roundtrip(self):
        gw1, gw2 = _make_gateway_pair()

        iff = {
            "trk-iff": {
                "identification": "friendly",
                "mode_3a_code": "7700",  # Emergency
                "mode_s_address": "A1B2C3",
                "mode_4_valid": True,
                "mode_5_valid": False,
            }
        }
        gw1.publish_iff(iff, 500.0)
        gw2.process_incoming()

        received = gw2.get_received_iff()
        assert len(received) == 1
        result = list(received.values())[0]
        assert result["identification"] == "friendly"
        assert result["mode_4_valid"] is True

    def test_track_drop_removes_from_receiver(self):
        gw1, gw2 = _make_gateway_pair()

        track = FakeTrack(fused_id="drop-target", position_geo=(0.0, 0.0, 0.0))
        gw1.publish_tracks([track], 1000.0)
        gw2.process_incoming()
        assert len(gw2.get_received_tracks()) == 1

        gw1.publish_track_drop("drop-target", 1001.0)
        gw2.process_incoming()
        # J3.2 drop should remove from received buffer
        assert len(gw2.get_received_tracks()) == 0


# ===========================================================================
# OmegaConf config loading
# ===========================================================================


class TestOmegaconfConfig:
    def test_from_default_yaml(self):
        """Load the real default.yaml and verify datalink config."""
        import os

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "default.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("default.yaml not found")
        raw = OmegaConf.load(yaml_path)
        dl_cfg = raw.sentinel.get("datalink", {})
        cfg = DataLinkConfig.from_omegaconf(dl_cfg)
        assert cfg.enabled is False
        assert cfg.max_track_numbers == 8192
        assert cfg.source_id == "SENTINEL-01"

    def test_gateway_from_config_disabled(self):
        raw = OmegaConf.create({"enabled": False})
        gw = DataLinkGateway.from_config(raw)
        assert gw is None

    def test_gateway_from_config_enabled(self):
        raw = OmegaConf.create({
            "enabled": True,
            "publish_rate_hz": 5.0,
            "max_track_numbers": 256,
        })
        gw = DataLinkGateway.from_config(raw)
        assert gw is not None


# ===========================================================================
# Statistics across operations
# ===========================================================================


class TestStatsAcrossOperations:
    def test_stats_accumulate(self):
        gw1, gw2 = _make_gateway_pair()

        # Publish tracks
        tracks = [FakeTrack(fused_id=f"s-{i:03d}") for i in range(3)]
        gw1.publish_tracks(tracks, 1000.0)

        # Publish IFF
        gw1.publish_iff({"trk-x": {"identification": "friendly"}}, 1000.0)

        # Check stats
        stats = gw1.get_stats()
        assert stats["messages_sent"] == 4  # 3 tracks + 1 IFF
        assert stats["tracks_published"] == 3
        assert stats["iff_published"] == 1

    def test_decode_errors_tracked(self):
        gw1, gw2 = _make_gateway_pair()

        # Send garbage from gw1's transport directly
        gw1.transport.send(b"\x00\x00")
        gw2.process_incoming()

        stats = gw2.get_stats()
        assert stats["decode_errors"] == 1


# ===========================================================================
# Message type dispatch
# ===========================================================================


class TestMessageTypeDispatch:
    def test_all_message_types_decode(self):
        """Verify all 4 message types survive encode→decode dispatch."""
        messages = [
            J2_2AirTrack(track_number=1, identity=L16Identity.FRIEND,
                         latitude_deg=10.0, longitude_deg=20.0, altitude_ft=5000,
                         speed_knots=300, course_deg=45.0),
            J3_2TrackManagement(track_number=2, identity=L16Identity.HOSTILE, action=1),
            J3_5EngagementStatus(track_number=3, engagement_auth=0, weapon_type=1, engagement_status=1),
            J7_0IFF(track_number=4, identity=L16Identity.UNKNOWN,
                    mode_3a=0o1234, mode_s_address=0xABCD),
        ]

        for orig in messages:
            data = encode_message(orig)
            decoded = decode_message(data)
            assert type(decoded) == type(orig)

    def test_validator_accepts_valid_messages(self):
        """All well-formed messages pass validation."""
        messages = [
            J2_2AirTrack(track_number=100, identity=L16Identity.NEUTRAL,
                         latitude_deg=40.0, longitude_deg=-74.0, altitude_ft=10000),
            J3_2TrackManagement(track_number=200, identity=L16Identity.FRIEND, action=0),
            J3_5EngagementStatus(track_number=300, engagement_auth=1, weapon_type=2,
                                 engagement_status=0),
            J7_0IFF(track_number=400, identity=L16Identity.HOSTILE),
        ]
        for msg in messages:
            errors = L16Validator.validate(msg)
            assert len(errors) == 0, f"Validation failed for {type(msg).__name__}: {errors}"
