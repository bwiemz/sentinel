"""Unit tests for network message catalog and serialization."""

import numpy as np
import pytest

from sentinel.core.types import MessageType
from sentinel.network.messages import (
    NetworkMessage,
    make_track_report,
    make_detection_report,
    make_iff_report,
    make_engagement_status,
    make_heartbeat,
    make_sensor_status,
    _HAS_MSGPACK,
)


# ===================================================================
# NetworkMessage basics
# ===================================================================


class TestNetworkMessage:
    def test_create_default(self):
        msg = NetworkMessage(
            msg_type=MessageType.HEARTBEAT,
            source_node="NODE-A",
            timestamp=1000.0,
        )
        assert msg.msg_type == MessageType.HEARTBEAT
        assert msg.source_node == "NODE-A"
        assert msg.timestamp == 1000.0
        assert msg.priority == 0
        assert msg.ttl == 3
        assert msg.payload == {}
        assert len(msg.msg_id) == 12

    def test_unique_msg_ids(self):
        msgs = [
            NetworkMessage(MessageType.HEARTBEAT, "N", 0.0)
            for _ in range(100)
        ]
        ids = {m.msg_id for m in msgs}
        assert len(ids) == 100

    def test_decrement_ttl(self):
        msg = NetworkMessage(MessageType.HEARTBEAT, "N", 0.0, ttl=3)
        forwarded = msg.decrement_ttl()
        assert forwarded.ttl == 2
        assert msg.ttl == 3  # original unchanged

    def test_is_expired(self):
        msg = NetworkMessage(MessageType.HEARTBEAT, "N", 0.0, ttl=0)
        assert msg.is_expired is True
        msg2 = NetworkMessage(MessageType.HEARTBEAT, "N", 0.0, ttl=1)
        assert msg2.is_expired is False

    def test_size_bytes(self):
        msg = NetworkMessage(MessageType.HEARTBEAT, "N", 0.0)
        size = msg.size_bytes
        assert isinstance(size, int)
        assert size > 0


# ===================================================================
# Serialization round-trip
# ===================================================================


class TestSerialization:
    def test_roundtrip_simple(self):
        msg = NetworkMessage(
            msg_type=MessageType.TRACK_REPORT,
            source_node="RADAR-01",
            timestamp=1000.5,
            payload={"track_id": "R-001", "position": [100.0, 200.0]},
            priority=1,
            ttl=2,
            sequence_num=42,
        )
        raw = msg.serialize()
        restored = NetworkMessage.deserialize(raw)
        assert restored.msg_type == MessageType.TRACK_REPORT
        assert restored.source_node == "RADAR-01"
        assert restored.timestamp == 1000.5
        assert restored.payload["track_id"] == "R-001"
        assert restored.priority == 1
        assert restored.ttl == 2
        assert restored.sequence_num == 42

    def test_roundtrip_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        msg = NetworkMessage(
            msg_type=MessageType.DETECTION_REPORT,
            source_node="N",
            timestamp=0.0,
            payload={"position": arr},
        )
        raw = msg.serialize()
        restored = NetworkMessage.deserialize(raw)
        pos = restored.payload["position"]
        assert isinstance(pos, np.ndarray)
        np.testing.assert_array_almost_equal(pos, arr)

    def test_roundtrip_nested_numpy(self):
        cov = np.eye(4) * 100.0
        msg = NetworkMessage(
            msg_type=MessageType.TRACK_REPORT,
            source_node="N",
            timestamp=0.0,
            payload={"covariance": cov},
        )
        raw = msg.serialize()
        restored = NetworkMessage.deserialize(raw)
        rcov = restored.payload["covariance"]
        assert isinstance(rcov, np.ndarray)
        np.testing.assert_array_almost_equal(rcov, cov)

    def test_roundtrip_none_values(self):
        msg = NetworkMessage(
            msg_type=MessageType.TRACK_REPORT,
            source_node="N",
            timestamp=0.0,
            payload={"position_geo": None, "covariance": None},
        )
        raw = msg.serialize()
        restored = NetworkMessage.deserialize(raw)
        assert restored.payload["position_geo"] is None
        assert restored.payload["covariance"] is None

    def test_roundtrip_all_message_types(self):
        for mt in MessageType:
            msg = NetworkMessage(msg_type=mt, source_node="N", timestamp=0.0)
            raw = msg.serialize()
            restored = NetworkMessage.deserialize(raw)
            assert restored.msg_type == mt

    def test_serialized_size_compact(self):
        """Serialized messages should be reasonably compact."""
        msg = make_track_report(
            source_node="RADAR-01",
            timestamp=1000.0,
            track_id="R-001",
            position=[1000.0, 2000.0, 500.0],
            velocity=[-50.0, 10.0, 0.0],
            threat_level="HIGH",
        )
        size = msg.size_bytes
        # Should be well under 1 KB for a simple track report
        assert size < 1024


# ===================================================================
# Payload builders
# ===================================================================


class TestTrackReport:
    def test_basic_fields(self):
        msg = make_track_report(
            source_node="NODE-A",
            timestamp=1000.0,
            track_id="NODE-A:R-001",
            position=[1000.0, 2000.0, 500.0],
            velocity=[-50.0, 10.0, 0.0],
            threat_level="HIGH",
            iff_identification="hostile",
            confidence=0.85,
        )
        assert msg.msg_type == MessageType.TRACK_REPORT
        assert msg.priority == 1  # priority level
        assert msg.payload["track_id"] == "NODE-A:R-001"
        assert msg.payload["position"] == [1000.0, 2000.0, 500.0]
        assert msg.payload["velocity"] == [-50.0, 10.0, 0.0]
        assert msg.payload["threat_level"] == "HIGH"
        assert msg.payload["iff_identification"] == "hostile"
        assert msg.payload["confidence"] == 0.85

    def test_with_covariance(self):
        cov = np.eye(4) * 100.0
        msg = make_track_report(
            source_node="N",
            timestamp=0.0,
            track_id="T-001",
            position=np.array([1000.0, 2000.0]),
            velocity=np.array([-50.0, 10.0]),
            covariance=cov,
        )
        assert msg.payload["covariance"] is not None
        assert len(msg.payload["covariance"]) == 4

    def test_with_geo_position(self):
        msg = make_track_report(
            source_node="N",
            timestamp=0.0,
            track_id="T-001",
            position=[0, 0],
            velocity=[0, 0],
            position_geo=(38.9, -77.0, 1000.0),
        )
        assert msg.payload["position_geo"] == [38.9, -77.0, 1000.0]


class TestDetectionReport:
    def test_radar_detection(self):
        msg = make_detection_report(
            source_node="RADAR-01",
            timestamp=1000.0,
            sensor_type="radar",
            range_m=5000.0,
            azimuth_deg=45.0,
            velocity_mps=-300.0,
            rcs_dbsm=10.0,
        )
        assert msg.msg_type == MessageType.DETECTION_REPORT
        assert msg.priority == 0  # routine
        assert msg.payload["sensor_type"] == "radar"
        assert msg.payload["range_m"] == 5000.0


class TestIFFReport:
    def test_basic(self):
        msg = make_iff_report(
            source_node="N",
            timestamp=0.0,
            target_id="TGT-01",
            identification="friendly",
            confidence=0.95,
            mode_3a_code="1200",
            spoof_suspect=False,
        )
        assert msg.msg_type == MessageType.IFF_REPORT
        assert msg.priority == 2  # immediate
        assert msg.payload["identification"] == "friendly"
        assert msg.payload["spoof_suspect"] is False


class TestEngagementStatus:
    def test_basic(self):
        msg = make_engagement_status(
            source_node="CMD-01",
            timestamp=0.0,
            track_id="T-005",
            engagement_auth="weapons_free",
            reason="hostile confirmed",
        )
        assert msg.msg_type == MessageType.ENGAGEMENT_STATUS
        assert msg.priority == 3  # flash
        assert msg.payload["engagement_auth"] == "weapons_free"


class TestHeartbeat:
    def test_basic(self):
        msg = make_heartbeat(
            source_node="RADAR-01",
            timestamp=1000.0,
            state="active",
            capabilities=["radar", "iff"],
            track_count=5,
            uptime_s=3600.0,
        )
        assert msg.msg_type == MessageType.HEARTBEAT
        assert msg.payload["state"] == "active"
        assert "radar" in msg.payload["capabilities"]


class TestSensorStatus:
    def test_basic(self):
        msg = make_sensor_status(
            source_node="NODE-B",
            timestamp=0.0,
            sensor_type="thermal",
            operational=True,
            max_range_m=30000.0,
        )
        assert msg.msg_type == MessageType.SENSOR_STATUS
        assert msg.payload["sensor_type"] == "thermal"
        assert msg.payload["operational"] is True
