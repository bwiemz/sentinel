"""SENTINEL Data Link / STANAG 5516 (Link 16) gateway.

Provides bidirectional conversion between SENTINEL track types and
Link 16 J-series binary messages for external C2 interoperability.
"""

from sentinel.datalink.adapter import DataLinkAdapter
from sentinel.datalink.codec import BitReader, BitWriter
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
from sentinel.datalink.gateway import DataLinkGateway
from sentinel.datalink.j_series import (
    J2_2AirTrack,
    J3_2TrackManagement,
    J3_5EngagementStatus,
    J7_0IFF,
)
from sentinel.datalink.stats import GatewayStats
from sentinel.datalink.track_mapping import TrackNumberAllocator
from sentinel.datalink.transport import DataLinkTransport, InMemoryDataLinkTransport
from sentinel.datalink.validator import L16Validator, ValidationResult

__all__ = [
    "BitReader",
    "BitWriter",
    "DataLinkAdapter",
    "DataLinkConfig",
    "DataLinkGateway",
    "DataLinkTransport",
    "GatewayStats",
    "InMemoryDataLinkTransport",
    "J2_2AirTrack",
    "J2_2Codec",
    "J3_2Codec",
    "J3_2TrackManagement",
    "J3_5Codec",
    "J3_5EngagementStatus",
    "J7_0Codec",
    "J7_0IFF",
    "L16Validator",
    "TrackNumberAllocator",
    "ValidationResult",
    "decode_message",
    "encode_message",
    "peek_message_type",
]
