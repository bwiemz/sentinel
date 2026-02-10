"""Tests for DataLinkConfig and GatewayStats."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from sentinel.datalink.config import DataLinkConfig
from sentinel.datalink.stats import GatewayStats


# ===========================================================================
# DataLinkConfig
# ===========================================================================


class TestDataLinkConfigDefaults:
    def test_defaults(self):
        cfg = DataLinkConfig()
        assert cfg.enabled is False
        assert cfg.source_id == "SENTINEL-01"
        assert cfg.transport_type == "in_memory"
        assert cfg.publish_rate_hz == 1.0
        assert cfg.publish_iff is True
        assert cfg.publish_engagement is True
        assert cfg.validate_outbound is True
        assert cfg.validate_inbound is True
        assert cfg.max_track_numbers == 8192
        assert cfg.accept_inbound is True
        assert cfg.merge_with_local is False


class TestDataLinkConfigFromOmegaconf:
    def test_from_none(self):
        cfg = DataLinkConfig.from_omegaconf(None)
        assert cfg.enabled is False

    def test_from_empty_dict(self):
        cfg = DataLinkConfig.from_omegaconf({})
        assert cfg.enabled is False
        assert cfg.source_id == "SENTINEL-01"

    def test_from_plain_dict(self):
        raw = {
            "enabled": True,
            "source_id": "NODE-42",
            "publish_rate_hz": 5.0,
            "max_track_numbers": 1024,
        }
        cfg = DataLinkConfig.from_omegaconf(raw)
        assert cfg.enabled is True
        assert cfg.source_id == "NODE-42"
        assert cfg.publish_rate_hz == 5.0
        assert cfg.max_track_numbers == 1024

    def test_from_omegaconf_dictconfig(self):
        raw = OmegaConf.create({
            "enabled": True,
            "source_id": "AWACS-01",
            "transport_type": "udp",
            "publish_rate_hz": 10.0,
            "publish_iff": False,
            "validate_outbound": False,
            "accept_inbound": False,
            "merge_with_local": True,
        })
        cfg = DataLinkConfig.from_omegaconf(raw)
        assert cfg.enabled is True
        assert cfg.source_id == "AWACS-01"
        assert cfg.transport_type == "udp"
        assert cfg.publish_rate_hz == 10.0
        assert cfg.publish_iff is False
        assert cfg.validate_outbound is False
        assert cfg.accept_inbound is False
        assert cfg.merge_with_local is True

    def test_partial_override(self):
        raw = OmegaConf.create({"enabled": True})
        cfg = DataLinkConfig.from_omegaconf(raw)
        assert cfg.enabled is True
        # Everything else is default
        assert cfg.publish_rate_hz == 1.0
        assert cfg.max_track_numbers == 8192


# ===========================================================================
# GatewayStats
# ===========================================================================


class TestGatewayStats:
    def test_defaults_all_zero(self):
        s = GatewayStats()
        d = s.to_dict()
        assert all(v == 0 for v in d.values())
        assert len(d) == 12

    def test_increment_and_to_dict(self):
        s = GatewayStats()
        s.messages_sent = 10
        s.tracks_published = 5
        d = s.to_dict()
        assert d["messages_sent"] == 10
        assert d["tracks_published"] == 5
        assert d["messages_received"] == 0

    def test_reset(self):
        s = GatewayStats()
        s.messages_sent = 100
        s.encode_errors = 3
        s.reset()
        assert s.messages_sent == 0
        assert s.encode_errors == 0
        assert all(v == 0 for v in s.to_dict().values())

    def test_to_dict_keys(self):
        s = GatewayStats()
        keys = set(s.to_dict().keys())
        expected = {
            "messages_sent", "messages_received", "messages_invalid",
            "encode_errors", "decode_errors",
            "tracks_published", "tracks_received", "tracks_skipped",
            "iff_published", "iff_received",
            "engagement_published", "engagement_received",
        }
        assert keys == expected
