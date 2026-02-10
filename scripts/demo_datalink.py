#!/usr/bin/env python3
"""Demo: Link 16 Data Link Gateway — two SENTINEL nodes exchanging J-series messages.

Simulates two SENTINEL gateway nodes connected via in-memory transport.
Node Alpha publishes tracks and IFF; Node Bravo receives, decodes, and
publishes engagement responses back.

Usage:
    python scripts/demo_datalink.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Data Link Gateway Demo")
    parser.add_argument("--verbose", action="store_true", help="Show detailed message info")
    args = parser.parse_args()

    # Late imports to verify the package loads cleanly
    from sentinel.core.types import L16Identity, L16MessageType
    from sentinel.datalink.config import DataLinkConfig
    from sentinel.datalink.encoding import J2_2Codec, J7_0Codec, decode_message
    from sentinel.datalink.gateway import DataLinkGateway
    from sentinel.datalink.j_series import J2_2AirTrack
    from sentinel.datalink.transport import InMemoryDataLinkTransport
    from sentinel.datalink.validator import L16Validator

    print("=" * 70)
    print("  SENTINEL — Link 16 Data Link Gateway Demo")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # 1. Create two linked gateways
    # ---------------------------------------------------------------

    cfg = DataLinkConfig(
        enabled=True,
        publish_rate_hz=100.0,
        validate_outbound=True,
        validate_inbound=True,
        accept_inbound=True,
        publish_iff=True,
        publish_engagement=True,
    )

    t_alpha = InMemoryDataLinkTransport("ALPHA")
    t_bravo = InMemoryDataLinkTransport("BRAVO")
    t_alpha.connect_peer(t_bravo)

    gw_alpha = DataLinkGateway(cfg, transport=t_alpha)
    gw_bravo = DataLinkGateway(cfg, transport=t_bravo)

    print("[1] Created two gateway nodes: ALPHA <-> BRAVO")
    print(f"    Transport: InMemory bidirectional")
    print()

    # ---------------------------------------------------------------
    # 2. Create simulated tracks on Node Alpha
    # ---------------------------------------------------------------

    class FakeTrack:
        def __init__(self, fused_id, lat, lon, alt_m, vx, vy, iff, threat, squawk=None):
            self.fused_id = fused_id
            self.position_geo = (lat, lon, alt_m)
            self.velocity = np.array([vx, vy])
            self.iff_identification = iff
            self.iff_mode_3a_code = squawk
            self.sensor_count = 3
            self.confidence = 0.85
            self.threat_level = threat
            self.is_stealth_candidate = False
            self.is_hypersonic_candidate = False
            self.is_decoy_candidate = False
            self.is_chaff_candidate = False

    tracks = [
        FakeTrack("HAWK-001", 38.897, -77.036, 9144.0, 150.0, 50.0, "friendly", "LOW", "1200"),
        FakeTrack("BOGEY-002", 39.100, -76.800, 12000.0, -200.0, 30.0, "unknown", "MEDIUM"),
        FakeTrack("BANDIT-003", 38.500, -77.500, 3000.0, 300.0, -100.0, "hostile", "CRITICAL"),
        FakeTrack("STEALTH-004", 39.500, -76.000, 15000.0, -250.0, 0.0, "assumed_hostile", "HIGH"),
        FakeTrack("NEUTRAL-005", 38.000, -78.000, 5000.0, 0.0, 80.0, "pending", "LOW"),
    ]

    print(f"[2] Created {len(tracks)} simulated tracks:")
    for t in tracks:
        print(f"    {t.fused_id:15s} | {t.iff_identification:15s} | {t.threat_level:8s} | "
              f"({t.position_geo[0]:.3f}, {t.position_geo[1]:.3f}, {t.position_geo[2]:.0f}m)")
    print()

    # ---------------------------------------------------------------
    # 3. Alpha publishes tracks (J2.2)
    # ---------------------------------------------------------------

    sent = gw_alpha.publish_tracks(tracks, time.time())
    print(f"[3] ALPHA published {sent} J2.2 Air Track messages")

    if args.verbose:
        # Peek at encoded message sizes
        for t in tracks:
            msg = gw_alpha._adapter.track_to_j2_2(t, time.time())
            if msg:
                data = J2_2Codec.encode(msg)
                errors = L16Validator.validate_j2_2(msg)
                print(f"    {msg.source_sentinel_id}: TN={msg.track_number}, "
                      f"{msg.identity.value}, {len(data)} bytes, "
                      f"{'VALID' if not errors else f'{len(errors)} errors'}")
    print()

    # ---------------------------------------------------------------
    # 4. Alpha publishes IFF (J7.0)
    # ---------------------------------------------------------------

    iff_results = {
        "HAWK-001": {
            "identification": "friendly",
            "mode_3a_code": "1200",
            "mode_s_address": "A1B2C3",
            "mode_4_valid": True,
            "mode_5_valid": False,
        },
        "BANDIT-003": {
            "identification": "hostile",
            "mode_4_valid": False,
            "mode_5_valid": False,
        },
    }

    iff_sent = gw_alpha.publish_iff(iff_results, time.time())
    print(f"[4] ALPHA published {iff_sent} J7.0 IFF messages")
    print()

    # ---------------------------------------------------------------
    # 5. Alpha publishes engagement (J3.5)
    # ---------------------------------------------------------------

    gw_alpha.publish_engagement("BANDIT-003", "weapons_free", time.time(), weapon_type=2, engagement_status=1)
    print("[5] ALPHA published J3.5 engagement for BANDIT-003 (weapons_free)")
    print()

    # ---------------------------------------------------------------
    # 6. Bravo processes all incoming messages
    # ---------------------------------------------------------------

    processed = gw_bravo.process_incoming()
    print(f"[6] BRAVO processed {processed} inbound messages")

    received_tracks = gw_bravo.get_received_tracks()
    received_iff = gw_bravo.get_received_iff()
    received_eng = gw_bravo.get_received_engagement()

    print(f"    Tracks received: {len(received_tracks)}")
    print(f"    IFF received:    {len(received_iff)}")
    print(f"    Engagement:      {len(received_eng)}")

    if args.verbose:
        for tr in received_tracks:
            print(f"    Track: {tr['track_id']:15s} | {tr['iff_identification']:15s} | "
                  f"threat={tr['threat_level']:8s} | "
                  f"conf={tr['confidence']:.2f}")
    print()

    # ---------------------------------------------------------------
    # 7. Bravo responds with track drop (J3.2)
    # ---------------------------------------------------------------

    gw_bravo.publish_track_drop("NEUTRAL-005", time.time())
    gw_alpha.process_incoming()
    print("[7] BRAVO sent track drop for NEUTRAL-005")
    print(f"    ALPHA processed drop -> "
          f"received tracks still buffered: {len(gw_alpha.get_received_tracks())}")
    print()

    # ---------------------------------------------------------------
    # 8. Gateway statistics
    # ---------------------------------------------------------------

    print("[8] Gateway Statistics:")
    for name, gw in [("ALPHA", gw_alpha), ("BRAVO", gw_bravo)]:
        stats = gw.get_stats()
        print(f"    {name}:")
        print(f"      messages_sent={stats['messages_sent']}, "
              f"messages_received={stats['messages_received']}")
        print(f"      tracks_published={stats['tracks_published']}, "
              f"tracks_received={stats['tracks_received']}")
        print(f"      iff_published={stats['iff_published']}, "
              f"iff_received={stats['iff_received']}")
        print(f"      engagement_published={stats['engagement_published']}, "
              f"engagement_received={stats['engagement_received']}")
        print(f"      errors: encode={stats['encode_errors']}, "
              f"decode={stats['decode_errors']}, invalid={stats['messages_invalid']}")
    print()

    # ---------------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------------

    total_sent = gw_alpha.get_stats()["messages_sent"] + gw_bravo.get_stats()["messages_sent"]
    total_recv = gw_alpha.get_stats()["messages_received"] + gw_bravo.get_stats()["messages_received"]

    print("=" * 70)
    print(f"  Demo complete: {total_sent} messages sent, {total_recv} received")
    print(f"  Message types: J2.2 Air Track, J3.2 Track Mgmt, J3.5 Engagement, J7.0 IFF")
    print(f"  Transport: In-memory bidirectional (zero errors)")
    print("=" * 70)


if __name__ == "__main__":
    main()
