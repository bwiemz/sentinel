"""SENTINEL tactical mesh network demo — multi-node track sharing.

Simulates 2-3 distributed SENTINEL nodes sharing tracks over a tactical
mesh network (CEC-inspired composite tracking).  Each node runs its own
sensors and local fusion, then publishes fused tracks through the network
bridge.  The CompositeFusion layer merges remote tracks into a unified
air picture at each node.

Network degradation profiles (latency, jitter, packet loss) are applied
per-link via the SimulatedTransport / TransportHub layer.

Run:
    python scripts/demo_network.py
    python scripts/demo_network.py --nodes 3 --degradation ttnt --steps 30
    python scripts/demo_network.py --degradation severe --steps 20

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from sentinel.core.clock import SimClock
from sentinel.core.types import NodeRole, SensorType
from sentinel.fusion.multi_sensor_fusion import EnhancedFusedTrack
from sentinel.network.bridge import NetworkBridge
from sentinel.network.composite_fusion import CompositeFusion
from sentinel.network.discovery import MeshDiscovery
from sentinel.network.node import NetworkNode
from sentinel.network.pubsub import PubSubBroker
from sentinel.network.transport import (
    DEGRADATION_PRESETS,
    DegradationProfile,
    SimulatedTransport,
    TransportHub,
)


# ---------------------------------------------------------------------------
# Simulated sensor nodes
# ---------------------------------------------------------------------------


@dataclass
class SimTarget:
    """A simulated airborne target."""

    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    threat_level: str = "UNKNOWN"
    iff_identification: str = "unknown"

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt


@dataclass
class SimNode:
    """A simulated SENTINEL node with its own sensor footprint."""

    node_id: str
    role: NodeRole
    capabilities: list[str]
    sensor_offset: np.ndarray  # XY offset from origin (meters)
    sensor_range_m: float = 15000.0

    # Network components (set during setup)
    transport: SimulatedTransport | None = None
    broker: PubSubBroker | None = None
    node: NetworkNode | None = None
    discovery: MeshDiscovery | None = None
    bridge: NetworkBridge | None = None
    composite_fusion: CompositeFusion | None = None

    def detect_targets(self, all_targets: list[SimTarget]) -> list[EnhancedFusedTrack]:
        """Simulate local detection + fusion — returns locally fused tracks."""
        tracks = []
        for tgt in all_targets:
            # Check if target is within sensor range (relative to node position)
            rel_pos = tgt.position - self.sensor_offset
            distance = float(np.linalg.norm(rel_pos))
            if distance <= self.sensor_range_m:
                # Add noise
                noise = np.random.normal(0, 15.0, size=rel_pos.shape)
                measured = rel_pos + noise
                pos_abs = measured + self.sensor_offset  # back to global frame

                # Build a mock fused track
                speed = float(np.linalg.norm(tgt.velocity))
                track = EnhancedFusedTrack(
                    fused_id=f"{self.node_id}:{tgt.target_id}",
                    position_m=pos_abs.copy(),
                    velocity_mps=speed,
                    threat_level=tgt.threat_level,
                    iff_identification=tgt.iff_identification,
                    sensor_sources={SensorType.RADAR},
                    confidence=0.8,
                )
                tracks.append(track)
        return tracks


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def build_targets() -> list[SimTarget]:
    """Build a mixed-threat scenario."""
    return [
        SimTarget(
            target_id="BANDIT-1",
            position=np.array([12000.0, 5000.0]),
            velocity=np.array([-80.0, -10.0]),
            threat_level="HIGH",
            iff_identification="hostile",
        ),
        SimTarget(
            target_id="TRACK-2",
            position=np.array([8000.0, -3000.0]),
            velocity=np.array([-50.0, 15.0]),
            threat_level="MEDIUM",
            iff_identification="unknown",
        ),
        SimTarget(
            target_id="FRIEND-3",
            position=np.array([5000.0, 2000.0]),
            velocity=np.array([30.0, 5.0]),
            threat_level="LOW",
            iff_identification="friendly",
        ),
        SimTarget(
            target_id="HYPER-4",
            position=np.array([20000.0, 0.0]),
            velocity=np.array([-1500.0, 50.0]),
            threat_level="CRITICAL",
            iff_identification="hostile",
        ),
    ]


def build_nodes(n_nodes: int) -> list[SimNode]:
    """Create distributed sensor nodes."""
    nodes = [
        SimNode(
            node_id="RADAR-NORTH",
            role=NodeRole.SENSOR,
            capabilities=["radar", "thermal"],
            sensor_offset=np.array([0.0, 5000.0]),
            sensor_range_m=15000.0,
        ),
        SimNode(
            node_id="RADAR-SOUTH",
            role=NodeRole.SENSOR,
            capabilities=["radar"],
            sensor_offset=np.array([0.0, -5000.0]),
            sensor_range_m=12000.0,
        ),
    ]
    if n_nodes >= 3:
        nodes.append(
            SimNode(
                node_id="QI-EAST",
                role=NodeRole.SENSOR,
                capabilities=["quantum_radar", "thermal"],
                sensor_offset=np.array([8000.0, 0.0]),
                sensor_range_m=10000.0,
            )
        )
    return nodes


def setup_network(
    nodes: list[SimNode],
    profile: DegradationProfile,
) -> TransportHub:
    """Wire up transport, pub/sub, node, discovery, bridge, and composite fusion."""
    hub = TransportHub()

    for sn in nodes:
        # Create transport — auto-registers with hub
        transport = SimulatedTransport(
            node_id=sn.node_id,
            hub=hub,
            profile=profile,
        )
        broker = PubSubBroker()
        node = NetworkNode(
            node_id=sn.node_id,
            role=sn.role,
            capabilities=sn.capabilities,
            heartbeat_interval_s=2.0,
        )
        node.start()
        discovery = MeshDiscovery(
            node=node,
            transport=transport,
            heartbeat_timeout_s=6.0,
        )
        bridge = NetworkBridge(
            node=node,
            transport=transport,
            broker=broker,
        )
        composite = CompositeFusion(
            distance_gate=50.0,
            stale_threshold_s=5.0,
            prefer_local=True,
        )

        sn.transport = transport
        sn.broker = broker
        sn.node = node
        sn.discovery = discovery
        sn.bridge = bridge
        sn.composite_fusion = composite

    return hub


def print_header(step: int, t: float, n_nodes: int) -> None:
    """Print step separator."""
    print(f"\n{'='*70}")
    print(f"  Step {step:3d}  |  t = {t:.1f}s  |  {n_nodes} active nodes")
    print(f"{'='*70}")


def print_node_tracks(node: SimNode, local: list, composite: list) -> None:
    """Print track summary for a node."""
    print(f"\n  [{node.node_id}] ({node.role.value})")
    print(f"    Local tracks:     {len(local)}")
    print(f"    Composite tracks: {len(composite)}")
    for tr in composite:
        fid = getattr(tr, "fused_id", "?")
        pos = getattr(tr, "position_m", None)
        thr = getattr(tr, "threat_level", "?")
        iff = getattr(tr, "iff_identification", "?")
        info = node.composite_fusion.get_composite_info(fid) if node.composite_fusion else None
        remote_tag = ""
        if info and info.is_remote_only:
            remote_tag = " [REMOTE]"
        elif info and info.contributing_nodes:
            remote_tag = f" [COMPOSITE: {'+'.join(info.contributing_nodes)}]"
        pos_str = f"({pos[0]:8.0f}, {pos[1]:8.0f})" if pos is not None else "(?, ?)"
        print(f"      {fid:30s}  pos={pos_str}  threat={thr:8s}  iff={iff:10s}{remote_tag}")


def _get_transport_stats(transport: SimulatedTransport, peers: list[str]) -> dict:
    """Gather per-peer stats from a transport."""
    stats = {}
    for peer in peers:
        stats[peer] = transport.get_link_stats(peer)
    return stats


def run_demo(n_nodes: int, profile_name: str, max_steps: int) -> None:
    """Main demo loop."""
    profile = DEGRADATION_PRESETS.get(profile_name)
    if profile is None:
        print(f"Unknown degradation profile: {profile_name}")
        print(f"Available: {', '.join(DEGRADATION_PRESETS.keys())}")
        sys.exit(1)

    dt = 0.5
    clock = SimClock(start_epoch=1_000_000.0)
    targets = build_targets()
    nodes = build_nodes(n_nodes)
    hub = setup_network(nodes, profile)

    # Collect all node IDs for stats lookup
    all_node_ids = [sn.node_id for sn in nodes]

    print("=" * 70)
    print("  SENTINEL Tactical Mesh Network Demo")
    print("=" * 70)
    print(f"  Nodes:        {len(nodes)}")
    print(f"  Degradation:  {profile_name} (latency={profile.latency_ms}ms, "
          f"jitter={profile.jitter_ms}ms, loss={profile.packet_loss_rate*100:.0f}%)")
    print(f"  Targets:      {len(targets)}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Clock:        SimClock (dt={dt}s, epoch={clock.now():.0f})")

    for step in range(1, max_steps + 1):
        t_now = clock.now()

        # 1. Move targets
        for tgt in targets:
            tgt.step(dt)

        # 2. Each node: detect → publish → send heartbeat
        all_local: dict[str, list[EnhancedFusedTrack]] = {}
        all_composite: dict[str, list] = {}

        for sn in nodes:
            # Local detection
            local_tracks = sn.detect_targets(targets)
            all_local[sn.node_id] = local_tracks

            # Publish tracks to network (broadcast to all peers)
            sn.bridge.publish_tracks(local_tracks, t_now)

            # Send heartbeat via discovery
            sn.discovery.step(t_now)

        # 3. Each node: process incoming messages → composite merge
        for sn in nodes:
            # Receive remote tracks from transport inbox
            sn.bridge.process_incoming()

            # Composite fusion: merge local + remote
            remote = sn.bridge.get_remote_tracks()
            composite = sn.composite_fusion.merge(
                local_tracks=all_local[sn.node_id],
                remote_tracks=remote,
                current_time=t_now,
            )
            all_composite[sn.node_id] = composite

        # 4. Print summary every 5 steps or first/last
        if step == 1 or step == max_steps or step % 5 == 0:
            print_header(step, t_now - 1_000_000.0, len(nodes))

            for sn in nodes:
                print_node_tracks(sn, all_local[sn.node_id], all_composite[sn.node_id])

            # Network stats
            print(f"\n  --- Network Stats ---")
            for sn in nodes:
                peer_ids = [nid for nid in all_node_ids if nid != sn.node_id]
                stats = _get_transport_stats(sn.transport, peer_ids)
                total_sent = sum(s.messages_sent for s in stats.values())
                total_recv = sum(s.messages_received for s in stats.values())
                total_drop = sum(s.messages_dropped for s in stats.values())
                peers = len(sn.node.peers) if sn.node else 0
                print(f"    [{sn.node_id}] peers={peers}  sent={total_sent}  "
                      f"recv={total_recv}  dropped={total_drop}")

        # 5. Advance clock
        clock.step(dt)

    # Final summary
    print(f"\n{'='*70}")
    print("  Demo complete.")
    print(f"{'='*70}")

    # Print per-link statistics
    print("\n  Per-link Transport Statistics:")
    for sn in nodes:
        peer_ids = [nid for nid in all_node_ids if nid != sn.node_id]
        for peer_id in peer_ids:
            ls = sn.transport.get_link_stats(peer_id)
            print(f"    {sn.node_id} -> {peer_id}: "
                  f"sent={ls.messages_sent} recv={ls.messages_received} "
                  f"dropped={ls.messages_dropped} bytes={ls.bytes_sent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL tactical mesh network demo")
    parser.add_argument("--nodes", type=int, default=2, choices=[2, 3],
                        help="Number of simulated nodes (default: 2)")
    parser.add_argument("--degradation", type=str, default="ttnt",
                        choices=list(DEGRADATION_PRESETS.keys()),
                        help="Network degradation profile (default: ttnt)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of simulation steps (default: 20)")
    args = parser.parse_args()

    try:
        run_demo(args.nodes, args.degradation, args.steps)
    except KeyboardInterrupt:
        print("\n  Interrupted. Exiting.")


if __name__ == "__main__":
    main()
