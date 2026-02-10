"""SENTINEL multi-target engagement zone demo.

Simulates a defended area with SAM batteries, CIWS, and geographic
engagement zones. Shows dynamic weapon-target assignment with Pk/TTI
scoring as threats approach.

Run:
    python scripts/demo_engagement.py
    python scripts/demo_engagement.py --steps 30 --verbose
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from sentinel.core.types import EngagementAuth, ZoneAuth, WeaponType
from sentinel.engagement.config import EngagementConfig
from sentinel.engagement.manager import EngagementManager
from sentinel.engagement.weapons import WeaponProfile
from sentinel.engagement.zones import (
    CircularZone,
    AnnularZone,
    SectorZone,
    ZoneManager,
)


# ---------------------------------------------------------------------------
# Mock track (lightweight stand-in for EnhancedFusedTrack)
# ---------------------------------------------------------------------------


class MockTrack:
    """Lightweight track object compatible with EngagementManager._extract_track_dict."""

    def __init__(
        self,
        fused_id: str,
        position_m: np.ndarray,
        velocity_mps: float,
        azimuth_deg: float,
        threat_level: str,
        engagement_auth: EngagementAuth,
        iff_identification: str,
        is_jammed: bool = False,
    ):
        self.fused_id = fused_id
        self.position_m = np.asarray(position_m, dtype=float)
        self.velocity_mps = velocity_mps
        self.azimuth_deg = azimuth_deg
        self.threat_level = threat_level
        self.engagement_auth = engagement_auth
        self.iff_identification = iff_identification
        self.is_jammed = is_jammed
        self.radar_track = None


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def build_zones() -> ZoneManager:
    """Build the four engagement zones around the defended area."""
    zones = [
        # Hospital no-fire zone (highest priority -- overrides everything)
        CircularZone(
            zone_id="ZONE-NOFIRE",
            name="Hospital No-Fire",
            center_xy=np.array([2000.0, 3000.0]),
            radius_m=1500.0,
            authorization=ZoneAuth.NO_FIRE,
            priority=10,
        ),
        # Main defense ring (weapons free between 5km and 25km)
        AnnularZone(
            zone_id="ZONE-WF",
            name="Weapons Free Ring",
            center_xy=np.array([0.0, 0.0]),
            inner_radius_m=5000.0,
            outer_radius_m=25000.0,
            authorization=ZoneAuth.WEAPONS_FREE,
            priority=1,
        ),
        # Restricted-fire northeast sector (diplomatic corridor)
        SectorZone(
            zone_id="ZONE-RF",
            name="Restricted Fire NE Sector",
            center_xy=np.array([0.0, 0.0]),
            radius_m=30000.0,
            azimuth_min_deg=30.0,
            azimuth_max_deg=60.0,
            authorization=ZoneAuth.RESTRICTED_FIRE,
            priority=5,
        ),
        # Inner self-defense zone (close-in protection)
        CircularZone(
            zone_id="ZONE-SD",
            name="Self-Defense Inner Zone",
            center_xy=np.array([0.0, 0.0]),
            radius_m=5000.0,
            authorization=ZoneAuth.SELF_DEFENSE_ONLY,
            priority=2,
        ),
    ]
    return ZoneManager(zones=zones, default_authorization=ZoneAuth.WEAPONS_FREE)


def build_weapons() -> list[WeaponProfile]:
    """Build the three weapon systems defending the area."""
    return [
        WeaponProfile(
            weapon_id="PATRIOT-1",
            name="Patriot",
            weapon_type=WeaponType.SAM_MEDIUM,
            position_xy=np.array([0.0, 0.0]),
            min_range_m=3000.0,
            max_range_m=70000.0,
            optimal_range_m=40000.0,
            max_target_speed_mps=2000.0,
            weapon_speed_mps=1700.0,
            pk_base=0.85,
            max_simultaneous_engagements=4,
            rounds_remaining=16,
            salvo_size=1,
            reload_time_s=8.0,
            min_altitude_m=0.0,
            max_altitude_m=25000.0,
        ),
        WeaponProfile(
            weapon_id="STINGER-1",
            name="Stinger",
            weapon_type=WeaponType.SAM_SHORT,
            position_xy=np.array([2000.0, 0.0]),
            min_range_m=200.0,
            max_range_m=5000.0,
            optimal_range_m=2500.0,
            max_target_speed_mps=600.0,
            weapon_speed_mps=750.0,
            pk_base=0.70,
            max_simultaneous_engagements=2,
            rounds_remaining=6,
            salvo_size=1,
            reload_time_s=10.0,
            min_altitude_m=0.0,
            max_altitude_m=3500.0,
        ),
        WeaponProfile(
            weapon_id="PHALANX-1",
            name="Phalanx",
            weapon_type=WeaponType.CIWS,
            position_xy=np.array([-500.0, 0.0]),
            min_range_m=200.0,
            max_range_m=1500.0,
            optimal_range_m=800.0,
            max_target_speed_mps=700.0,
            weapon_speed_mps=1100.0,
            pk_base=0.60,
            max_simultaneous_engagements=1,
            rounds_remaining=1550,
            salvo_size=1,
            reload_time_s=0.0,
            min_altitude_m=0.0,
            max_altitude_m=2000.0,
        ),
    ]


# Target specification: [id, start_pos, velocity, threat, auth, iff]
TARGET_DEFS: list[dict] = [
    {
        "id": "BANDIT-1",
        "description": "Conventional fighter from NE",
        "start": [40000.0, 20000.0],
        "velocity": [-200.0, -100.0],
        "threat": "HIGH",
        "auth": EngagementAuth.WEAPONS_FREE,
        "iff": "hostile",
    },
    {
        "id": "VAMPIRE-1",
        "description": "Cruise missile from N",
        "start": [5000.0, 45000.0],
        "velocity": [0.0, -350.0],
        "threat": "CRITICAL",
        "auth": EngagementAuth.WEAPONS_FREE,
        "iff": "hostile",
    },
    {
        "id": "GHOST-1",
        "description": "Stealth aircraft from E",
        "start": [35000.0, 0.0],
        "velocity": [-150.0, 30.0],
        "threat": "HIGH",
        "auth": EngagementAuth.WEAPONS_TIGHT,
        "iff": "assumed_hostile",
    },
    {
        "id": "FRIEND-1",
        "description": "Friendly transport from S",
        "start": [-10000.0, -30000.0],
        "velocity": [50.0, 100.0],
        "threat": "LOW",
        "auth": EngagementAuth.HOLD_FIRE,
        "iff": "friendly",
    },
    {
        "id": "DECOY-1",
        "description": "Decoy near hospital",
        "start": [3000.0, 8000.0],
        "velocity": [0.0, -50.0],
        "threat": "LOW",
        "auth": EngagementAuth.WEAPONS_FREE,
        "iff": "unknown",
    },
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def compute_azimuth_deg(vx: float, vy: float) -> float:
    """Compass bearing from velocity vector (0=North/+Y, CW)."""
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(vx, vy)) % 360.0


def build_tracks(step: int, dt: float) -> list[MockTrack]:
    """Create MockTrack list with positions updated for the current step."""
    tracks: list[MockTrack] = []
    for td in TARGET_DEFS:
        t = step * dt
        vx, vy = td["velocity"]
        px = td["start"][0] + vx * t
        py = td["start"][1] + vy * t
        speed = math.sqrt(vx * vx + vy * vy)
        azimuth = compute_azimuth_deg(vx, vy)

        tracks.append(MockTrack(
            fused_id=td["id"],
            position_m=np.array([px, py]),
            velocity_mps=speed,
            azimuth_deg=azimuth,
            threat_level=td["threat"],
            engagement_auth=td["auth"],
            iff_identification=td["iff"],
        ))
    return tracks


def range_from_origin(pos: np.ndarray) -> float:
    """2-D range from [0, 0]."""
    return float(np.linalg.norm(pos[:2]))


def zone_name_for_track(zone_mgr: ZoneManager, pos: np.ndarray) -> str:
    """Human-readable name for the effective zone at a position."""
    auth = zone_mgr.resolve_authorization(pos)
    return auth.value.upper()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_step(
    step: int,
    total_steps: int,
    t: float,
    tracks: list[MockTrack],
    plan,  # EngagementPlan
    zone_mgr: ZoneManager,
    verbose: bool,
) -> None:
    """Print one simulation step."""
    assignment = plan.assignment

    print(f"\n{'='*72}")
    print(f"=== Step {step} / {total_steps}  (t={t:.1f}s) ===")
    print(f"{'='*72}")

    # -- Targets --
    print("  Targets:")
    for trk in tracks:
        pos = trk.position_m
        rng = range_from_origin(pos)
        zone_label = zone_name_for_track(zone_mgr, pos)
        print(
            f"    {trk.fused_id:12s}  pos=[{pos[0]:7.0f}, {pos[1]:7.0f}]"
            f"  range={rng:7.0f}m  zone={zone_label:20s}  threat={trk.threat_level}"
        )

    # -- Assignments --
    if assignment.assignments:
        print("  Assignments:")
        for wid, tid in assignment.assignments:
            feas = assignment.feasibility_map.get((wid, tid))
            if feas:
                pk = feas.pk
                tti = feas.tti_s
                quality = feas.quality_score
                tti_str = f"{tti:6.1f}s" if math.isfinite(tti) else "   inf"
                print(
                    f"    {wid:12s}  -> {tid:12s}"
                    f"  Pk={pk:.2f}  TTI={tti_str}  quality={quality:.2f}"
                )
    else:
        print("  Assignments: (none)")

    # -- Unassigned --
    if assignment.unassigned_tracks:
        print(f"  Unassigned tracks: {', '.join(assignment.unassigned_tracks)}")
    if assignment.unassigned_weapons:
        print(f"  Unassigned weapons: {', '.join(assignment.unassigned_weapons)}")

    # -- Verbose feasibility details --
    if verbose and assignment.feasibility_map:
        print("  Feasibility details:")
        for (wid, tid), feas in sorted(assignment.feasibility_map.items()):
            wez = feas.wez
            permitted = "YES" if feas.engagement_permitted else "NO"
            print(
                f"    {wid:12s} x {tid:12s}: "
                f"range={wez.slant_range_m:7.0f}m  "
                f"inWEZ={wez.feasible}  "
                f"zone={feas.zone_authorization.value:16s}  "
                f"roe={feas.roe_authorization.value:14s}  "
                f"permitted={permitted}"
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    total_steps: int,
    zone_mgr: ZoneManager,
    weapons: list[WeaponProfile],
    max_assignments: int,
    friendly_excluded: bool,
    nofire_respected: bool,
) -> None:
    """Print final engagement summary."""
    zones = zone_mgr.get_all_zones()
    zone_names = [z.authorization.value.upper() for z in zones]
    weapon_names = [w.name for w in weapons]
    target_names = [td["id"] for td in TARGET_DEFS]

    print(f"\n{'='*72}")
    print("=== ENGAGEMENT SUMMARY ===")
    print(f"{'='*72}")
    print(f"Total steps: {total_steps}")
    print(f"Zones: {len(zones)} ({', '.join(zone_names)})")
    print(f"Weapons: {len(weapons)} ({', '.join(weapon_names)})")
    print(f"Targets: {len(target_names)} ({', '.join(target_names)})")
    print(f"Max simultaneous assignments: {max_assignments}")
    print(f"Friendly correctly excluded: {'yes' if friendly_excluded else 'no'}")
    print(f"NO_FIRE zone respected: {'yes' if nofire_respected else 'no'}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_demo(total_steps: int, verbose: bool) -> None:
    """Run the engagement zone demo."""
    dt = 1.0  # seconds per step

    # Build scenario
    zone_mgr = build_zones()
    weapons = build_weapons()

    # Build EngagementManager manually (bypass config-file path)
    config = EngagementConfig(
        enabled=True,
        pk_weight=0.4,
        tti_weight=0.3,
        threat_weight=0.3,
        max_tti_s=120.0,
    )
    manager = EngagementManager(
        config=config,
        zone_manager=zone_mgr,
    )
    # Add weapons directly
    for w in weapons:
        manager.add_weapon(w)

    # Tracking variables for summary
    max_assignments = 0
    friendly_ever_assigned = False
    nofire_ever_assigned = False

    # Header
    print("=" * 72)
    print("  SENTINEL Multi-Target Engagement Zone Demo")
    print("=" * 72)
    print(f"  Steps:   {total_steps}")
    print(f"  Zones:   {len(zone_mgr.get_all_zones())}")
    print(f"  Weapons: {len(weapons)} "
          f"({', '.join(w.name + ' [' + w.weapon_type.value + ']' for w in weapons)})")
    print(f"  Targets: {len(TARGET_DEFS)} "
          f"({', '.join(td['id'] + ' - ' + td['description'] for td in TARGET_DEFS)})")
    print(f"  Verbose: {verbose}")

    for step in range(1, total_steps + 1):
        t = step * dt
        tracks = build_tracks(step, dt)

        # Evaluate engagement
        plan = manager.evaluate(tracks, current_time=t)
        assignment = plan.assignment

        # Track summary metrics
        n_assign = len(assignment.assignments)
        if n_assign > max_assignments:
            max_assignments = n_assign

        # Check if FRIEND-1 was ever assigned
        for wid, tid in assignment.assignments:
            if tid == "FRIEND-1":
                friendly_ever_assigned = True

        # Check if any track in the no-fire zone was assigned
        for wid, tid in assignment.assignments:
            # Find the matching track
            for trk in tracks:
                if trk.fused_id == tid:
                    zone_auth = zone_mgr.resolve_authorization(trk.position_m)
                    if zone_auth == ZoneAuth.NO_FIRE:
                        nofire_ever_assigned = True

        # Print every step
        print_step(step, total_steps, t, tracks, plan, zone_mgr, verbose)

    # Final summary
    print_summary(
        total_steps=total_steps,
        zone_mgr=zone_mgr,
        weapons=weapons,
        max_assignments=max_assignments,
        friendly_excluded=not friendly_ever_assigned,
        nofire_respected=not nofire_ever_assigned,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SENTINEL multi-target engagement zone demo",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of simulation steps (default: 20)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-weapon feasibility details each step",
    )
    args = parser.parse_args()

    try:
        run_demo(args.steps, args.verbose)
    except KeyboardInterrupt:
        print("\n  Interrupted. Exiting.")


if __name__ == "__main__":
    main()
