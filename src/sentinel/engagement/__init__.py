"""Engagement zone management — WEZ, geographic zones, weapon-target assignment.

Provides spatial engagement logic: geographic engagement zones with
authorization levels, weapon system profiles with performance envelopes,
dynamic Weapon Engagement Zone (WEZ) computation, Pk/TTI feasibility
scoring, and optimal weapon-target assignment via Hungarian algorithm.
"""

from sentinel.engagement.zones import (
    AnnularZone,
    CircularZone,
    EngagementZone,
    PolygonZone,
    SectorZone,
    ZoneManager,
)
from sentinel.engagement.weapons import (
    WEZCalculator,
    WEZResult,
    WeaponProfile,
)
from sentinel.engagement.feasibility import (
    EngagementFeasibility,
    FeasibilityCalculator,
)
from sentinel.engagement.assignment import (
    EngagementAssignment,
    WeaponTargetAssigner,
)
from sentinel.engagement.manager import (
    EngagementManager,
    EngagementPlan,
)

__all__ = [
    "AnnularZone",
    "CircularZone",
    "EngagementAssignment",
    "EngagementFeasibility",
    "EngagementManager",
    "EngagementPlan",
    "EngagementZone",
    "FeasibilityCalculator",
    "PolygonZone",
    "SectorZone",
    "WEZCalculator",
    "WEZResult",
    "WeaponProfile",
    "WeaponTargetAssigner",
    "ZoneManager",
]
