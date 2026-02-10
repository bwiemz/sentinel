"""Weapon system profiles and Weapon Engagement Zone (WEZ) computation.

Defines weapon performance envelopes and evaluates whether specific
targets fall within a weapon's engagement zone based on range, altitude,
speed, and aspect angle constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sentinel.core.types import WeaponType


@dataclass(frozen=True)
class WeaponProfile:
    """Performance envelope for a weapon system."""

    weapon_id: str
    name: str
    weapon_type: WeaponType
    position_xy: np.ndarray  # ENU [east, north]
    altitude_m: float = 0.0

    # Range envelope
    min_range_m: float = 500.0
    max_range_m: float = 20000.0
    optimal_range_m: float = 10000.0

    # Altitude envelope
    min_altitude_m: float = 100.0
    max_altitude_m: float = 20000.0

    # Speed limits
    max_target_speed_mps: float = 800.0
    weapon_speed_mps: float = 1200.0

    # Kill probability parameters
    pk_base: float = 0.85
    pk_range_falloff: float = 2.0
    pk_speed_penalty: float = 0.3
    pk_ecm_penalty: float = 0.3

    # Aspect angle limits (degrees from target nose)
    min_aspect_angle_deg: float = 0.0
    max_aspect_angle_deg: float = 180.0

    # Capacity
    max_simultaneous_engagements: int = 2
    rounds_remaining: int = 10
    salvo_size: int = 1
    reload_time_s: float = 5.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WeaponProfile):
            return NotImplemented
        return self.weapon_id == other.weapon_id

    def __hash__(self) -> int:
        return hash(self.weapon_id)

    @classmethod
    def from_config(cls, cfg: dict, weapon_id: str = "") -> WeaponProfile:
        """Parse a weapon profile from a config dict."""
        wid = cfg.get("weapon_id", weapon_id) or f"WPN-{id(cfg)}"
        wtype_str = cfg.get("weapon_type", "sam_medium")
        try:
            wtype = WeaponType(wtype_str)
        except ValueError:
            wtype = WeaponType.SAM_MEDIUM
        pos = np.array(cfg.get("position_xy", [0.0, 0.0]), dtype=float)
        return cls(
            weapon_id=wid,
            name=cfg.get("name", wid),
            weapon_type=wtype,
            position_xy=pos,
            altitude_m=float(cfg.get("altitude_m", 0.0)),
            min_range_m=float(cfg.get("min_range_m", 500.0)),
            max_range_m=float(cfg.get("max_range_m", 20000.0)),
            optimal_range_m=float(cfg.get("optimal_range_m", 10000.0)),
            min_altitude_m=float(cfg.get("min_altitude_m", 100.0)),
            max_altitude_m=float(cfg.get("max_altitude_m", 20000.0)),
            max_target_speed_mps=float(cfg.get("max_target_speed_mps", 800.0)),
            weapon_speed_mps=float(cfg.get("weapon_speed_mps", 1200.0)),
            pk_base=float(cfg.get("pk_base", 0.85)),
            pk_range_falloff=float(cfg.get("pk_range_falloff", 2.0)),
            pk_speed_penalty=float(cfg.get("pk_speed_penalty", 0.3)),
            pk_ecm_penalty=float(cfg.get("pk_ecm_penalty", 0.3)),
            min_aspect_angle_deg=float(cfg.get("min_aspect_angle_deg", 0.0)),
            max_aspect_angle_deg=float(cfg.get("max_aspect_angle_deg", 180.0)),
            max_simultaneous_engagements=int(cfg.get("max_simultaneous_engagements", 2)),
            rounds_remaining=int(cfg.get("rounds_remaining", 10)),
            salvo_size=int(cfg.get("salvo_size", 1)),
            reload_time_s=float(cfg.get("reload_time_s", 5.0)),
        )

    def to_dict(self) -> dict:
        return {
            "weapon_id": self.weapon_id,
            "name": self.name,
            "weapon_type": self.weapon_type.value,
            "position_xy": self.position_xy.tolist(),
            "altitude_m": self.altitude_m,
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
            "optimal_range_m": self.optimal_range_m,
            "min_altitude_m": self.min_altitude_m,
            "max_altitude_m": self.max_altitude_m,
            "max_target_speed_mps": self.max_target_speed_mps,
            "weapon_speed_mps": self.weapon_speed_mps,
            "pk_base": self.pk_base,
            "max_simultaneous_engagements": self.max_simultaneous_engagements,
            "rounds_remaining": self.rounds_remaining,
        }


@dataclass
class WEZResult:
    """Weapon Engagement Zone evaluation for a specific weapon-target pair."""

    weapon_id: str
    track_id: str

    in_range: bool = False
    in_altitude: bool = False
    in_speed: bool = False
    in_aspect: bool = False
    feasible: bool = False

    slant_range_m: float = 0.0
    target_altitude_m: float = 0.0
    target_speed_mps: float = 0.0
    aspect_angle_deg: float = 0.0
    closing_speed_mps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "weapon_id": self.weapon_id,
            "track_id": self.track_id,
            "feasible": self.feasible,
            "in_range": self.in_range,
            "in_altitude": self.in_altitude,
            "in_speed": self.in_speed,
            "in_aspect": self.in_aspect,
            "slant_range_m": round(self.slant_range_m, 1),
            "target_altitude_m": round(self.target_altitude_m, 1),
            "target_speed_mps": round(self.target_speed_mps, 1),
            "aspect_angle_deg": round(self.aspect_angle_deg, 1),
            "closing_speed_mps": round(self.closing_speed_mps, 1),
        }


class WEZCalculator:
    """Computes Weapon Engagement Zone evaluations for weapon-target pairs."""

    def evaluate(
        self,
        weapon: WeaponProfile,
        track_position: np.ndarray,
        track_velocity: np.ndarray,
        track_id: str = "",
    ) -> WEZResult:
        """Evaluate whether a target is within a weapon's engagement zone.

        Args:
            weapon: Weapon system profile.
            track_position: Target ENU position [x, y] or [x, y, z].
            track_velocity: Target velocity [vx, vy] or [vx, vy, vz].
            track_id: Target track identifier.
        """
        target_alt = float(track_position[2]) if len(track_position) >= 3 else 0.0
        slant_range = self.compute_slant_range(
            weapon.position_xy, weapon.altitude_m,
            track_position, target_alt,
        )
        speed = float(np.linalg.norm(track_velocity))
        aspect = self.compute_aspect_angle(
            weapon.position_xy, track_position, track_velocity,
        )
        closing = self.compute_closing_speed(
            weapon.position_xy, track_position, track_velocity,
        )

        in_range = weapon.min_range_m <= slant_range <= weapon.max_range_m
        in_alt = weapon.min_altitude_m <= target_alt <= weapon.max_altitude_m
        in_speed = speed <= weapon.max_target_speed_mps
        in_aspect = weapon.min_aspect_angle_deg <= aspect <= weapon.max_aspect_angle_deg

        return WEZResult(
            weapon_id=weapon.weapon_id,
            track_id=track_id,
            in_range=in_range,
            in_altitude=in_alt,
            in_speed=in_speed,
            in_aspect=in_aspect,
            feasible=in_range and in_alt and in_speed and in_aspect,
            slant_range_m=slant_range,
            target_altitude_m=target_alt,
            target_speed_mps=speed,
            aspect_angle_deg=aspect,
            closing_speed_mps=closing,
        )

    @staticmethod
    def compute_slant_range(
        weapon_pos: np.ndarray,
        weapon_alt: float,
        target_pos: np.ndarray,
        target_alt: float,
    ) -> float:
        """3D slant range from weapon to target."""
        dx = float(target_pos[0]) - float(weapon_pos[0])
        dy = float(target_pos[1]) - float(weapon_pos[1])
        dz = target_alt - weapon_alt
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def compute_aspect_angle(
        weapon_pos: np.ndarray,
        target_pos: np.ndarray,
        target_velocity: np.ndarray,
    ) -> float:
        """Angle between weapon-to-target LOS and target heading (degrees).

        0° = head-on (target flying toward weapon)
        180° = tail-on (target flying away from weapon)
        """
        # Weapon-to-target line of sight vector
        los = np.array([
            float(target_pos[0]) - float(weapon_pos[0]),
            float(target_pos[1]) - float(weapon_pos[1]),
        ])
        los_norm = np.linalg.norm(los)
        vel_2d = np.array([float(target_velocity[0]), float(target_velocity[1])])
        vel_norm = np.linalg.norm(vel_2d)

        if los_norm < 1e-9 or vel_norm < 1e-9:
            return 90.0  # Undefined — return broadside

        cos_angle = np.dot(-los, vel_2d) / (los_norm * vel_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def compute_closing_speed(
        weapon_pos: np.ndarray,
        target_pos: np.ndarray,
        target_velocity: np.ndarray,
    ) -> float:
        """Radial closing speed (positive = target approaching weapon)."""
        los = np.array([
            float(weapon_pos[0]) - float(target_pos[0]),
            float(weapon_pos[1]) - float(target_pos[1]),
        ])
        los_norm = np.linalg.norm(los)
        if los_norm < 1e-9:
            return 0.0
        los_unit = los / los_norm
        vel_2d = np.array([float(target_velocity[0]), float(target_velocity[1])])
        return float(np.dot(vel_2d, los_unit))
