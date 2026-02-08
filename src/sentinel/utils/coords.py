"""Coordinate transform utilities for sensor fusion."""

from __future__ import annotations

import numpy as np


def polar_to_cartesian(range_m: float, azimuth_rad: float) -> np.ndarray:
    """Convert (range, azimuth) to Cartesian [x, y].

    Convention: azimuth=0 is along +x axis, positive counter-clockwise.
    """
    return np.array([
        range_m * np.cos(azimuth_rad),
        range_m * np.sin(azimuth_rad),
    ])


def cartesian_to_polar(x: float, y: float) -> tuple[float, float]:
    """Convert Cartesian (x, y) to (range_m, azimuth_rad).

    Returns:
        (range_m, azimuth_rad) where azimuth is in [-pi, pi].
    """
    r = np.sqrt(x * x + y * y)
    az = np.arctan2(y, x)
    return float(r), float(az)


def azimuth_deg_to_rad(deg: float) -> float:
    """Convert azimuth from degrees to radians."""
    return float(np.radians(deg))


def azimuth_rad_to_deg(rad: float) -> float:
    """Convert azimuth from radians to degrees."""
    return float(np.degrees(rad))


def normalize_angle(angle_rad: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)
