"""Temporal alignment for multi-sensor track fusion.

Predicts all tracks forward to a common reference epoch before fusion,
eliminating errors from asynchronous sensor updates. Critical for
high-speed targets where even 100ms mismatch introduces significant
position error (e.g., 171m at Mach 5).

Uses Constant Velocity (CV) propagation with Continuous White Noise
Acceleration (CWNA) process noise model for time extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AlignedTrackState:
    """Track state predicted to a common reference time.

    Attributes:
        position: Estimated 2D position [x, y] at alignment_time.
        covariance: 2x2 position covariance at alignment_time.
        track_id: Original track identifier.
        sensor_type: Source sensor ("camera", "radar", "thermal").
        alignment_time: The reference epoch this state was predicted to.
        original_track: Reference to the source track object.
    """

    position: np.ndarray
    covariance: np.ndarray
    track_id: str
    sensor_type: str
    alignment_time: float
    original_track: Any


def build_cv_transition(dt: float, dim: int = 2) -> np.ndarray:
    """Build constant-velocity state transition matrix for arbitrary dt.

    State layout per axis: [pos, vel], full state: [x, vx, y, vy, ...].

    Args:
        dt: Time step in seconds.
        dim: Number of spatial dimensions (2 for [x,y], 3 for [x,y,z]).

    Returns:
        F matrix of shape (2*dim, 2*dim).
    """
    n = 2 * dim
    F = np.eye(n)
    for i in range(dim):
        F[2 * i, 2 * i + 1] = dt
    return F


def build_cv_process_noise(dt: float, dim: int = 2, sigma_a: float = 1.0) -> np.ndarray:
    """Build CWNA process noise matrix for arbitrary dt.

    Continuous White Noise Acceleration model:
        Q_block = sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

    Args:
        dt: Time step in seconds.
        dim: Number of spatial dimensions.
        sigma_a: Acceleration noise std (m/s^2 or px/s^2).

    Returns:
        Q matrix of shape (2*dim, 2*dim).
    """
    n = 2 * dim
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q_block = sigma_a**2 * np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2],
    ])
    Q = np.zeros((n, n))
    for i in range(dim):
        Q[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = q_block
    return Q


def _get_sensor_type(track: Any) -> str:
    """Infer sensor type from track class name."""
    cls_name = type(track).__name__.lower()
    if "radar" in cls_name:
        return "radar"
    if "thermal" in cls_name:
        return "thermal"
    return "camera"


def _extract_position_cov(x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract 2D position and covariance from [x, vx, y, vy, ...] state.

    Args:
        x: Full state vector (4D or 6D).
        P: Full state covariance matrix.

    Returns:
        (pos_2d, cov_2x2) where pos_2d = [x, y] and cov_2x2 is the
        position-only covariance submatrix.
    """
    pos = np.array([x[0], x[2]])
    cov = np.array([
        [P[0, 0], P[0, 2]],
        [P[2, 0], P[2, 2]],
    ])
    return pos, cov


def predict_track_to_epoch(track: Any, target_time: float) -> AlignedTrackState:
    """Predict a single track to the target reference time.

    Uses the track's predict_to_time() method for non-mutating prediction.

    Args:
        track: Track object with predict_to_time() method.
        target_time: Reference epoch to predict to.

    Returns:
        AlignedTrackState with position/covariance at target_time.
    """
    x_pred, P_pred = track.predict_to_time(target_time)
    pos, cov = _extract_position_cov(x_pred, P_pred)

    return AlignedTrackState(
        position=pos,
        covariance=cov,
        track_id=track.track_id,
        sensor_type=_get_sensor_type(track),
        alignment_time=target_time,
        original_track=track,
    )


def align_tracks_to_epoch(
    tracks: list[Any],
    reference_time: float,
) -> list[AlignedTrackState]:
    """Align multiple tracks to a common reference epoch.

    Args:
        tracks: List of track objects (any sensor type).
        reference_time: Target epoch for alignment.

    Returns:
        List of AlignedTrackState, one per input track.
    """
    return [predict_track_to_epoch(t, reference_time) for t in tracks]
