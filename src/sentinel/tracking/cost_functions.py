"""Cost functions for data association."""

from __future__ import annotations

import numpy as np

from sentinel.tracking._accel import _HAS_CPP, _sentinel_core
from sentinel.tracking.filters import KalmanFilter


def iou_bbox(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] bounding boxes.

    Returns IoU in [0, 1].
    """
    if _HAS_CPP:
        return _sentinel_core.cost.iou_bbox(
            np.asarray(bbox_a, dtype=np.float64),
            np.asarray(bbox_b, dtype=np.float64),
        )

    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0, bbox_a[2] - bbox_a[0]) * max(0, bbox_a[3] - bbox_a[1])
    area_b = max(0, bbox_b[2] - bbox_b[0]) * max(0, bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def mahalanobis_distance(kf: KalmanFilter, measurement: np.ndarray) -> float:
    """Compute squared Mahalanobis distance for gating.

    Uses the Kalman filter's predicted measurement and innovation covariance.
    """
    return kf.gating_distance(measurement)


def centroid_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(pos_a - pos_b))


def track_to_track_mahalanobis(
    pos1: np.ndarray,
    cov1: np.ndarray,
    pos2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Squared Mahalanobis distance between two track position estimates.

    d² = (x1 - x2)' * (P1 + P2)^{-1} * (x1 - x2)

    Used for track-to-track correlation in multi-sensor fusion.
    Under correct association with Gaussian states, d² ~ chi²(dim).

    Args:
        pos1: Position estimate from track 1.
        cov1: Position covariance from track 1.
        pos2: Position estimate from track 2.
        cov2: Position covariance from track 2.

    Returns:
        Squared Mahalanobis distance (non-negative scalar).
        Returns inf if the combined covariance is singular.
    """
    if _HAS_CPP:
        return _sentinel_core.cost.track_to_track_mahalanobis(
            np.asarray(pos1, dtype=float),
            np.asarray(cov1, dtype=float),
            np.asarray(pos2, dtype=float),
            np.asarray(cov2, dtype=float),
        )

    dx = np.asarray(pos1, dtype=float) - np.asarray(pos2, dtype=float)
    S = np.asarray(cov1, dtype=float) + np.asarray(cov2, dtype=float)
    try:
        return float(dx @ np.linalg.solve(S, dx))
    except np.linalg.LinAlgError:
        return float("inf")
