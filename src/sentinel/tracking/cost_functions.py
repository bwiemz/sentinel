"""Cost functions for data association."""

from __future__ import annotations

import numpy as np

from sentinel.tracking.filters import KalmanFilter


def iou_bbox(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] bounding boxes.

    Returns IoU in [0, 1].
    """
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
