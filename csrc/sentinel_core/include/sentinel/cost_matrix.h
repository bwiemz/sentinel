#pragma once

#include "sentinel/common.h"

namespace sentinel {

// IoU between two [x1,y1,x2,y2] bounding boxes
double iou_bbox(const double* bbox_a, const double* bbox_b);

// Track-to-track Mahalanobis: d² = dx'*(P1+P2)^{-1}*dx
double track_to_track_mahalanobis(
    const Vec& pos1, const Mat& cov1,
    const Vec& pos2, const Mat& cov2);

} // namespace sentinel
