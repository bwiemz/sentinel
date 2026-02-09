#include "sentinel/cost_matrix.h"
#include <algorithm>
#include <limits>

namespace sentinel {

double iou_bbox(const double* bbox_a, const double* bbox_b) {
    double x1 = std::max(bbox_a[0], bbox_b[0]);
    double y1 = std::max(bbox_a[1], bbox_b[1]);
    double x2 = std::min(bbox_a[2], bbox_b[2]);
    double y2 = std::min(bbox_a[3], bbox_b[3]);

    double inter_w = std::max(0.0, x2 - x1);
    double inter_h = std::max(0.0, y2 - y1);
    double intersection = inter_w * inter_h;

    double area_a = std::max(0.0, bbox_a[2] - bbox_a[0]) *
                    std::max(0.0, bbox_a[3] - bbox_a[1]);
    double area_b = std::max(0.0, bbox_b[2] - bbox_b[0]) *
                    std::max(0.0, bbox_b[3] - bbox_b[1]);
    double uni = area_a + area_b - intersection;

    if (uni <= 0.0) return 0.0;
    return intersection / uni;
}

double track_to_track_mahalanobis(
    const Vec& pos1, const Mat& cov1,
    const Vec& pos2, const Mat& cov2)
{
    Vec dx = pos1 - pos2;
    Mat S = cov1 + cov2;

    // Check for near-singular matrix (all-zero or degenerate covariance)
    double s_norm = S.norm();
    if (s_norm < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }

    auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
        return std::numeric_limits<double>::infinity();
    }
    Vec Sinv_dx = ldlt.solve(dx);
    return dx.dot(Sinv_dx);
}

} // namespace sentinel
