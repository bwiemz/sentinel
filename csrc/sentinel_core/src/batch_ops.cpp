#include "sentinel/batch_ops.h"
#include <algorithm>
#include <cmath>

namespace sentinel {

// ---------------------------------------------------------------------------
// Tier 1: KF batch cost matrix
// ---------------------------------------------------------------------------

Mat batch_kf_gating_matrix(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& Z,
    const Mat& H,
    const Mat& R,
    double gate)
{
    const int T = static_cast<int>(X.rows());
    const int D = static_cast<int>(Z.rows());
    const double INF = std::numeric_limits<double>::infinity();

    Mat cost = Mat::Constant(T, D, INF);

    // Precompute H' for reuse
    Mat Ht = H.transpose();

    for (int i = 0; i < T; ++i) {
        // S_i = H * P_i * H' + R
        Mat S = H * P[i] * Ht + R;
        auto ldlt = S.ldlt();
        if (ldlt.info() != Eigen::Success) continue;

        // Predicted measurement for track i
        Vec z_pred = H * X.row(i).transpose();

        for (int j = 0; j < D; ++j) {
            Vec y = Z.row(j).transpose() - z_pred;
            Vec Sinv_y = ldlt.solve(y);
            double d = y.dot(Sinv_y);
            if (d <= gate) {
                cost(i, j) = d;
            }
        }
    }
    return cost;
}

Mat batch_iou_matrix(const Mat& bboxes_a, const Mat& bboxes_b)
{
    const int T = static_cast<int>(bboxes_a.rows());
    const int D = static_cast<int>(bboxes_b.rows());

    Mat iou = Mat::Zero(T, D);

    for (int i = 0; i < T; ++i) {
        double ax1 = bboxes_a(i, 0), ay1 = bboxes_a(i, 1);
        double ax2 = bboxes_a(i, 2), ay2 = bboxes_a(i, 3);
        double area_a = std::max(0.0, ax2 - ax1) * std::max(0.0, ay2 - ay1);

        for (int j = 0; j < D; ++j) {
            double bx1 = bboxes_b(j, 0), by1 = bboxes_b(j, 1);
            double bx2 = bboxes_b(j, 2), by2 = bboxes_b(j, 3);

            double inter_w = std::max(0.0, std::min(ax2, bx2) - std::max(ax1, bx1));
            double inter_h = std::max(0.0, std::min(ay2, by2) - std::max(ay1, by1));
            double intersection = inter_w * inter_h;

            double area_b = std::max(0.0, bx2 - bx1) * std::max(0.0, by2 - by1);
            double uni = area_a + area_b - intersection;

            iou(i, j) = (uni > 0.0) ? intersection / uni : 0.0;
        }
    }
    return iou;
}

Mat batch_camera_cost_matrix(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& Z,
    const Mat& H,
    const Mat& R,
    const Mat& bboxes_a,
    const Mat& bboxes_b,
    double alpha,
    double gate)
{
    const int T = static_cast<int>(X.rows());
    const int D = static_cast<int>(Z.rows());
    const double INF = std::numeric_limits<double>::infinity();

    // Compute Mahalanobis cost matrix (gated)
    Mat maha = batch_kf_gating_matrix(X, P, Z, H, R, gate);

    // Compute IoU matrix
    Mat iou = batch_iou_matrix(bboxes_a, bboxes_b);

    // Combined cost
    Mat cost = Mat::Constant(T, D, INF);
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            if (maha(i, j) < INF) {
                cost(i, j) = alpha * maha(i, j) + (1.0 - alpha) * (1.0 - iou(i, j));
            }
        }
    }
    return cost;
}

// ---------------------------------------------------------------------------
// Tier 2: EKF batch gating
// ---------------------------------------------------------------------------

Mat batch_ekf_gating_matrix(
    const std::vector<Vec>& states,
    const std::vector<Mat>& covariances,
    const Mat& Z,
    const std::vector<Mat>& H_jacobians,
    const std::vector<Vec>& h_predictions,
    const Mat& R,
    const std::vector<int>& angular_indices,
    double gate)
{
    const int T = static_cast<int>(states.size());
    const int D = static_cast<int>(Z.rows());
    const double INF = std::numeric_limits<double>::infinity();

    Mat cost = Mat::Constant(T, D, INF);

    for (int i = 0; i < T; ++i) {
        const Mat& Hi = H_jacobians[i];
        Mat S = Hi * covariances[i] * Hi.transpose() + R;
        auto ldlt = S.ldlt();
        if (ldlt.info() != Eigen::Success) continue;

        const Vec& z_pred = h_predictions[i];

        for (int j = 0; j < D; ++j) {
            Vec y = Z.row(j).transpose() - z_pred;

            // Wrap angular components to [-pi, pi]
            for (int k : angular_indices) {
                if (k < y.size()) {
                    y(k) = wrap_angle(y(k));
                }
            }

            Vec Sinv_y = ldlt.solve(y);
            double d = y.dot(Sinv_y);
            if (d <= gate) {
                cost(i, j) = d;
            }
        }
    }
    return cost;
}

// ---------------------------------------------------------------------------
// Tier 3: JPDA batch likelihood
// ---------------------------------------------------------------------------

Vec batch_gaussian_likelihood(
    const Mat& innovations,
    const Mat& S_inv,
    double log_det_S)
{
    const int N = static_cast<int>(innovations.rows());
    const int d = static_cast<int>(innovations.cols());

    double log_norm = -0.5 * (d * std::log(2.0 * M_PI) + log_det_S);

    Vec result(N);
    for (int i = 0; i < N; ++i) {
        Vec y = innovations.row(i).transpose();
        double mahal = y.dot(S_inv * y);
        double log_L = log_norm - 0.5 * mahal;
        result(i) = std::exp(log_L);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tier 4: Batch predict
// ---------------------------------------------------------------------------

std::pair<Mat, std::vector<Mat>> batch_kf_predict(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& F,
    const Mat& Q)
{
    const int T = static_cast<int>(X.rows());
    Mat Ft = F.transpose();

    // X_pred = X * F'  (each row: F * x_i^T transposed back)
    Mat X_pred = (F * X.transpose()).transpose();

    std::vector<Mat> P_pred(T);
    for (int i = 0; i < T; ++i) {
        P_pred[i] = F * P[i] * Ft + Q;
    }

    return {X_pred, P_pred};
}

} // namespace sentinel
