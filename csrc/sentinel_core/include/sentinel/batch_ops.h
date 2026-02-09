#pragma once

#include "sentinel/common.h"
#include <limits>

namespace sentinel {

// ---------------------------------------------------------------------------
// Tier 1: KF batch cost matrix
// ---------------------------------------------------------------------------

// Build (T, D) Mahalanobis distance matrix for T tracks vs D measurements.
// X: (T, n) state matrix, P: T covariance matrices (n, n),
// Z: (D, m) measurement matrix, H: (m, n) observation, R: (m, m) meas noise.
// gate: chi-squared threshold; entries exceeding it are set to +inf.
Mat batch_kf_gating_matrix(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& Z,
    const Mat& H,
    const Mat& R,
    double gate);

// Build (T, D) IoU matrix between two sets of bounding boxes.
// bboxes_a: (T, 4) and bboxes_b: (D, 4), each row = [x1, y1, x2, y2].
Mat batch_iou_matrix(
    const Mat& bboxes_a,
    const Mat& bboxes_b);

// Combined camera cost: alpha * mahalanobis + (1 - alpha) * (1 - IoU).
// Applies gate on Mahalanobis component; gated entries = +inf.
Mat batch_camera_cost_matrix(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& Z,
    const Mat& H,
    const Mat& R,
    const Mat& bboxes_a,
    const Mat& bboxes_b,
    double alpha,
    double gate);

// ---------------------------------------------------------------------------
// Tier 2: EKF batch gating
// ---------------------------------------------------------------------------

// Build (T, D) cost matrix for EKF tracks with per-track Jacobians.
// states: T state vectors (may differ in dim), covariances: T covariances,
// Z: (D, m) measurements, H_jacobians: T Jacobian matrices,
// h_predictions: T predicted measurement vectors,
// R: (m, m) shared measurement noise.
// angular_indices: which measurement components are angles (for wrapping).
Mat batch_ekf_gating_matrix(
    const std::vector<Vec>& states,
    const std::vector<Mat>& covariances,
    const Mat& Z,
    const std::vector<Mat>& H_jacobians,
    const std::vector<Vec>& h_predictions,
    const Mat& R,
    const std::vector<int>& angular_indices,
    double gate);

// ---------------------------------------------------------------------------
// Tier 3: JPDA batch likelihood
// ---------------------------------------------------------------------------

// Compute Gaussian likelihood for N innovations against shared S.
// innovations: (N, m), S_inv: (m, m) pre-inverted, log_det_S: log|S|.
// Returns: (N,) likelihood values.
Vec batch_gaussian_likelihood(
    const Mat& innovations,
    const Mat& S_inv,
    double log_det_S);

// ---------------------------------------------------------------------------
// Tier 4: Batch predict
// ---------------------------------------------------------------------------

// Predict T KF states in one call.
// X: (T, n), P: T covariances, F: (n, n), Q: (n, n).
// Returns: (X_pred, P_pred).
std::pair<Mat, std::vector<Mat>> batch_kf_predict(
    const Mat& X,
    const std::vector<Mat>& P,
    const Mat& F,
    const Mat& Q);

} // namespace sentinel
