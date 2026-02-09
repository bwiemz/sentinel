#pragma once

#include "sentinel/common.h"

namespace sentinel {

// KF predict: x_new = F*x, P_new = F*P*F' + Q
std::pair<Vec, Mat> kf_predict(
    const Vec& x, const Mat& P, const Mat& F, const Mat& Q);

// KF update (Joseph form): returns (x_new, P_new)
std::pair<Vec, Mat> kf_update(
    const Vec& x, const Mat& P, const Mat& H, const Mat& R,
    const Vec& z);

// Mahalanobis gating distance: y'*S^{-1}*y where y = z - H*x, S = H*P*H' + R
double kf_gating_distance(
    const Vec& x, const Mat& P, const Mat& H, const Mat& R,
    const Vec& z);

// Innovation covariance: S = H*P*H' + R
Mat kf_innovation_covariance(
    const Mat& P, const Mat& H, const Mat& R);

// Predicted measurement: z_pred = H * x
Vec kf_predicted_measurement(const Vec& x, const Mat& H);

} // namespace sentinel
