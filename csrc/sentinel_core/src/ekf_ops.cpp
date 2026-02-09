#include "sentinel/ekf_ops.h"
#include <algorithm>

namespace sentinel {

// --- 2D polar EKF: state [x, vx, y, vy] ---
Vec2 ekf_h_polar(const Vec4& x) {
    double px = x(0), py = x(2);
    double r = std::sqrt(px * px + py * py);
    double az = std::atan2(py, px);
    return Vec2(std::max(r, 1e-6), az);
}

Eigen::Matrix<double, 2, 4> ekf_H_polar(const Vec4& x) {
    double px = x(0), py = x(2);
    double r = std::max(std::sqrt(px * px + py * py), 1e-6);
    double r2 = r * r;
    Eigen::Matrix<double, 2, 4> H;
    H << px / r, 0.0, py / r, 0.0,
        -py / r2, 0.0, px / r2, 0.0;
    return H;
}

// --- Bearing-only EKF ---
double ekf_h_bearing(const Vec4& x) {
    return std::atan2(x(2), x(0));
}

Eigen::Matrix<double, 1, 4> ekf_H_bearing(const Vec4& x) {
    double px = x(0), py = x(2);
    double r2 = std::max(px * px + py * py, 1e-12);
    Eigen::Matrix<double, 1, 4> H;
    H << -py / r2, 0.0, px / r2, 0.0;
    return H;
}

// --- 3D polar EKF: state [x, vx, y, vy, z, vz] ---
Vec3 ekf_h_3d(const Eigen::Matrix<double, 6, 1>& x) {
    double px = x(0), py = x(2), pz = x(4);
    double r = std::max(std::sqrt(px * px + py * py + pz * pz), 1e-6);
    double az = std::atan2(py, px);
    double r_xy = std::sqrt(px * px + py * py);
    double el = std::atan2(pz, std::max(r_xy, 1e-6));
    return Vec3(r, az, el);
}

Eigen::Matrix<double, 3, 6> ekf_H_3d(const Eigen::Matrix<double, 6, 1>& x) {
    double px = x(0), py = x(2), pz = x(4);
    double r_xy_sq = std::max(px * px + py * py, 1e-12);
    double r_xy = std::sqrt(r_xy_sq);
    double r_sq = r_xy_sq + pz * pz;
    double r = std::max(std::sqrt(r_sq), 1e-6);

    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    // d(range)/d(state)
    H(0, 0) = px / r;
    H(0, 2) = py / r;
    H(0, 4) = pz / r;
    // d(azimuth)/d(state)
    H(1, 0) = -py / r_xy_sq;
    H(1, 2) = px / r_xy_sq;
    // d(elevation)/d(state)
    double denom = r_sq * r_xy;
    if (denom > 1e-12) {
        H(2, 0) = -px * pz / (r_sq * r_xy);
        H(2, 2) = -py * pz / (r_sq * r_xy);
        H(2, 4) = r_xy / r_sq;
    }
    return H;
}

// --- Doppler EKF: state [x, vx, y, vy], meas [range, az, v_radial] ---
Vec3 ekf_h_doppler(const Vec4& x) {
    double px = x(0), vx = x(1), py = x(2), vy = x(3);
    double r = std::max(std::sqrt(px * px + py * py), 1e-6);
    double az = std::atan2(py, px);
    double v_r = (px * vx + py * vy) / r;
    return Vec3(r, az, v_r);
}

Eigen::Matrix<double, 3, 4> ekf_H_doppler(const Vec4& x) {
    double px = x(0), vx = x(1), py = x(2), vy = x(3);
    double r = std::max(std::sqrt(px * px + py * py), 1e-6);
    double r2 = r * r;
    double r3 = r2 * r;
    double dot_pv = px * vx + py * vy;

    Eigen::Matrix<double, 3, 4> H;
    H << px / r, 0.0, py / r, 0.0,
        -py / r2, 0.0, px / r2, 0.0,
         vx / r - px * dot_pv / r3, px / r,
         vy / r - py * dot_pv / r3, py / r;
    return H;
}

// --- CA-EKF: state [x, vx, ax, y, vy, ay] ---
Vec2 ekf_h_ca_polar(const Eigen::Matrix<double, 6, 1>& x) {
    double px = x(0), py = x(3);
    double r = std::max(std::sqrt(px * px + py * py), 1e-6);
    double az = std::atan2(py, px);
    return Vec2(r, az);
}

Eigen::Matrix<double, 2, 6> ekf_H_ca_polar(const Eigen::Matrix<double, 6, 1>& x) {
    double px = x(0), py = x(3);
    double r = std::max(std::sqrt(px * px + py * py), 1e-6);
    double r2 = r * r;
    Eigen::Matrix<double, 2, 6> H = Eigen::Matrix<double, 2, 6>::Zero();
    H(0, 0) = px / r;
    H(0, 3) = py / r;
    H(1, 0) = -py / r2;
    H(1, 3) = px / r2;
    return H;
}

// --- Generic EKF update with angular wrapping ---
std::pair<Vec, Mat> ekf_update(
    const Vec& x, const Mat& P, const Mat& R,
    const Vec& z, const Vec& h_x, const Mat& H,
    const std::vector<int>& angular_indices)
{
    int n = static_cast<int>(x.size());
    Vec y = z - h_x;
    for (int idx : angular_indices) {
        if (idx >= 0 && idx < y.size()) {
            y(idx) = wrap_angle(y(idx));
        }
    }
    Mat S = H * P * H.transpose() + R;
    Mat PH = P * H.transpose();
    Mat K = S.transpose().ldlt().solve(PH.transpose()).transpose();
    Vec x_new = x + K * y;
    Mat I_KH = Mat::Identity(n, n) - K * H;
    Mat P_new = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    return {x_new, P_new};
}

double ekf_gating_distance(
    const Vec& x, const Mat& P, const Mat& R,
    const Vec& z, const Vec& h_x, const Mat& H,
    const std::vector<int>& angular_indices)
{
    Vec y = z - h_x;
    for (int idx : angular_indices) {
        if (idx >= 0 && idx < y.size()) {
            y(idx) = wrap_angle(y(idx));
        }
    }
    Mat S = H * P * H.transpose() + R;
    Vec Sinv_y = S.ldlt().solve(y);
    return y.dot(Sinv_y);
}

} // namespace sentinel
