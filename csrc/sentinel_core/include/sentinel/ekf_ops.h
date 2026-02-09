#pragma once

#include "sentinel/common.h"

namespace sentinel {

// --- 2D polar EKF: state [x, vx, y, vy], meas [range, azimuth] ---
Vec2 ekf_h_polar(const Vec4& x);
Eigen::Matrix<double, 2, 4> ekf_H_polar(const Vec4& x);

// --- Bearing-only EKF: state [x, vx, y, vy], meas [azimuth] ---
double ekf_h_bearing(const Vec4& x);
Eigen::Matrix<double, 1, 4> ekf_H_bearing(const Vec4& x);

// --- 3D polar EKF: state [x, vx, y, vy, z, vz], meas [range, az, el] ---
Vec3 ekf_h_3d(const Eigen::Matrix<double, 6, 1>& x);
Eigen::Matrix<double, 3, 6> ekf_H_3d(const Eigen::Matrix<double, 6, 1>& x);

// --- Doppler EKF: state [x, vx, y, vy], meas [range, az, v_radial] ---
Vec3 ekf_h_doppler(const Vec4& x);
Eigen::Matrix<double, 3, 4> ekf_H_doppler(const Vec4& x);

// --- CA-EKF: state [x, vx, ax, y, vy, ay], meas [range, azimuth] ---
Vec2 ekf_h_ca_polar(const Eigen::Matrix<double, 6, 1>& x);
Eigen::Matrix<double, 2, 6> ekf_H_ca_polar(const Eigen::Matrix<double, 6, 1>& x);

// Generic EKF update with angular wrapping on specified indices.
// h_x = h(x) already computed, H = Jacobian already computed.
std::pair<Vec, Mat> ekf_update(
    const Vec& x, const Mat& P, const Mat& R,
    const Vec& z, const Vec& h_x, const Mat& H,
    const std::vector<int>& angular_indices);

// EKF gating with angular wrapping
double ekf_gating_distance(
    const Vec& x, const Mat& P, const Mat& R,
    const Vec& z, const Vec& h_x, const Mat& H,
    const std::vector<int>& angular_indices);

} // namespace sentinel
