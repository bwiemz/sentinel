#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sentinel {

// Standard aliases
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat2 = Eigen::Matrix2d;
using Mat4 = Eigen::Matrix4d;

// Angle wrapping to [-pi, pi]
// Note: std::fmod differs from Python's % for negatives, so we handle explicitly.
inline double wrap_angle(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

// Build CWNA process noise block for one axis:
// Q_block = sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
inline Eigen::Matrix2d cwna_block(double dt, double sigma_a) {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double s2 = sigma_a * sigma_a;
    Eigen::Matrix2d Q;
    Q << s2 * dt4 / 4.0, s2 * dt3 / 2.0,
         s2 * dt3 / 2.0, s2 * dt2;
    return Q;
}

} // namespace sentinel
