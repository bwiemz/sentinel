#include "sentinel/kalman_ops.h"

namespace sentinel {

std::pair<Vec, Mat> kf_predict(
    const Vec& x, const Mat& P, const Mat& F, const Mat& Q)
{
    Vec x_new = F * x;
    Mat P_new = F * P * F.transpose() + Q;
    return {x_new, P_new};
}

std::pair<Vec, Mat> kf_update(
    const Vec& x, const Mat& P, const Mat& H, const Mat& R,
    const Vec& z)
{
    int n = static_cast<int>(x.size());
    Vec y = z - H * x;
    Mat S = H * P * H.transpose() + R;
    // Kalman gain: K = P * H' * S^{-1}
    // Solve S' * K' = (P * H')' => K = (S'.solve((P*H')'))'
    Mat PH = P * H.transpose();
    Mat K = S.transpose().ldlt().solve(PH.transpose()).transpose();
    Vec x_new = x + K * y;
    // Joseph form: P_new = (I - K*H)*P*(I - K*H)' + K*R*K'
    Mat I_KH = Mat::Identity(n, n) - K * H;
    Mat P_new = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    return {x_new, P_new};
}

double kf_gating_distance(
    const Vec& x, const Mat& P, const Mat& H, const Mat& R,
    const Vec& z)
{
    Vec y = z - H * x;
    Mat S = H * P * H.transpose() + R;
    // d² = y' * S^{-1} * y
    Vec Sinv_y = S.ldlt().solve(y);
    return y.dot(Sinv_y);
}

Mat kf_innovation_covariance(
    const Mat& P, const Mat& H, const Mat& R)
{
    return H * P * H.transpose() + R;
}

Vec kf_predicted_measurement(const Vec& x, const Mat& H) {
    return H * x;
}

} // namespace sentinel
