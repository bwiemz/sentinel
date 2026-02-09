#include "sentinel/jpda_ops.h"
#include <cmath>
#include <algorithm>

namespace sentinel {

double gaussian_likelihood(const Vec& innovation, const Mat& S) {
    int d = static_cast<int>(innovation.size());

    // Log-determinant via LDLT
    auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success) return 0.0;

    // Compute log(det(S)) from diagonal of L*D*L'
    double logdet = 0.0;
    auto D = ldlt.vectorD();
    for (int i = 0; i < D.size(); ++i) {
        if (D(i) <= 0.0) return 0.0;
        logdet += std::log(D(i));
    }

    double mahal = innovation.dot(ldlt.solve(innovation));
    double log_L = -0.5 * (d * std::log(2.0 * M_PI) + logdet + mahal);
    // Clamp to prevent underflow
    log_L = std::max(log_L, -700.0);
    return std::exp(log_L);
}

std::pair<Vec, double> compute_beta_coefficients(
    const Vec& likelihoods, double P_D, double lambda_FA)
{
    int n = static_cast<int>(likelihoods.size());
    double miss_term = lambda_FA * (1.0 - P_D);
    Vec det_terms = likelihoods * P_D;
    double c = det_terms.sum() + miss_term;

    if (c <= 0.0) {
        return {Vec::Zero(n), 1.0};
    }

    Vec betas = det_terms / c;
    double beta_0 = miss_term / c;
    return {betas, beta_0};
}

Mat jpda_covariance_update(
    const Mat& P_prior,
    const Mat& K, const Mat& H, const Mat& R,
    const std::vector<Vec>& innovations,
    const Vec& betas, double beta_0,
    const Vec& combined_innovation)
{
    int n = static_cast<int>(P_prior.rows());
    int m = static_cast<int>(innovations.empty() ? 0 : innovations[0].size());

    // Standard KF covariance (Joseph form without spread)
    Mat I_KH = Mat::Identity(n, n) - K * H;
    Mat P_std = I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();

    // Spread of innovations
    Mat P_spread = Mat::Zero(m, m);
    for (int j = 0; j < static_cast<int>(innovations.size()); ++j) {
        P_spread += betas(j) * (innovations[j] * innovations[j].transpose());
    }
    P_spread -= combined_innovation * combined_innovation.transpose();

    // Combined: P = beta_0 * P_prior + (1 - beta_0) * P_std + K * P_spread * K'
    Mat P_new = beta_0 * P_prior + (1.0 - beta_0) * P_std +
                K * P_spread * K.transpose();
    return P_new;
}

} // namespace sentinel
