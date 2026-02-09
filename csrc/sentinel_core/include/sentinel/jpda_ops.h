#pragma once

#include "sentinel/common.h"

namespace sentinel {

// Gaussian likelihood: N(innovation; 0, S)
double gaussian_likelihood(const Vec& innovation, const Mat& S);

// Beta coefficients: returns (betas, beta_0)
std::pair<Vec, double> compute_beta_coefficients(
    const Vec& likelihoods, double P_D, double lambda_FA);

// JPDA covariance update (spread-of-innovations + Joseph form)
Mat jpda_covariance_update(
    const Mat& P_prior,
    const Mat& K, const Mat& H, const Mat& R,
    const std::vector<Vec>& innovations,
    const Vec& betas, double beta_0,
    const Vec& combined_innovation);

} // namespace sentinel
