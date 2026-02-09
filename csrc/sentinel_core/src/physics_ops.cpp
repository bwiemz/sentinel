#include "sentinel/physics_ops.h"
#include <cmath>
#include <algorithm>

namespace sentinel {

double radar_snr(double rcs_m2, double range_m,
                 double ref_range_m, double ref_rcs_m2, double base_snr_db)
{
    if (range_m <= 0.0 || rcs_m2 <= 0.0) return -100.0;
    double ratio_rcs = rcs_m2 / ref_rcs_m2;
    double ratio_range = ref_range_m / range_m;
    double ratio_range4 = ratio_range * ratio_range * ratio_range * ratio_range;
    return base_snr_db + 10.0 * std::log10(ratio_rcs * ratio_range4);
}

double snr_to_pd(double snr_db) {
    if (snr_db < -50.0) return 0.0;
    double snr_linear = std::pow(10.0, snr_db / 10.0);
    double threshold = 20.0;
    return std::min(1.0, 1.0 - std::exp(-snr_linear / threshold));
}

double qi_snr_advantage_db(double n_signal) {
    if (n_signal <= 0.0) return 40.0;  // Cap
    double ratio = 4.0 / n_signal;
    return std::min(40.0, 10.0 * std::log10(ratio));
}

double qi_practical_pd(double rcs_m2, double range_m, double n_signal,
                       double receiver_eff,
                       double ref_range_m, double ref_rcs_m2,
                       double base_snr_db)
{
    double snr = radar_snr(rcs_m2, range_m, ref_range_m, ref_rcs_m2, base_snr_db);
    double qi_boost = qi_snr_advantage_db(n_signal) * receiver_eff;
    return snr_to_pd(snr + qi_boost);
}

} // namespace sentinel
