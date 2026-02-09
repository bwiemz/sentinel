#pragma once

namespace sentinel {

double radar_snr(double rcs_m2, double range_m,
                 double ref_range_m, double ref_rcs_m2, double base_snr_db);

double snr_to_pd(double snr_db);

double qi_snr_advantage_db(double n_signal);

double qi_practical_pd(double rcs_m2, double range_m, double n_signal,
                       double receiver_eff,
                       double ref_range_m, double ref_rcs_m2,
                       double base_snr_db);

} // namespace sentinel
