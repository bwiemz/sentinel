#pragma once
#include "sentinel/common.h"
#include <tuple>

namespace sentinel {

// WGS84 ellipsoid constants
constexpr double WGS84_A  = 6378137.0;
constexpr double WGS84_F  = 1.0 / 298.257223563;
constexpr double WGS84_B  = WGS84_A * (1.0 - WGS84_F);
constexpr double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
constexpr double WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2);

// All angles in radians internally. Python bindings convert deg↔rad.

std::tuple<double,double,double> geodetic_to_ecef(double lat_rad, double lon_rad, double alt_m);
std::tuple<double,double,double> ecef_to_geodetic(double x, double y, double z);
std::tuple<double,double,double> geodetic_to_enu(double lat_rad, double lon_rad, double alt_m,
                                                  double lat0_rad, double lon0_rad, double alt0_m);
std::tuple<double,double,double> enu_to_geodetic(double e, double n, double u,
                                                  double lat0_rad, double lon0_rad, double alt0_m);
double haversine_distance(double lat1_rad, double lon1_rad,
                          double lat2_rad, double lon2_rad);
double geodetic_bearing(double lat1_rad, double lon1_rad,
                        double lat2_rad, double lon2_rad);

} // namespace sentinel
