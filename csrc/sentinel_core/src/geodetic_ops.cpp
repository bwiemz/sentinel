#include "sentinel/geodetic_ops.h"
#include <cmath>
#include <algorithm>

namespace sentinel {

std::tuple<double,double,double> geodetic_to_ecef(
    double lat_rad, double lon_rad, double alt_m)
{
    double sin_lat = std::sin(lat_rad);
    double cos_lat = std::cos(lat_rad);
    double sin_lon = std::sin(lon_rad);
    double cos_lon = std::cos(lon_rad);

    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    double X = (N + alt_m) * cos_lat * cos_lon;
    double Y = (N + alt_m) * cos_lat * sin_lon;
    double Z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat;
    return {X, Y, Z};
}

std::tuple<double,double,double> ecef_to_geodetic(double x, double y, double z)
{
    double lon = std::atan2(y, x);
    double p = std::sqrt(x * x + y * y);

    // Bowring's parametric latitude
    double theta = std::atan2(z * WGS84_A, p * WGS84_B);
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);

    double lat = std::atan2(
        z + WGS84_EP2 * WGS84_B * sin_theta * sin_theta * sin_theta,
        p - WGS84_E2 * WGS84_A * cos_theta * cos_theta * cos_theta
    );

    double sin_lat = std::sin(lat);
    double cos_lat = std::cos(lat);
    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

    double alt;
    if (std::abs(cos_lat) > 1e-10) {
        alt = p / cos_lat - N;
    } else {
        alt = std::abs(z) / std::abs(sin_lat) - N * (1.0 - WGS84_E2);
    }

    return {lat, lon, alt};
}

std::tuple<double,double,double> geodetic_to_enu(
    double lat_rad, double lon_rad, double alt_m,
    double lat0_rad, double lon0_rad, double alt0_m)
{
    auto [x, y, z] = geodetic_to_ecef(lat_rad, lon_rad, alt_m);
    auto [x0, y0, z0] = geodetic_to_ecef(lat0_rad, lon0_rad, alt0_m);
    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;

    double sin_lat0 = std::sin(lat0_rad);
    double cos_lat0 = std::cos(lat0_rad);
    double sin_lon0 = std::sin(lon0_rad);
    double cos_lon0 = std::cos(lon0_rad);

    double east  = -sin_lon0 * dx + cos_lon0 * dy;
    double north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz;
    double up    =  cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz;

    return {east, north, up};
}

std::tuple<double,double,double> enu_to_geodetic(
    double e, double n, double u,
    double lat0_rad, double lon0_rad, double alt0_m)
{
    double sin_lat0 = std::sin(lat0_rad);
    double cos_lat0 = std::cos(lat0_rad);
    double sin_lon0 = std::sin(lon0_rad);
    double cos_lon0 = std::cos(lon0_rad);

    // Inverse rotation: ENU -> ECEF delta
    double dx = -sin_lon0 * e - sin_lat0 * cos_lon0 * n + cos_lat0 * cos_lon0 * u;
    double dy =  cos_lon0 * e - sin_lat0 * sin_lon0 * n + cos_lat0 * sin_lon0 * u;
    double dz =                  cos_lat0 * n             + sin_lat0 * u;

    auto [x0, y0, z0] = geodetic_to_ecef(lat0_rad, lon0_rad, alt0_m);
    return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz);
}

double haversine_distance(
    double lat1_rad, double lon1_rad,
    double lat2_rad, double lon2_rad)
{
    double dlat = lat2_rad - lat1_rad;
    double dlon = lon2_rad - lon1_rad;
    double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
               std::cos(lat1_rad) * std::cos(lat2_rad) *
               std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
    double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    double R = (WGS84_A + WGS84_B) / 2.0;  // Mean radius
    return R * c;
}

double geodetic_bearing(
    double lat1_rad, double lon1_rad,
    double lat2_rad, double lon2_rad)
{
    double dlon = lon2_rad - lon1_rad;
    double x = std::sin(dlon) * std::cos(lat2_rad);
    double y = std::cos(lat1_rad) * std::sin(lat2_rad) -
               std::sin(lat1_rad) * std::cos(lat2_rad) * std::cos(dlon);
    double bearing = std::atan2(x, y);
    // Normalize to [0, 2*pi)
    if (bearing < 0) bearing += 2.0 * M_PI;
    return bearing;
}

} // namespace sentinel
