#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include "sentinel/geodetic_ops.h"

namespace py = pybind11;

static constexpr double DEG2RAD = M_PI / 180.0;
static constexpr double RAD2DEG = 180.0 / M_PI;

void bind_geodetic(py::module_& m) {
    // All Python-facing functions use degrees; C++ internals use radians.

    m.def("geodetic_to_ecef",
        [](double lat_deg, double lon_deg, double alt_m) {
            auto [x, y, z] = sentinel::geodetic_to_ecef(
                lat_deg * DEG2RAD, lon_deg * DEG2RAD, alt_m);
            return py::make_tuple(x, y, z);
        },
        py::arg("lat_deg"), py::arg("lon_deg"), py::arg("alt_m"),
        "Geodetic (deg) to ECEF (meters)");

    m.def("ecef_to_geodetic",
        [](double x, double y, double z) {
            auto [lat_rad, lon_rad, alt] = sentinel::ecef_to_geodetic(x, y, z);
            return py::make_tuple(lat_rad * RAD2DEG, lon_rad * RAD2DEG, alt);
        },
        py::arg("x"), py::arg("y"), py::arg("z"),
        "ECEF (meters) to geodetic (deg)");

    m.def("geodetic_to_enu",
        [](double lat_deg, double lon_deg, double alt_m,
           double lat0_deg, double lon0_deg, double alt0_m) {
            auto [e, n, u] = sentinel::geodetic_to_enu(
                lat_deg * DEG2RAD, lon_deg * DEG2RAD, alt_m,
                lat0_deg * DEG2RAD, lon0_deg * DEG2RAD, alt0_m);
            return py::make_tuple(e, n, u);
        },
        py::arg("lat_deg"), py::arg("lon_deg"), py::arg("alt_m"),
        py::arg("lat0_deg"), py::arg("lon0_deg"), py::arg("alt0_m"),
        "Geodetic (deg) to ENU meters relative to reference");

    m.def("enu_to_geodetic",
        [](double e, double n, double u,
           double lat0_deg, double lon0_deg, double alt0_m) {
            auto [lat_rad, lon_rad, alt] = sentinel::enu_to_geodetic(
                e, n, u,
                lat0_deg * DEG2RAD, lon0_deg * DEG2RAD, alt0_m);
            return py::make_tuple(lat_rad * RAD2DEG, lon_rad * RAD2DEG, alt);
        },
        py::arg("e_m"), py::arg("n_m"), py::arg("u_m"),
        py::arg("lat0_deg"), py::arg("lon0_deg"), py::arg("alt0_m"),
        "ENU meters to geodetic (deg)");

    m.def("haversine_distance",
        [](double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
            return sentinel::haversine_distance(
                lat1_deg * DEG2RAD, lon1_deg * DEG2RAD,
                lat2_deg * DEG2RAD, lon2_deg * DEG2RAD);
        },
        py::arg("lat1_deg"), py::arg("lon1_deg"),
        py::arg("lat2_deg"), py::arg("lon2_deg"),
        "Great-circle distance in meters");

    m.def("geodetic_bearing",
        [](double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
            double bearing_rad = sentinel::geodetic_bearing(
                lat1_deg * DEG2RAD, lon1_deg * DEG2RAD,
                lat2_deg * DEG2RAD, lon2_deg * DEG2RAD);
            return bearing_rad * RAD2DEG;
        },
        py::arg("lat1_deg"), py::arg("lon1_deg"),
        py::arg("lat2_deg"), py::arg("lon2_deg"),
        "Initial bearing in degrees (0=North, clockwise)");
}
