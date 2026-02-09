#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sentinel/ekf_ops.h"

namespace py = pybind11;

void bind_ekf(py::module_& m) {
    // 2D polar
    m.def("h_polar", &sentinel::ekf_h_polar, py::arg("x"));
    m.def("H_polar", &sentinel::ekf_H_polar, py::arg("x"));

    // Bearing-only
    m.def("h_bearing", &sentinel::ekf_h_bearing, py::arg("x"));
    m.def("H_bearing", &sentinel::ekf_H_bearing, py::arg("x"));

    // 3D polar
    m.def("h_3d", &sentinel::ekf_h_3d, py::arg("x"));
    m.def("H_3d", &sentinel::ekf_H_3d, py::arg("x"));

    // Doppler
    m.def("h_doppler", &sentinel::ekf_h_doppler, py::arg("x"));
    m.def("H_doppler", &sentinel::ekf_H_doppler, py::arg("x"));

    // CA polar
    m.def("h_ca_polar", &sentinel::ekf_h_ca_polar, py::arg("x"));
    m.def("H_ca_polar", &sentinel::ekf_H_ca_polar, py::arg("x"));

    // Generic update/gating with angular wrapping
    m.def("update", &sentinel::ekf_update,
        py::arg("x"), py::arg("P"), py::arg("R"),
        py::arg("z"), py::arg("h_x"), py::arg("H"),
        py::arg("angular_indices"),
        "EKF update with angular wrapping on specified indices");

    m.def("gating_distance", &sentinel::ekf_gating_distance,
        py::arg("x"), py::arg("P"), py::arg("R"),
        py::arg("z"), py::arg("h_x"), py::arg("H"),
        py::arg("angular_indices"),
        "EKF gating distance with angular wrapping");
}
