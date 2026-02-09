#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sentinel/kalman_ops.h"

namespace py = pybind11;

void bind_kalman(py::module_& m) {
    m.def("predict", &sentinel::kf_predict,
        py::arg("x"), py::arg("P"), py::arg("F"), py::arg("Q"),
        "KF predict step: returns (x_new, P_new)");

    m.def("update", &sentinel::kf_update,
        py::arg("x"), py::arg("P"), py::arg("H"), py::arg("R"), py::arg("z"),
        "KF update (Joseph form): returns (x_new, P_new)");

    m.def("gating_distance", &sentinel::kf_gating_distance,
        py::arg("x"), py::arg("P"), py::arg("H"), py::arg("R"), py::arg("z"),
        "Squared Mahalanobis gating distance");

    m.def("innovation_covariance", &sentinel::kf_innovation_covariance,
        py::arg("P"), py::arg("H"), py::arg("R"),
        "Innovation covariance S = H*P*H' + R");

    m.def("predicted_measurement", &sentinel::kf_predicted_measurement,
        py::arg("x"), py::arg("H"),
        "Predicted measurement z_pred = H*x");
}
