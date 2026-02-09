#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sentinel/batch_ops.h"

namespace py = pybind11;

void bind_batch(py::module_& m) {
    m.def("batch_kf_gating_matrix",
        &sentinel::batch_kf_gating_matrix,
        py::arg("X"), py::arg("P"), py::arg("Z"),
        py::arg("H"), py::arg("R"), py::arg("gate"),
        "Build (T,D) Mahalanobis gating cost matrix for KF tracks");

    m.def("batch_iou_matrix",
        &sentinel::batch_iou_matrix,
        py::arg("bboxes_a"), py::arg("bboxes_b"),
        "Build (T,D) IoU matrix between two sets of bounding boxes");

    m.def("batch_camera_cost_matrix",
        &sentinel::batch_camera_cost_matrix,
        py::arg("X"), py::arg("P"), py::arg("Z"),
        py::arg("H"), py::arg("R"),
        py::arg("bboxes_a"), py::arg("bboxes_b"),
        py::arg("alpha"), py::arg("gate"),
        "Build combined camera cost matrix (alpha*maha + (1-alpha)*(1-IoU))");

    m.def("batch_ekf_gating_matrix",
        &sentinel::batch_ekf_gating_matrix,
        py::arg("states"), py::arg("covariances"),
        py::arg("Z"), py::arg("H_jacobians"),
        py::arg("h_predictions"), py::arg("R"),
        py::arg("angular_indices"), py::arg("gate"),
        "Build (T,D) gating cost matrix for EKF tracks with per-track Jacobians");

    m.def("batch_gaussian_likelihood",
        &sentinel::batch_gaussian_likelihood,
        py::arg("innovations"), py::arg("S_inv"),
        py::arg("log_det_S"),
        "Compute Gaussian likelihood for N innovations against shared S");

    m.def("batch_kf_predict",
        &sentinel::batch_kf_predict,
        py::arg("X"), py::arg("P"),
        py::arg("F"), py::arg("Q"),
        "Predict T KF states in one call, returns (X_pred, P_pred)");
}
