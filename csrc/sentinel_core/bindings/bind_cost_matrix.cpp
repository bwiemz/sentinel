#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sentinel/cost_matrix.h"

namespace py = pybind11;

void bind_cost_matrix(py::module_& m) {
    m.def("iou_bbox",
        [](py::array_t<double> a, py::array_t<double> b) {
            auto a_buf = a.unchecked<1>();
            auto b_buf = b.unchecked<1>();
            return sentinel::iou_bbox(a_buf.data(0), b_buf.data(0));
        },
        py::arg("bbox_a"), py::arg("bbox_b"),
        "IoU between two [x1,y1,x2,y2] bounding boxes");

    m.def("track_to_track_mahalanobis",
        &sentinel::track_to_track_mahalanobis,
        py::arg("pos1"), py::arg("cov1"),
        py::arg("pos2"), py::arg("cov2"),
        "Track-to-track squared Mahalanobis distance");
}
