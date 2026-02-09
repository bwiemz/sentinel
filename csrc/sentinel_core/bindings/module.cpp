#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_kalman(py::module_& m);
void bind_ekf(py::module_& m);
void bind_cost_matrix(py::module_& m);
void bind_jpda(py::module_& m);
void bind_physics(py::module_& m);
void bind_geodetic(py::module_& m);

PYBIND11_MODULE(_sentinel_core, m) {
    m.doc() = "SENTINEL C++ acceleration kernels";

    auto kalman = m.def_submodule("kalman", "Kalman filter operations");
    bind_kalman(kalman);

    auto ekf = m.def_submodule("ekf", "Extended Kalman filter operations");
    bind_ekf(ekf);

    auto cost = m.def_submodule("cost", "Cost matrix and distance functions");
    bind_cost_matrix(cost);

    auto jpda = m.def_submodule("jpda", "JPDA operations");
    bind_jpda(jpda);

    auto physics = m.def_submodule("physics", "Radar physics");
    bind_physics(physics);

    auto geodetic = m.def_submodule("geodetic", "WGS84 geodetic conversions");
    bind_geodetic(geodetic);
}
