#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sentinel/jpda_ops.h"

namespace py = pybind11;

void bind_jpda(py::module_& m) {
    m.def("gaussian_likelihood", &sentinel::gaussian_likelihood,
        py::arg("innovation"), py::arg("S"),
        "Gaussian likelihood N(innovation; 0, S)");

    m.def("compute_beta_coefficients", &sentinel::compute_beta_coefficients,
        py::arg("likelihoods"), py::arg("P_D"), py::arg("lambda_FA"),
        "Beta coefficients: returns (betas, beta_0)");

    m.def("jpda_covariance_update", &sentinel::jpda_covariance_update,
        py::arg("P_prior"), py::arg("K"), py::arg("H"), py::arg("R"),
        py::arg("innovations"), py::arg("betas"), py::arg("beta_0"),
        py::arg("combined_innovation"),
        "JPDA covariance update with spread-of-innovations");
}
