#include <pybind11/pybind11.h>
#include "sentinel/physics_ops.h"

namespace py = pybind11;

void bind_physics(py::module_& m) {
    m.def("radar_snr", &sentinel::radar_snr,
        py::arg("rcs_m2"), py::arg("range_m"),
        py::arg("ref_range_m") = 10000.0,
        py::arg("ref_rcs_m2") = 10.0,
        py::arg("base_snr_db") = 15.0,
        "Radar SNR in dB");

    m.def("snr_to_pd", &sentinel::snr_to_pd,
        py::arg("snr_db"),
        "Convert SNR (dB) to detection probability");

    m.def("qi_snr_advantage_db", &sentinel::qi_snr_advantage_db,
        py::arg("n_signal"),
        "Quantum illumination SNR advantage in dB");

    m.def("qi_practical_pd", &sentinel::qi_practical_pd,
        py::arg("rcs_m2"), py::arg("range_m"), py::arg("n_signal"),
        py::arg("receiver_eff") = 0.5,
        py::arg("ref_range_m") = 10000.0,
        py::arg("ref_rcs_m2") = 10.0,
        py::arg("base_snr_db") = 15.0,
        "QI practical detection probability");
}
