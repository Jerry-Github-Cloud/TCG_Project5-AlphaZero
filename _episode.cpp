#include "episode.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;

PYBIND11_MODULE(_episode, m) {
    m.doc() = "class episode";
    py::class_<episode>(m, "episode")
        .def(py::init<>())
        .def("apply_action", &episode::apply_action)
        .def("__repr__",
            [](const episode& ep) {
                std::ostringstream out;
                out << ep;
                return out.str();
            });
}
