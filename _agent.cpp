#include "agent.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;

PYBIND11_MODULE(_agent, m) {
    m.doc() = "class agent";
    py::class_<AlphaZeroPlayer>(m, "AlphaZeroPlayer")
        .def(py::init<const std::string &>())
        .def("take_action", &AlphaZeroPlayer::take_action)
        .def("load_model", &AlphaZeroPlayer::load_model)
        ;
}
