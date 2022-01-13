#include "action.h"
#include "board.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;

PYBIND11_MODULE(_action, m) {
    m.doc() = "class agent";
    py::class_<action>(m, "action")
        .def("apply", py::overload_cast<board&>(&action::apply, py::const_));
        // .def("apply", &action::apply<board&>())
}