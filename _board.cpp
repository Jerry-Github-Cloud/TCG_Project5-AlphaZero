#include "board.h"
// #include <pybind/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;

PYBIND11_MODULE(_board, m) {
    m.doc() = "class board";
    // py::class_<board::point>(m, "Point")
    //     .def(py::init<>());
    py::class_<board>(m, "Board")
        .def(py::init<>())
        // .def("place", &board::place)
        .def("place", py::overload_cast<int, int, unsigned>(&board::place))
        // .def("place", &board::place, "Place x, y", py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("who").noconvert())
        // .def("place", py::overload_cast<const point& p, unsigned who = piece_type>(&board::place))
        .def("__repr__",
            [](const board &b) {
                std::ostringstream out;
                out << b;
                return out.str();
            })
        .def("observation_tensor", &board::observation_tensor)
        .def("check_liberty", &board::check_liberty)
        .def("rotate", &board::rotate);
}