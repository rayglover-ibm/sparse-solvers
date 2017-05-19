#include "ss/ss.h"
#include "ss/ss_config.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace
{
    std::valarray<int> get_version() {
        return std::valarray<int>({
            ss_VERSION_MAJOR, ss_VERSION_MINOR, ss_VERSION_PATCH
        });
    }

    py::array_t<int> solve(py::array_t<float> A, py::array_t<float> x)
    {
        auto bufA = A.request(), bufx = x.request();

        if (bufA.ndim != 2) throw std::runtime_error("A must be a matrix");
        if (bufx.ndim != 1) throw std::runtime_error("x must be a vector");

        auto result = py::array_t<float>(1);
        return result;
    }
}

PYBIND11_PLUGIN(binding)
{
    py::module m("binding", "python binding example");

    m.def("version", &::get_version, "Module version");
    m.def("solve",   &::solve,       "Ax = b");

    return m.ptr();
}