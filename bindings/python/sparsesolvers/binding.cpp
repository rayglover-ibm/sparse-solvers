#include "ss/ss.h"
#include "ss/ss_config.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <limits>

namespace py = pybind11;

namespace ss
{
    template <size_t N, typename T>
    inline ss::ndspan<T, N> as_span(const py::buffer_info& info)
    {
        if (info.ndim != N) throw std::runtime_error(
            "Unexpected number of dimensions. Expected " + std::to_string(N) + " but got " + std::to_string(info.ndim));

        std::array<size_t, N> shape;
        for (size_t d = 0; d < N; d++) shape[d] = info.shape[d];

        return ss::as_span<N, T>(reinterpret_cast<T*>(info.ptr), info.size, shape);
    }
}

namespace util
{
    std::valarray<int> get_version() {
        return std::valarray<int>({
            ss_VERSION_MAJOR, ss_VERSION_MINOR, ss_VERSION_PATCH
        });
    }

    template<typename R>
    bool try_throw(const kernelpp::maybe<R>& r)
    {
        if (r.template is<kernelpp::error>())
            throw std::runtime_error(r.template get<kernelpp::error>().data());

        return false;
    }

    inline
    bool try_throw(const kernelpp::status& r)
    {
        if (r) throw std::runtime_error(r.get().data());
        return false;
    }
}

PYBIND11_PLUGIN(binding)
{
    py::module m("binding", "python binding example");

    m.def("version", &::util::get_version, LIB_NAME " version");

    py::class_<ss::homotopy_report>(m, "HomotopyReport")
        .def(py::init())
        .def_readwrite("iter", &ss::homotopy_report::iter)
        .def_readwrite("solution_error", &ss::homotopy_report::solution_error);

    py::class_<ss::homotopy>(m, "Homotopy")
        .def(py::init())
        .def("solve",
            [](ss::homotopy& solver,
               py::array_t<float> A,
               py::array_t<float> b,
               float tolerance = std::numeric_limits<float>::epsilon() * 10,
               uint32_t max_iterations = 100)
            {
                auto bufA = A.request(),
                     bufb = b.request();

                auto x = py::array_t<float>(bufA.shape[1]);
                auto result = solver.solve(
                    ss::as_span<2, float>(bufA),
                    ss::as_span<1, float>(bufb),
                    tolerance,
                    max_iterations,
                    ss::as_span<1, float>(x.request()));

                util::try_throw(result);
                ss::homotopy_report hr = result.get<ss::homotopy_report>();

                return std::make_tuple(x, std::move(hr));
            },

            "Execute the solver on the given inputs.",
            py::arg("A"),
            py::arg("b"),
            py::arg("tolerance") = std::numeric_limits<float>::epsilon() * 10,
            py::arg("max_iterations") = 100);

    return m.ptr();
}