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
    inline ss::ndspan<T, N> as_span(py::array_t<T, py::array::c_style>& arr)
    {
        if (arr.ndim() != N) throw std::runtime_error(
            "Unexpected number of dimensions. Expected " + std::to_string(N) + " but got " + std::to_string(arr.ndim()));

        std::array<size_t, N> shape;
        for (size_t d = 0; d < N; d++) shape[d] = arr.shape(d);

        return ss::as_span<N, T>(arr.mutable_data(), shape);
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

namespace builders
{
    template <typename T, typename Solver>
    void solve(py::class_<Solver>& solver)
    {
        solver.def("solve",
            [](ss::homotopy& solver,
               py::array_t<T, py::array::c_style> A,
               py::array_t<T, py::array::c_style> b,
               T tol = std::numeric_limits<T>::epsilon() * 10,
               uint32_t maxiter = 100)
            {
                auto x = py::array_t<T, py::array::c_style>(A.shape(1));
                kernelpp::maybe<ss::homotopy_report> result = solver.solve(
                    ss::as_span<2>(A), ss::as_span<1>(b), tol, maxiter, ss::as_span<1>(x));

                util::try_throw(result);
                return std::make_tuple(x, result.get<ss::homotopy_report>());
            },

            "Execute the solver on the given inputs.",
            py::arg("A").noconvert(),
            py::arg("b").noconvert(),
            py::arg("tolerance") = std::numeric_limits<T>::epsilon() * 10,
            py::arg("max_iterations") = 100);
    }
}

PYBIND11_PLUGIN(binding)
{
    py::module m("binding", "python binding example");
    m.def("version", &::util::get_version, LIB_NAME " version");

    /* homotopy report */
    py::class_<ss::homotopy_report>(m, "HomotopyReport")
        .def(py::init())
        .def_readwrite("iter", &ss::homotopy_report::iter)
        .def_readwrite("solution_error", &ss::homotopy_report::solution_error);

    /* homotopy solver */
    auto homotopy = py::class_<ss::homotopy>(m, "Homotopy").def(py::init());
    builders::solve<float>(homotopy);
    builders::solve<double>(homotopy);

    return m.ptr();
}