#include "ss/ss.h"
#include "ss/ss_config.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <limits>

namespace py = pybind11;

namespace util
{
    std::valarray<int> get_version() {
        return std::valarray<int>({
            ss_VERSION_MAJOR, ss_VERSION_MINOR, ss_VERSION_PATCH
        });
    }

    template <typename T, size_t NDim>
    struct as_span {
        static ss::ndspan<T, NDim> convert(const py::buffer_info& info);
    };

    template <typename T>
    struct as_span<T, 1> {
        static ss::ndspan<T, 1> convert(const py::buffer_info& info) {
            return ss::ndspan<T, 1>{
                { reinterpret_cast<T*>(info.ptr), info.size }, { info.size }
            };
        }
    };

    template <typename T>
    struct as_span<T, 2> {
        static ss::ndspan<T, 2> convert(const py::buffer_info& info)
        {
            if (info.ndim != 2) throw std::runtime_error("Unexpected number of dimensions. Expected 2 but got " + info.ndim);
            return ss::ndspan<T, 2>{
                { reinterpret_cast<T*>(info.ptr), info.size }, { info.shape[0], info.shape[1] }
            };
        }
    };

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

                if (bufA.ndim != 2) throw std::runtime_error("A must be a matrix");
                if (bufb.ndim != 1) throw std::runtime_error("b must be a vector");

                auto x = py::array_t<float>(bufA.shape[1]);
                auto result = solver.solve(
                    util::as_span<float, 2>::convert(bufA),
                    util::as_span<float, 1>::convert(bufb),
                    tolerance,
                    max_iterations,
                    util::as_span<float, 1>::convert(x.request()));

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