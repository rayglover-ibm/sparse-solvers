#include "ss/ss.h"
#include "ss/ss_config.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <limits>

namespace py = pybind11;

namespace ss
{
    template <typename T>
    inline bool is_c_contiguous(py::array_t<T>& arr)
    {
        return py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ ==
            (arr.flags() & py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_);
    }

    template <size_t N, typename T>
    inline ss::ndspan<T, N> as_span(py::array_t<T>& arr)
    {
        if (arr.ndim() != N) throw std::runtime_error(
            "Unexpected number of dimensions. Expected " + std::to_string(N) + " but got " + std::to_string(arr.ndim()));

        std::array<size_t, N> shape;
        std::array<size_t, N> strides;

        for (size_t d = 0; d < N; d++)
        {
            shape[d] = arr.shape(d);
            strides[d] = arr.strides(d) / sizeof(T);
        }

        return ss::as_span<N, T>(arr.mutable_data(), shape, strides);
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
    using namespace ss;
    
    template <typename Policy>
    struct py_solver
    {
        std::array<size_t, 2> m_shape;
        
        kernelpp::variant<
            solver<float, Policy>, solver<double, Policy>
            > m;
    };

    template <typename T, typename P>
    void init(py::class_<py_solver<P>>& cls)
    {
        cls.def(py::init([](py::array_t<T> A_) {
            auto A = as_span<2>(A_);
            return new py_solver<P>{ A.shape(), solver<T, homotopy_policy>(A) };
        }));
    }
    
    template <typename T, typename P>
    void solve(py::class_<py_solver<P>>& cls)
    {
        cls.def("solve",
            [](py_solver<P>& instance,
               py::array_t<T> b,
               T tol = std::numeric_limits<T>::epsilon() * 10,
               uint32_t maxiter = 100)
            {
                py::array_t<T> x(instance.m_shape[1]);

                auto& s = instance.m.template get<solver<T, P>>();
                auto result = s.solve(as_span<1>(b), tol, maxiter, as_span<1>(x));

                util::try_throw(result);
                return std::make_tuple(x, result.template get<homotopy_report>());
            },

            "Execute the solver on the given inputs.",
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
    auto homotopy = py::class_< builders::py_solver<ss::homotopy_policy>>(m, "Homotopy");

    builders::init<float>(homotopy);
    builders::init<double>(homotopy);
    builders::solve<float>(homotopy);
    builders::solve<double>(homotopy);

    return m.ptr();
}