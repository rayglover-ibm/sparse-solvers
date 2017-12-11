/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#include "solvers/irls.h"

#include "linalg/common.h"
#include "linalg/blas_wrapper.h"
#include "linalg/cholesky_decomposition.h"

#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xio.hpp>

#include <assert.h>
#include <algorithm>

namespace ss
{
    template <typename T>
    void irls_newton(
        const qr_decomposition<T>& QR,
        const ndspan<T, 2> Q,
        const ndspan<T> y,
        const ndspan<T> w,
        ndspan<T> x)
    {
        xt::xtensor<T, 2> qw = ss::view(Q) * w;
        xt::xtensor<T, 2> qTqw = blas::xgemm(CblasTrans, CblasNoTrans, T{1}, Q, qw);

        ss::cholesky_decomposition<T> chol(as_span(qTqw));
        auto qTb = blas::xgemv(CblasTrans, T{1}, Q, y);

        auto s = xt::xtensor<T, 1>::from_shape(y.shape());
        chol.solve(qTb, s);

        auto t = blas::xgemv(CblasNoTrans, T{1}, Q, s);
        auto qTt = blas::xgemv(CblasTrans, T{1}, Q, t);

        QR.solve(qTt, x);
        ss::view(x) /= xt::sum(x);
    }

    template <typename T>
    irls_report run_solver(
        const qr_decomposition<T>& QR,
        const std::uint32_t max_iter,
        const T tolerance,
        const ndspan<T> y,
        ndspan<T> x)
    {
        auto Q = QR.q();

        assert(max_iter > 0
            && dim<0>(Q) == dim<1>(Q)
            && y.size() == dim<0>(Q)
            && x.size() == dim<0>(Q));

        size_t N = dim<0>(x);

        xt::xtensor<T, 1> w = xt::ones<T>({ N });
        xt::xtensor<T, 1> xsorted = xt::ones<T>({ N });

        T eps{ 1 };
        std::uint32_t iter{ 0u };

        do {
            /* update x */
            irls_newton(QR, as_span(Q), y, as_span(w), x);

            /* find the second largest abs value of x */
            view(xsorted) = xt::abs(x);
            std::nth_element(xsorted.begin(), xsorted.begin() + 1, xsorted.end(), std::greater<T>());

            /* update eps */
            eps = std::min(eps, xsorted(1) / T(N));

            /* update weights */
            view(w) = T{1} / xt::sqrt(x * x + eps * eps);

            iter++;
        }
        while (iter < max_iter && xsorted(1) > tolerance);

        return { iter, eps };
    }

    template <> kernelpp::variant<irls_report, error_code>
    solve_irls::op<compute_mode::CPU, float>(
        const qr_decomposition<float>& QR,
        const ndspan<float> y,
        float tolerance,
        std::uint32_t max_iterations,
        ndspan<float> x)
    {
        return run_solver<float>(QR, max_iterations, tolerance, y, x);
    }

    template <> kernelpp::variant<irls_report, error_code>
    solve_irls::op<compute_mode::CPU, double>(
        const qr_decomposition<double>& QR,
        const ndspan<double> y,
        double tolerance,
        std::uint32_t max_iterations,
        ndspan<double> x)
    {
        return run_solver<double>(QR, max_iterations, tolerance, y, x);
    }
}