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
#include <xtensor/xsort.hpp>

#include <assert.h>
#include <algorithm>

namespace ss
{
    template <typename T, typename E>
    void threshold(E& x, T threshmin, T newval)
    {
        for (T& val : x) {
            if (val < threshmin) {
                val = newval;
            }
        }
    }

    template <typename T>
    bool irls_newton(
        const ndspan<T, 2> R,
        const ndspan<T, 2> Q,
        const ndspan<T> y,
        const ndspan<T> w,
        ndspan<T> x)
    {
        xt::xtensor<T, 2> qw = ss::view(Q) * w;
        xt::xtensor<T, 2> qTqw = blas::xgemm(CblasTrans, CblasNoTrans, T{1}, Q, qw);

        ss::cholesky_decomposition<T> chol(as_span(qTqw));
        if (!chol.isspd()) { return false; }

        auto qTb = blas::xgemv(CblasTrans, T{1}, Q, y);
        auto s = xt::xtensor<T, 1>::from_shape({ dim<0>(qTb) });
        chol.solve(qTb, s);

        auto t = blas::xgemv(CblasNoTrans, T{1}, Q, s);
        auto qTt = blas::xgemv(CblasTrans, T{1}, Q, t);

        blas::xtrsm(CblasUpper, CblasNoTrans, CblasNonUnit, T{1},
            R, ss::as_span(qTt));

        ss::view(x) = qTt;

        return true;
    }

    template <typename T>
    irls_report run_solver(
        const qr_decomposition<T>& QR,
        const std::uint32_t max_iter,
        const T tolerance,
        const ndspan<T> y,
        ndspan<T> x)
    {
        const T p{ 0.9 };

        auto Q = QR.q();
        auto R = QR.r();

        assert(max_iter > 0
            && y.size() == dim<0>(Q)
            && x.size() == dim<1>(Q));

        size_t N = dim<0>(x);

        /* initialize the result */
        view(x) = T{ 0 };

        xt::xtensor<T, 1> w = xt::ones<T>({ N });
        xt::xtensor<T, 1> xnext = xt::ones<T>({ N });

        std::uint32_t iter{ 0u };
        bool spd_error{ false };
        T abstol{ 1.0 };
        T eps{ 1 };

        do {
            /* update x */
            if (!irls_newton(as_span(R), as_span(Q), y, as_span(w), as_span(xnext))) {
                spd_error = true;
                break;
            }

            /* use tolerance as a proportion of the max value of x */
            abstol = xt::amax(xnext)() * tolerance;

            /* threshold and normalize */
            threshold(xnext, abstol, T{ 0 });
            view(x) = xnext;

            /* find the second largest abs value of x */
            std::nth_element(xnext.begin(), xnext.begin() + 1, xnext.end(), std::greater<T>());

            /* update eps */
            eps = std::min(eps, xnext(1) / T(N));

            /* update weights and normalize */
            view(w) = xt::pow(x * x + eps, (p / 2.0) - 1.0);
            view(w) /= xt::sum(w);

            iter++;
        }
        while (iter < max_iter && xnext(1) > abstol);

        /* finally, normalize x */
        view(x) /= xt::sum(x);

        return { iter, eps, spd_error };
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