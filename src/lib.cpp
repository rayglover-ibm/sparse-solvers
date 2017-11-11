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

#include <kernelpp/kernel_invoke.h>

#include "ss/ss.h"
#include "solvers/homotopy.h"

#include "linalg/common.h"
#include "linalg/norms.h"
#include "linalg/blas_wrapper.h"

namespace ss
{
    /* Homotopy solver ----------------------------------------------------- */

    homotopy_policy::homotopy_policy() = default;
    homotopy_policy::homotopy_policy(homotopy_policy&&) = default;
    homotopy_policy::~homotopy_policy() = default;

    kernelpp::maybe<ss::homotopy_report> homotopy_policy::run(
        const ndspan<float, 2>& A,
        const ndspan<float> y,
        float tol, uint32_t maxiter,
        ndspan<float> x)
    {
        return kernelpp::run<solve_homotopy>(A, y, tol, maxiter, x);
    }

    kernelpp::maybe<ss::homotopy_report> homotopy_policy::run(
        const ndspan<double, 2>& A,
        const ndspan<double> y,
        double tol, uint32_t maxiter,
        ndspan<double> x)
    {
        return kernelpp::run<solve_homotopy>(A, y, tol, maxiter, x);
    }

    /* Utils --------------------------------------------------------------- */

    namespace detail
    {
        template <typename T>
        void reconstruct_signal(
            const ndspan<T, 2> A, const ndspan<T> x, ndspan<T> y)
        {
            assert (dim<1>(A) == x.size()
                &&  dim<0>(A) == y.size());

            size_t m = dim<0>(A), n = dim<1>(A);

            blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
                A.storage_cbegin(), n,
                x.storage_cbegin(), 1, 0.0,
                y.storage_begin(), 1);
        }
    }

    void reconstruct_signal(
        const ndspan<float, 2> A, const ndspan<float> x, ndspan<float> y) {
        detail::reconstruct_signal(A, x, y);
    }

    void reconstruct_signal(
        const ndspan<double, 2> A, const ndspan<double> x, ndspan<double> y) {
        detail::reconstruct_signal(A, x, y);
    }

    void norm_l1(ndspan<float, 2> A) {
        l1<float>(A);
    }

    void norm_l1(ndspan<double, 2> A) {
        l1<double>(A);
    }
}