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

#pragma once
#include "common.h"
#include "blas_prelude.h"

#include <assert.h>
#include <algorithm>

namespace ss {
namespace blas
{
    namespace detail
    {
        template <typename T, size_t N>
        T* data(ndspan<T, N>& view) {
            return view.raw_data() + view.raw_data_offset();
        }

        template <typename T, size_t N>
        const T* data(const ndspan<T, N>& view) {
            return view.raw_data() + view.raw_data_offset();
        }

        template <typename T>
        CBLAS_ORDER order(const ndspan<T, 2>& view) {
            return (stride<0>(view) == 1) ? CblasColMajor : CblasRowMajor;
        }

        template <typename T>
        size_t leading_stride(const ndspan<T, 1>& view) {
            return std::max(size_t(1), stride<0>(view));
        }

        template <typename T>
        size_t leading_stride(const ndspan<T, 2>& view)
        {
            /* xtensor strides are not fully compatible with cblas. Here we
               ensure the stride is at least equal to n when row-major, or m
               when col-major. */
            if (order<T>(view) == CblasRowMajor)
                return std::max(dim<1>(view), stride<0>(view));
            else
                return std::max(dim<0>(view), stride<1>(view));
        }
    }

    /* xnrm2 --------------------------------------------------------------- */

    inline double xnrm2(
        const blasint N, const double *X, const blasint incX) {
        return cblas_dnrm2(N, X, incX);
    }

    inline float xnrm2(
        const blasint N, const float *X, const blasint incX) {
        return cblas_snrm2(N, X, incX);
    }


    /* xgemv --------------------------------------------------------------- */

    inline void xgemv(
        const CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
        const double alpha, const double *a, const blasint lda, const double *x,
        const blasint incx, const double beta, double *y, const blasint incy)
    {
        cblas_dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    inline void xgemv(
        const CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
        const float alpha, const float *a, const blasint lda, const float *x,
        const blasint incx, const float beta, float *y, const blasint incy)
    {
        cblas_sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    template <typename T> void xgemv(
        CBLAS_TRANSPOSE trans, T alpha,
        const ndspan<T, 2> a,
        const ndspan<T, 1> x, T beta,
        ndspan<T, 1> y)
    {
        using namespace detail;

        xgemv(order(a), trans, dim<0>(a), dim<1>(a), alpha,
            data(a), leading_stride(a),
            data(x), leading_stride(x), beta,
            data(y), leading_stride(y));
    }


    /* xger ---------------------------------------------------------------- */

    inline void xger(
        const enum CBLAS_ORDER order, const blasint M, const blasint N,
        const double alpha, const double *X, const blasint incX,
        const double *Y, const blasint incY, double *A, const blasint lda)
    {
        cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }

    inline void xger(
        const enum CBLAS_ORDER order, const blasint M, const blasint N,
        const float alpha, const float *X, const blasint incX,
        const float *Y, const blasint incY, float *A, const blasint lda)
    {
        cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }

    template <typename T> void xger(
        T alpha,
        const ndspan<T, 1> x,
        const ndspan<T, 1> y,
        ndspan<T, 2> A)
    {
        using namespace detail;

        xger(order(A), dim<0>(A), dim<1>(A), alpha,
            data(x), leading_stride(x),
            data(y), leading_stride(y),
            data(A), leading_stride(A));
    }


    /* xdot ---------------------------------------------------------------- */

    inline double xdot(
        const blasint n, const double *x, const blasint incx, const double *y, const blasint incy)
    {
        return cblas_ddot(n, x, incx, y, incy);
    }

    inline float xdot(
        const blasint n, const float *x, const blasint incx, const float *y, const blasint incy)
    {
        return cblas_sdot(n, x, incx, y, incy);
    }

    template <typename T> T xdot(
        const ndspan<T, 1> x, const ndspan<T, 1> y)
    {
        using namespace detail;

        return xdot(dim<0>(x),
            data(x), leading_stride(x),
            data(y), leading_stride(y));
    }


    /* xscal --------------------------------------------------------------- */

    inline void xscal(
        const blasint N, const double alpha, double *X, const blasint incX) {
        cblas_dscal(N, alpha, X, incX);
    }

    inline void xscal(
        const blasint N, const float alpha, float *X, const blasint incX) {
        cblas_sscal(N, alpha, X, incX);
    }

    template <typename T> T xscal(
        T alpha, ndspan<T, 1> x)
    {
        using namespace detail;
        return xscal(dim<0>(x), alpha, leading_stride(x));
    }


    /* ixamax -------------------------------------------------------------- */

    inline size_t ixamax(
        const blasint n, const double *x, const blasint incx) {
        return cblas_idamax(n, x, incx);
    }

    inline size_t ixamax(
        const blasint n, const float *x, const blasint incx) {
        return cblas_isamax(n, x, incx);
    }
}
}