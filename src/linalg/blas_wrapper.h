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
#include "linalg/common.h"
#include "linalg/blas_prelude.h"

#include <dlibxx.hxx>
#include <algorithm>
#include <memory>
#include <string>

namespace ss {
namespace blas
{
    class cblas final : dlibxx::handle_fascade
    {
        struct loader;

        static std::unique_ptr<cblas> m;
        static void configure();

        inline cblas(const std::string& path)
            : dlibxx::handle_fascade{ path.c_str() }
        {}

      public:
        op<decltype(::cblas_dnrm2)>  dnrm2 { this, "cblas_dnrm2" };
        op<decltype(::cblas_snrm2)>  snrm2 { this, "cblas_snrm2" };
        op<decltype(::cblas_dgemv)>  dgemv { this, "cblas_dgemv" };
        op<decltype(::cblas_sgemv)>  sgemv { this, "cblas_sgemv" };
        op<decltype(::cblas_sgemm)>  sgemm { this, "cblas_sgemm" };
        op<decltype(::cblas_dgemm)>  dgemm { this, "cblas_dgemm" };
        op<decltype(::cblas_dger)>   dger  { this, "cblas_dger" };
        op<decltype(::cblas_sger)>   sger  { this, "cblas_sger" };
        op<decltype(::cblas_ddot)>   ddot  { this, "cblas_ddot" };
        op<decltype(::cblas_sdot)>   sdot  { this, "cblas_sdot" };
        op<decltype(::cblas_dscal)>  dscal { this, "cblas_dscal" };
        op<decltype(::cblas_sscal)>  sscal { this, "cblas_sscal" };
        op<decltype(::cblas_idamax)> idamax{ this, "cblas_idamax" };
        op<decltype(::cblas_isamax)> isamax{ this, "cblas_isamax" };
        op<decltype(::cblas_strsm)>  strsm { this, "cblas_strsm" };
        op<decltype(::cblas_dtrsm)>  dtrsm { this, "cblas_dtrsm" };
        op<decltype(::cblas_strsv)>  strsv { this, "cblas_strsv" };
        op<decltype(::cblas_dtrsv)>  dtrsv { this, "cblas_dtrsv" };

        static cblas* get();
    };

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
        return cblas::get()->dnrm2(N, X, incX);
    }

    inline float xnrm2(
        const blasint N, const float *X, const blasint incX) {
        return cblas::get()->snrm2(N, X, incX);
    }


    /* xgemv --------------------------------------------------------------- */

    inline void xgemv(
        const enum CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
        const double alpha, const double *a, const blasint lda, const double *x,
        const blasint incx, const double beta, double *y, const blasint incy)
    {
        cblas::get()->dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    inline void xgemv(
        const enum CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
        const float alpha, const float *a, const blasint lda, const float *x,
        const blasint incx, const float beta, float *y, const blasint incy)
    {
        cblas::get()->sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
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

    template <typename T, typename A, typename X, typename Y> void xgemv(
        const enum CBLAS_TRANSPOSE trans, T alpha,
        const A& a,
        const X& x, T beta,
        Y& y)
    {
        xgemv(trans, alpha, as_span(a), as_span(x), beta, as_span(y));
    }

    template <typename T, typename A, typename X> xt::xtensor<T, 1> xgemv(
        const enum CBLAS_TRANSPOSE trans, T alpha,
        const A& a,
        const X& x)
    {
        auto y = xt::xtensor<T, 1>::from_shape({ trans == CblasNoTrans ? dim<0>(a) : dim<1>(a) });
        xgemv(trans, alpha, as_span(a), as_span(x), T{0}, as_span(y));
        return y;
    }


    /* xgemm --------------------------------------------------------------- */

    inline void xgemm(
        const CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
        const blasint m, const blasint n, const blasint k,
        const float alpha, const float *a, const blasint lda,
        const float *b, const blasint ldb, const float beta,
        float *c, const blasint ldc)
    {
        cblas::get()->sgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline void xgemm(
        const CBLAS_ORDER order,
        const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
        const blasint m, const blasint n, const blasint k,
        const double alpha, const double *a, const blasint lda,
        const double *b, const blasint ldb, const double beta,
        double *c, const blasint ldc)
    {
        cblas::get()->dgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    template <typename T> void xgemm(
        CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, T alpha,
        const ndspan<T, 2> a,
        const ndspan<T, 2> b, T beta,
        ndspan<T, 2> c)
    {
        using namespace detail;
        auto k = transA == CblasNoTrans ? dim<1>(a) : dim<0>(a);

        xgemm(order(a), transA, transB, dim<0>(c), dim<1>(c), k, alpha,
            data(a), leading_stride(a),
            data(b), leading_stride(b), beta,
            data(c), leading_stride(c));
    }

    template <typename T, typename A, typename B, typename C> void xgemm(
        CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, T alpha,
        const A& a,
        const B& b, T beta,
        C& c)
    {
        xgemm(transA, transB, alpha, as_span(a), as_span(b), beta, as_span(c));
    }

    template <typename T, typename A, typename B> xt::xtensor<T, 2> xgemm(
        CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, T alpha,
        const A& a,
        const B& b)
    {
        xt::xtensor<T, 2> c({
            transA == CblasNoTrans ? dim<0>(a) : dim<1>(a),
            transB == CblasNoTrans ? dim<1>(b) : dim<0>(b)
        });

        xgemm(transA, transB, alpha, as_span(a), as_span(b), T{0}, as_span(c));
        return c;
    }


    /* xger ---------------------------------------------------------------- */

    inline void xger(
        const enum CBLAS_ORDER order, const blasint M, const blasint N,
        const double alpha, const double *X, const blasint incX,
        const double *Y, const blasint incY, double *A, const blasint lda)
    {
        cblas::get()->dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }

    inline void xger(
        const enum CBLAS_ORDER order, const blasint M, const blasint N,
        const float alpha, const float *X, const blasint incX,
        const float *Y, const blasint incY, float *A, const blasint lda)
    {
        cblas::get()->sger(order, M, N, alpha, X, incX, Y, incY, A, lda);
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

    template <typename T, typename X, typename Y, typename A> void xger(
        T alpha, const X& x, const Y& y, A& a)
    {
        xger(alpha, as_span(x), as_span(y), as_span(a));
    }


    /* xdot ---------------------------------------------------------------- */

    inline double xdot(
        const blasint n, const double *x, const blasint incx,
        const double *y, const blasint incy)
    {
        return cblas::get()->ddot(n, x, incx, y, incy);
    }

    inline float xdot(
        const blasint n, const float *x, const blasint incx,
        const float *y, const blasint incy)
    {
        return cblas::get()->sdot(n, x, incx, y, incy);
    }

    template <typename T> T xdot(
        const ndspan<T, 1> x, const ndspan<T, 1> y)
    {
        using namespace detail;

        return xdot(dim<0>(x),
            data(x), leading_stride(x),
            data(y), leading_stride(y));
    }

    template <typename X, typename Y> auto xdot(const X& x, const Y& y) {
        return xdot(as_span(x), as_span(y));
    }


    /* xscal --------------------------------------------------------------- */

    inline void xscal(
        const blasint N, const double alpha, double *X, const blasint incX) {
        cblas::get()->dscal(N, alpha, X, incX);
    }

    inline void xscal(
        const blasint N, const float alpha, float *X, const blasint incX) {
        cblas::get()->sscal(N, alpha, X, incX);
    }

    template <typename T> void xscal(
        T alpha, ndspan<T, 1> x)
    {
        using namespace detail;
        xscal(dim<0>(x), alpha, data(x), leading_stride(x));
    }

    template <typename T, typename X> void xscal(T alpha, X& x) {
        xscal(alpha, as_span(x));
    }

    /* ixamax -------------------------------------------------------------- */

    inline size_t ixamax(
        const blasint n, const double *x, const blasint incx) {
        return cblas::get()->idamax(n, x, incx);
    }

    inline size_t ixamax(
        const blasint n, const float *x, const blasint incx) {
        return cblas::get()->isamax(n, x, incx);
    }


    /* xtrsm ---------------------------------------------------------------- */

    inline void xtrsm(
        const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
        const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
        const enum CBLAS_DIAG diag,
        const blasint m, const blasint n,
        const float alpha, const float *A, const blasint lda,
        float *B, const blasint ldb)
    {
        cblas::get()->strsm(order, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    }

    inline void xtrsm(
        const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
        const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
        const enum CBLAS_DIAG diag,
        const blasint m, const blasint n,
        const double alpha, const double *A, const blasint lda,
        double *B, const blasint ldb)
    {
        cblas::get()->dtrsm(order, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    }

    template <typename T> void xtrsm(
        const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
        const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
        const T alpha, const ndspan<T, 2> A, ndspan<T, 1> b)
    {
        using namespace detail;

        xtrsm(order(A), side, uplo, trans, diag,
            dim<0>(b), 1, alpha, data(A), leading_stride(A),
            data(b), 1);
    }


    /* xtrsv ---------------------------------------------------------------- */

    inline void xtrsv(
        const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
        const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
        const blasint n, const float *A, const blasint lda,
        float *x, const blasint incx)
    {
        cblas::get()->strsv(order, uplo, trans, diag, n, A, lda, x, incx);
    }

    inline void xtrsv(
        const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
        const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
        const blasint n, const double *A, const blasint lda,
        double *x, const blasint incx)
    {
        cblas::get()->dtrsv(order, uplo, trans, diag, n, A, lda, x, incx);
    }

    template <typename T> void xtrsv(
        const enum CBLAS_UPLO uplo,
        const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
        const ndspan<T, 2> A, ndspan<T, 1> b)
    {
        using namespace detail;

        xtrsv(order(A), uplo, trans, diag,
            dim<1>(A), data(A), leading_stride(A),
            data(b), 1);
    }
}
}