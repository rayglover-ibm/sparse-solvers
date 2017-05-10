#pragma once
#include "blas_prelude.h"

namespace ss {
namespace blas 
{
    inline double xnrm2(
        const blasint N, const double *X, const blasint incX) {
        return cblas_dnrm2(N, X, incX);
    }

    inline float xnrm2(
        const blasint N, const float *X, const blasint incX) {
        return cblas_snrm2(N, X, incX);
    }

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

    inline void xcopy(
        const blasint n, const double *x, const blasint incx, double *y, const blasint incy)
    {
        cblas_dcopy(n, x, incx, y, incy);
    }

    inline void xcopy(
        const blasint n, const float *x, const blasint incx, float *y, const blasint incy)
    {
        cblas_scopy(n, x, incx, y, incy);
    }

    inline void xscal(
        const blasint N, const double alpha, double *X, const blasint incX) {
        cblas_dscal(N, alpha, X, incX);
    }

    inline void xscal(
        const blasint N, const float alpha, float *X, const blasint incX) {
        cblas_sscal(N, alpha, X, incX);
    }

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