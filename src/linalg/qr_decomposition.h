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

#include "linalg/blas_wrapper.h"
#include "linalg/common.h"

#include <ss/ndspan.h>
#include <xtensor/xtensor.hpp>
#include <cassert>

namespace ss
{
    /*  Forms the QR factorization of a general m-by-n matrix A by
     *  Householder reflection.
     *
     *  The routine does not form the matrix Q explicitly. Instead, Q is
     *  represented as a product of min(m, n) elementary reflectors.
     *  Routines are provided to work with Q in this representation.
     */
    template <typename T>
    class qr_decomposition
    {
      public:
        qr_decomposition(const ndspan<T, 2> A);

        xt::xtensor<T, 2> q() const;
        xt::xtensor<T, 2> r() const;

        /*  Solves A*x == b for vector x.
         *
         *  Then this function finds the least squares solution to the
         *  equation A*X = B and returns X.  X has the following properties:
         *    - X is the matrix that minimizes the two norm of A*X-B, i.e. it
         *      minimizes sum(squared(A*X - B)).
         */
        void solve(const ndspan<T> b, ndspan<T> x) const;

        template <typename B, typename X>
        void solve(const B& b, X& x) const { solve(as_span(b), as_span(x)); }

      private:
        xt::xtensor<T, 2> _qr;
        xt::xtensor<T, 1> _rdiag;
    };
}

/* Definions --------------------------------------------------------------- */

namespace ss{ namespace detail
{
    /* Store and extract houseolder vectors (colmns of A)
     * in to temporary storage, and compute 2-norm of the vector. */
    template <typename T>
    void next_householder(
        xt::xtensor<T, 2>& A, int64_t n, xt::xtensor<T, 1>& hvec, T& nrm2)
    {
        auto M = dim<0>(A);
        auto N = dim<1>(A);

        if (n < N) {
            nrm2 = T{0};
            if (n > 0) { A(n-1, n-1) = hvec(n-1); }

            for (int64_t m = n; m < M; m++) {
                T& hm = hvec(m);
                if (n > 0) { A(m, n-1) = hm; }

                hm = A(m, n);
                nrm2 = std::hypot(nrm2, hm);
            }
        } else {
            /* restore the current householder (n-1) to A */
            for (int64_t m = n-1; m < M; m++)
                A(m, n-1) = hvec(m);
        }
    }
}}

namespace ss
{
    template <typename T>
    qr_decomposition<T>::qr_decomposition(const ndspan<T, 2> A)
        : _qr(A)
        , _rdiag({ dim<1>(A) }, xt::layout_type::row_major)
    {
        const int64_t M = dim<0>(A);
        const int64_t N = dim<1>(A);

        assert(M > 0 && N > 0 && M >= N);

        auto s    = xt::xtensor<T, 1>({ unsigned(N) }, T{0});
        auto hvec = xt::xtensor<T, 1>({ unsigned(M) }, T{0});

        int64_t m, n;
        T nrm2;

        for (int64_t k = 0; k < N; k++) {
            /* Compute 2-norm of k-th column without under/overflow. */
            detail::next_householder(_qr, k, hvec, nrm2);
            if (nrm2 != T{0}) {
                if (hvec(k) < 0) { nrm2 = -nrm2; }

                /* Form k-th Householder vector. (lower triangular) */
                xt::view(hvec, xt::range(k, M)) /= nrm2;
                hvec(k) += T{1};

                /* Apply transformation to remaining columns. */
                view(s) = T(0);
                for (m = k; m < M; m++) {
                    for (n = k + 1; n < N; n++) {
                        s(n) += hvec(m) * _qr(m, n);
                    }
                }

                view(s) /= -hvec(k);
                for (m = k; m < M; m++) {
                    for (n = k + 1; n < N; n++) {
                        _qr(m, n) += hvec(m) * s(n);
                    }
                }
            }
            _rdiag(k) = -nrm2;
        }
        detail::next_householder(_qr, N, hvec, nrm2);
    }

    template <typename T>
    xt::xtensor<T, 2> qr_decomposition<T>::q() const
    {
        auto q = xt::xtensor<T, 2>::from_shape(_qr.shape());

        const int64_t M = dim<0>(_qr);
        const int64_t N = dim<1>(_qr);

        int64_t m=0, n=0, k=0;

        for (k = N-1; k >= 0; k--) {
            /* current column in qr */
            const auto qrcol = xt::view(_qr, xt::all(), k);
            /* initilize column in q */
            xt::view(q, xt::all(), k) = T{0};
            q(k, k) = T{1};

            for (n = k; n < N; n++) {
                if (qrcol(k) != T{0}) {
                    T s{0};
                    for (m = k; m < M; m++) {
                        s += qrcol(m) * q(m, n);
                    }

                    s = -s / qrcol(k);

                    for (m = k; m < M; m++) {
                        q(m, n) += s * qrcol(m);
                    }
                }
            }
        }

        return q;
    }

    template <typename T>
    xt::xtensor<T, 2> qr_decomposition<T>::r() const
    {
        const auto N = dim<1>(_qr);
        auto r = xt::xtensor<T, 2>({ N, N }, T{0});

        for (int64_t m = 0; m < N; m++) {
            r(m, m) = _rdiag(m);
            for (int64_t n = m + 1; n < N; n++) {
                r(m, n) = _qr(m, n);
            }
        }

        return r;
    }

    template <typename T>
    void qr_decomposition<T>::solve(const ndspan<T> b, ndspan<T> x) const
    {
        const int64_t M = dim<0>(_qr);
        const int64_t N = dim<1>(_qr);

        assert(M == dim<0>(b) && N == dim<0>(x));

        /* Compute Y = transpose(Q)*B */
        xt::xtensor<T, 1> s = b;

        for (int64_t n = 0; n < N; n++) {
            T w{0};
            for (int64_t m = n; m < M; m++) {
                /* lower triangular */
                w += _qr(m, n) * s(m);
            }

            w = -w / _qr(n, n);

            for (int64_t m = n; m < M; m++) {
                s(m) += w * _qr(m, n);
            }
        }

        /* Solve R*X = Y */
        for (int64_t n = N - 1; n >= 0; n--) {
            s(n) /= _rdiag(n);

            for (int64_t m = 0; m < n; m++) {
                s(m) -= s(n) * _qr(m, n);
            }
        }
        /* x becomes s[0..N] */
        view(x) = xt::view(s, xt::range(0, N));
    }
}