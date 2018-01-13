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
    /*  Forms the Cholesky factorization of a square
     *  matrix A if one exists.
     */
    template <typename T>
    class cholesky_decomposition
    {
      public:
        cholesky_decomposition(const ndspan<T, 2> A);

        ndspan<T, 2> l() const;

        /* Solves A*x == b for vector x. */
        void solve(const ndspan<T> b, ndspan<T> x) const;

        template <typename B, typename X>
        void solve(const B& b, X& x) const { solve(as_span(b), as_span(x)); }

        template <typename B>
        xt::xtensor<T, 1> solve(const B& b) const;

        bool isspd() { return _isspd; }

      private:
        xt::xtensor<T, 2> _l;
        bool _isspd;
    };
}

/* Definions --------------------------------------------------------------- */

namespace ss
{
    template <typename T>
    cholesky_decomposition<T>::cholesky_decomposition(const ndspan<T, 2> A)
        : _l({ dim<1>(A), dim<1>(A) })
        , _isspd{ true }
    {
        using namespace xt;
        const int64_t N = dim<1>(A);
        assert(dim<0>(A) > 0 && dim<0>(A) == N);

        _isspd = true;
        const T eps = std::numeric_limits<T>::epsilon();

        view(_l) = xt::tril(A); 

        for (int64_t j = 0; j < N; ++j) {
            auto v = xt::view(_l, xt::range(j, N), j);

            if (j > 0) {
                auto w = xt::view(_l, j, xt::range(0, j));
                auto m = xt::view(_l, xt::range(j, N), xt::range(0, j));
                blas::xgemv(CblasNoTrans, T{-1}, m, w, T{1}, v);
            }

            T ajj = std::sqrt(_l(j, j));
            if (ajj <= eps) {
                _isspd = false;
            }
            blas::xscal(T{1} / ajj, v);
        }
    }

    template <typename T>
    ndspan<T, 2> cholesky_decomposition<T>::l() const {
        return as_span(_l);
    }

    template <typename T>
    void cholesky_decomposition<T>::solve(const ndspan<T> b, ndspan<T> x) const
    {
        assert(dim<0>(b) == dim<0>(_l)
            && dim<0>(x) == dim<0>(_l));

        view(x) = b;
        
        blas::xtrsv(CblasLower, CblasNoTrans, CblasNonUnit, as_span(_l), x);
        blas::xtrsv(CblasLower, CblasTrans, CblasNonUnit, as_span(_l), x);
    }

    template <typename T>
    template <typename B>
    xt::xtensor<T, 1> cholesky_decomposition<T>::solve(const B& b) const
    {
        auto x = xt::xtensor<T, 1>::from_shape({ dim<0>(b) });
        solve(as_span(b), as_span(x));
        return x;
    }
}