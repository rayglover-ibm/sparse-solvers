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

#include "blas_wrapper.h"
#include "linalg/common.h"

#include <ss/ndspan.h>
#include <xtensor/xview.hpp>

#include <kernelpp/kernel.h>
#include <kernelpp/kernel_invoke.h>

#include <memory>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <cstring>

namespace ss
{
    /*  Maintains the (A^T * A) inverse of column-wise subset
     *  of a matrix A.
     *
     *  Refer to ./docs/algorithms/online-matrix-inverse for more information
     */
    template <typename T>
    class online_column_inverse
    {
      public:
        online_column_inverse(size_t m, size_t capacity = 1);

        /* returns a view of the inverse */
        const mat_view<T> inverse();

        /* Inserts a column in to the inverse at the given index. */
        template <typename It>
        void insert(const size_t column_idx, It begin, It end);

        /* Removes the column from the inverse at the given index. */
        void remove(size_t column_idx);

        /* returns the size of the subset */
        const size_t N() { return _n; }

      private:
        /* A_gamma transposed */
        aligned_vector<T> _A_sub_t;
        /* the inverse of A_gamma */
        aligned_vector<T> _inv;
        /* fixed size of columns in the inverse */
        const size_t _m;
        /* number of colunms currently in the inverse */
        size_t _n;
    };
}

/* Implementation ---------------------------------------------------------- */

namespace ss { namespace detail
{
    using namespace ::kernelpp;

    /*  Permutes the given square matrix `A` such that the row and column `src`
     *  is moved to `dest`, with intermediate rows and columns shifted to
     *  account for this movement.
     */
    KERNEL_DECL(square_permute, compute_mode::CPU)
    {
        template <compute_mode M, typename T>
        static void op(mat_view<T> A, const size_t src, const size_t dest)
        {
            using rit = std::reverse_iterator<T*>;
            assert(dim<0>(A) == dim<1>(A));

            T* ptr = &A(0, 0);
            const ptrdiff_t N = dim<1>(A), srci = src, desti = dest;

            if (N == 1 || desti == srci) {
                return;
            }
            else if (desti > srci) {
                /* traverse forwards over all rows */
                for (ptrdiff_t m = 0, i = 0; m < N; m++, i += N) {
                    T* row = &ptr[i];

                    if (m >= srci && m < desti) {
                        /* single row rotation */
                        std::rotate(row, &row[N], &row[N + N]);
                    }
                    /* column rotation */
                    std::rotate(&row[srci], &row[srci + 1], &row[desti + 1]);
                }
            }
            else {
                /* traverse backwards over all rows */
                for (ptrdiff_t m = N-1, i = (N * N); m >= 0; m--, i -= N) {
                    T* row = &ptr[i - N];

                    if (m <= srci && m > desti) {
                        /* single row rotation */
                        std::rotate(&row[-N], row, &row[N]);
                    }
                    /* column rotation */
                    std::rotate(rit(&row[srci + 1]), rit(&row[srci]), rit(&row[desti]));
                }
            }
        }
    };

    /*  Removes the last row and column from the given matrix
     *  of M rows and N columns
     */
    KERNEL_DECL(erase_last_rowcol, compute_mode::CPU)
    {
        template <compute_mode, typename T> static void op(
            aligned_vector<T>& A, const size_t M, const size_t N)
        {
            assert(A.size() == M * N);
            const size_t LEN = (N - 1) * sizeof(T);

            /* traversing forwards, shift values left such
             * that the last column of each row is removed */
            for (size_t i = 0, dest = 0;
                 i < N * (M-1);
                 i += N, dest += N-1)
            {
                std::memmove(&A[dest], &A[i], LEN);
            }

            A.erase(A.end() - (N + M - 1), A.end());
        }
    };

    /*  Appends a row and column to the given matrix
     *  of M rows and N columns
     */
    KERNEL_DECL(insert_last_rowcol, compute_mode::CPU)
    {
        template <compute_mode, typename T> static void op(
            aligned_vector<T>& A, const size_t M, const size_t N, const T val)
        {
            assert(A.size() == M * N);
            const size_t LEN = N * sizeof(T);

            A.resize(A.size() + N + M + 1, val);

            /* traversing backwards */
            for (ptrdiff_t i = (M * N) - N, dest = i + (M-1);
                 dest >= 0;
                 dest -= N + 1, i -= N)
            {
                /* fill last column on this row */
                A[dest + N] = val;

                /* shift values right such that a column
                   on each row is inserted */
                std::memmove(&A[dest], &A[i], LEN);
            }
        }
    };
}}

namespace ss
{
    template <typename T>
    online_column_inverse<T>::online_column_inverse(size_t m, size_t capacity)
        : _m{ m }
        , _n{ 0u }
    {
        _inv.reserve(capacity * capacity);
        _A_sub_t.reserve(capacity * m);
    }

    template <typename T>
    template <typename It>
    void online_column_inverse<T>::insert(const size_t idx, It begin, It end)
    {
        assert(idx <= _n);
        assert(std::distance(begin, end) == _shape[0]);

        const size_t M = _m;
        const size_t n = _n;

        if (n == 0) {
            /* initialize */
            _A_sub_t.insert(_A_sub_t.begin(), begin, end);

            T A_gamma_norm{ blas::xnrm2(M, _A_sub_t.data(), 1) };
            T inv_at_A { T(1) / (A_gamma_norm * A_gamma_norm) };

            _inv.push_back(inv_at_A);
        }
        else {
            /* compute the inverse as if adding a column to the end */
            auto u1 = std::make_unique<T[]>(n);
            T vcol_dot = 0;
            {
                /* append the input */
                auto row = _A_sub_t.insert(_A_sub_t.end(), begin, end);
                const T* rowptr = std::addressof(*row);

                vcol_dot = blas::xdot(M, rowptr, 1, rowptr, 1);

                /* current view of A_sub_t */
                mat_view<T> At = as_span<2>(_A_sub_t.data(), { n, M });
                blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(At), dim<1>(At), 1.0,
                    At.cbegin(), dim<1>(At),
                    rowptr, 1, 0.0,
                    u1.get(), 1);

                /* move the new row to the new row point */
                std::rotate(_A_sub_t.begin() + idx * M, row, _A_sub_t.end());
            }

            auto u2 = std::make_unique<T[]>(n);
            blas::xgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0,
                _inv.data(), n,
                u1.get(), 1, 0.0,
                u2.get(), 1);

            T d = T(1) / (vcol_dot - blas::xdot(n, u1.get(), 1, u2.get(), 1));

            kernelpp::run<detail::insert_last_rowcol>(_inv, n, n, T(0));
            const size_t new_n = n + 1;

            blas::xger(CblasRowMajor, n, n, d,
                u2.get(), 1,
                u2.get(), 1,
                _inv.data(), new_n);

            auto new_inv = as_span<2>(_inv.data(), { new_n, new_n });
            /* assign u3 to bottom row/right-most column */
            for (size_t i{ 0 }; i < n; ++i)
            {
                T u3{ -d * u2[i] };

                new_inv(i, n) = u3;
                new_inv(n, i) = u3;
            }

            /* assign u3 to bottom right */
            new_inv(n, n) = d;

            /* permute to get the matrix corresponding to original X */
            kernelpp::run<detail::square_permute>(new_inv, n, idx);
        }
        _n++;
    }

    template <typename T>
    void online_column_inverse<T>::remove(size_t idx)
    {
        assert(idx < N());

        const size_t M = _m;
        const size_t n = N();

        if (n == 1) {
            _inv.clear();
            _A_sub_t.clear();
        }
        else {
            /* permute to bring the column at the end in X */
            mat_view<T> inv = as_span<2>(_inv.data(), { n, n });
            {
                /* erase row from the transposed subset */
                auto it = _A_sub_t.begin() + (idx * M);
                _A_sub_t.erase(it, it + M);

                /* shift to last column */
                kernelpp::run<detail::square_permute>(inv, idx, dim<1>(inv) - 1);
            }

            /* update the inverse by removing the last column */
            {
                const size_t new_n = n - 1;
                T d = inv(new_n, new_n);

                blas::xscal(new_n, -(T(1) / d), &inv(0, new_n), n);

                /* A := alpha*x*y**T + A
                   note: A - d * x == -d * x + A
                 */
                blas::xger(CblasRowMajor, new_n, new_n, -d,
                    &inv(0, new_n), n,
                    &inv(0, new_n), n,
                    &inv(0, 0),     n);

                /* resize and assign */
                kernelpp::run<detail::erase_last_rowcol>(_inv, n, n);
            }
        }
        _n--;
    }

    template <typename T>
    const mat_view<T> online_column_inverse<T>::inverse()
    {
        assert(_inv.size() >= N() * N());
        return as_span<2>(_inv.data(), { N(), N() });
    }
}