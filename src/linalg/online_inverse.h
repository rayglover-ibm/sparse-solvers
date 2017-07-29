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

namespace ss
{
    /*  Given a non-owning reference to a matrix `A`, maintains
     *  the inverse of column-wise subset of `A`.
     *
     *  Refer to ./docs/algorithms/online-matrix-inverse for more information
     */
    template <typename T>
    class online_column_inverse
    {
      public:
        online_column_inverse(const mat_view<T>& A);

        /* returns a view of the inverse */
        const mat_view<T> inverse();

        /*  Inserts a column of `A` in to the inverse. Returns a
         *  view of the updated inverse.
         */
        void insert(const size_t column_idx);

        /*  Removes a column of `A` from the inverse. Returns a
         *  view of the updated inverse.
         */
        void remove(size_t column_idx);

        /*  Inverts the membership of the column at the given
         *  column index in the inverse. Returns a view of the
         *  updated inverse.
         */
        void flip(const size_t index);

        /* returns the indices of `A` in the inverse */
        const std::vector<bool>& indices() const { return _indices; }

        /* returns the size of the inverse */
        const size_t N() { return N; }

      private:
        mat_view<T> subset_transposed();

        /*  Returns the index of the column in the inverse currently
         *  corresponding to the given column index of _A
         */
        size_t insertion_index(size_t column_idx);

        /* reference matrix */
        const mat_view<T> _A;
        /* A_gamma transposed */
        std::vector<T> _A_sub_t;
        /* the inverse of A_gamma */
        std::vector<T> _inv;
        /* number of columns of _A corresponding to the inverse */
        size_t _n;
        /* column indices of _A corresponding to the inverse */
        std::vector<bool> _indices;
    };
}

/* Implementation ---------------------------------------------------------- */

namespace ss { namespace detail
{
    /*  Permutes the given matrix `A` such that the row and column `src`
     *  is moved to `dest`, with intermediate rows and columns shifted to
     *  account for this movement.
     */
    KERNEL_DECL(square_permute, ::kernelpp::compute_mode::CPU)
    {
        template<::kernelpp::compute_mode, typename T> static void op(
            mat_view<T> A, const size_t src, const size_t dest)
        {
            assert(dim<0>(A) == dim<1>(A));

            T* ptr = &A(0, 0);
            ptrdiff_t N = dim<1>(A), srci = src, desti = dest;

            if (N == 1 || desti == srci) {
                return;
            }
            else if (desti > srci) {
                /* traverse forwards */
                for (ptrdiff_t m = 0, i = 0; m < N; m++) {
                    /* row rotation */
                    if (m >= srci && m < desti) {
                        for (ptrdiff_t j = i; j < i + N; j++) {
                            T tmp = ptr[j];

                            ptr[j] = ptr[j + N];
                            ptr[j + N] = tmp;
                        }
                    }

                    /* move to src column */
                    i += srci;

                    /* column rotation */
                    for (ptrdiff_t n = srci; n < desti; n++, i++) {
                        T tmp = ptr[i];

                        ptr[i] = ptr[i + 1];
                        ptr[i + 1] = tmp;
                    }

                    /* move to next row */
                    i += N - desti;
                }
            }
            else {
                /* traverse backwards */
                for (ptrdiff_t m = N-1, i = (N * N)-1; m >= 0; m--) {
                    /* row rotation */
                    if (m <= srci && m > desti) {
                        for (ptrdiff_t j = i; j > i - N; j--) {
                            T tmp = ptr[j];

                            ptr[j] = ptr[j - N];
                            ptr[j - N] = tmp;
                        }
                    }

                    /* move to src column */
                    i -= (N - 1) - srci;

                    /* column rotation */
                    for (ptrdiff_t n = srci; n > desti; n--, i--) {
                        T tmp = ptr[i];

                        ptr[i] = ptr[i - 1];
                        ptr[i - 1] = tmp;
                    }

                    /* move to next row */
                    i -= desti + 1;
                }
            }
        }
    };

    KERNEL_DECL(erase_last_rowcol, ::kernelpp::compute_mode::CPU)
    {
        template<::kernelpp::compute_mode, typename T> static void op(
            std::vector<T>& v, const size_t M, const size_t N)
        {
            assert(v.size() == M * N);

            size_t i = 0;
            for (size_t m = 0; m < M-1; m++)
            {
                /* traversing forwards, shift values left such
                 * that the last column is removed */
                for (size_t n = 0; n < N-1; n++, i++)
                    v[i] = v[i + m];
            }

            v.erase(v.end() - (N + M - 1), v.end());
        }
    };

    KERNEL_DECL(insert_last_rowcol, ::kernelpp::compute_mode::CPU)
    {
        template<::kernelpp::compute_mode, typename T> static void op(
            std::vector<T>& v, const size_t M, const size_t N, const T& val)
        {
            assert(v.size() == M * N);
            v.resize(v.size() + N + M + 1, val);

            ptrdiff_t i = (M * N) - 1;
            for (ptrdiff_t m = M-1; m >= 0; m--)
            {
                /* fill last column */
                v[i + m + 1] = val;

                /* traversing backwards, shift values right such
                * that a column is appended */
                for (ptrdiff_t n = N-1; n >= 0; n--, i--)
                    v[i + m] = v[i];
            }
        }
    };

    template <typename T>
    void insert_col_into_row(
        std::vector<T>& v,
        const mat_view<T>& A,
        const size_t src_col,
        const size_t dest_row
        )
    {
        auto m = dim<0>(A);
        auto x = xt::view(A, xt::all(), src_col);

        v.insert(v.begin() + (dest_row * m), x.cbegin(), x.cend());
    }
}}

namespace ss
{
    template <typename T>
    online_column_inverse<T>::online_column_inverse(const mat_view<T>& A)
        : _A{ A }
        , _n{ 0 }
        , _indices(dim<1>(A), false)
    {
        _inv.reserve(10 * 10);
        _A_sub_t.reserve(10 * dim<0>(A));
    }

    template <typename T>
    void online_column_inverse<T>::insert(const size_t column_idx)
    {
        assert(column_idx < dim<1>(_A));

        if (_indices[column_idx]) {
            return;
        }

        size_t const M = dim<0>(_A);
        if (_n == 0) {
            /* initialize */
            detail::insert_col_into_row(_A_sub_t, _A, column_idx, 0);

            T A_gamma_norm{ blas::xnrm2(M, _A_sub_t.data(), 1) };
            T inv_at_A { T(1) / (A_gamma_norm * A_gamma_norm) };

            _inv.push_back(inv_at_A);
        }
        else {
            /* compute the inverse as if adding a column to the end */
            size_t idx{ insertion_index(column_idx) };

            auto u1 = std::make_unique<T[]>(_n);
            T vcol_dot = 0;
            {
                xt::xtensor<T, 1> vcol = xt::view(_A, xt::all(), column_idx);
                vcol_dot = blas::xdot(vcol.size(), vcol.cbegin(), 1, vcol.cbegin(), 1);

                /* current view of A_sub_t */
                mat_view<T> At = subset_transposed();

                blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(At), dim<1>(At), 1.0,
                    At.cbegin(), dim<1>(At),
                    vcol.cbegin(), 1, 0.0,
                    u1.get(), 1);

                /* update A_sub_t */
                detail::insert_col_into_row(_A_sub_t, _A, column_idx, idx);
            }

            auto u2 = std::make_unique<T[]>(_n);
            blas::xgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0,
                _inv.data(), _n,
                u1.get(), 1, 0.0,
                u2.get(), 1);

            T d = T(1) / (vcol_dot - blas::xdot(_n, u1.get(), 1, u2.get(), 1));

            assert(!kernelpp::run<detail::insert_last_rowcol>(_inv, _n, _n, T(0)));
            size_t new_n{ _n + 1 };

            blas::xger(CblasRowMajor, _n, _n, d,
                u2.get(), 1,
                u2.get(), 1,
                _inv.data(), new_n);

            auto new_inv = as_span<2>(_inv.data(), { new_n, new_n });
            /* assign u3 to bottom row/right-most column */
            for (size_t i{ 0 }; i < _n; ++i)
            {
                T u3{ -d * u2[i] };

                new_inv(i, _n) = u3;
                new_inv(_n, i) = u3;
            }

            /* assign u3 to bottom right */
            new_inv(_n, _n) = d;

            /* permute to get the matrix corresponding to original X */
            assert(!kernelpp::run<detail::square_permute>(new_inv, _n, idx));
        }

        _indices[column_idx] = true;
        _n++;
    }

    template <typename T>
    void online_column_inverse<T>::remove(size_t column_idx)
    {
        assert(_n > 0);
        assert(column_idx < dim<1>(_A));

        if (!_indices[column_idx]) {
            return;
        }

        if (_n == 1) {
            _inv.clear();
            _A_sub_t.clear();
        }
        else {
            /* permute to bring the column at the end in X */
            mat_view<T> inv = as_span<2>(_inv.data(), { _n, _n });
            {
                /* calculate column to remove */
                size_t idx{ insertion_index(column_idx) };

                /* erase row from the transposed subset */
                auto it = _A_sub_t.begin() + (idx * dim<0>(_A));
                _A_sub_t.erase(it, it + dim<0>(_A));

                /* shift to last column */
                assert(!kernelpp::run<detail::square_permute>(inv, idx, dim<1>(inv) - 1));
            }

            /* update the inverse by removing the last column */
            {
                size_t new_n{ _n - 1 };
                T d{ inv(new_n, new_n) };

                blas::xscal(new_n, -(T(1) / d), &inv(0, new_n), _n);

                /* A := alpha*x*y**T + A
                   note: A - d * x == -d * x + A
                 */
                blas::xger(CblasRowMajor, new_n, new_n, -d,
                    &inv(0, new_n), _n,
                    &inv(0, new_n), _n,
                    &inv(0, 0),     _n);

                /* resize and assign */
                assert(!kernelpp::run<detail::erase_last_rowcol>(_inv, _n, _n));
            }
        }

        _indices[column_idx] = false;
        _n--;
    }

    template <typename T>
    void online_column_inverse<T>::flip(const size_t index)
    {
        assert(index < _indices.size());
        if (_indices[index])
            remove(index);
        else
            insert(index);
    }

    template <typename T>
    const mat_view<T> online_column_inverse<T>::inverse()
    {
        assert(_inv.size() >= _n * _n);
        return as_span<2>(_inv.data(), { _n, _n });
    }

    template <typename T>
    size_t online_column_inverse<T>::insertion_index(size_t column_idx)
    {
        assert(column_idx < dim<1>(_A));

        size_t idx{ 0u };
        for (uint32_t i{ 0u }; i < column_idx; ++i) {
            if (_indices[i]) {
                idx++;
            }
        }
        return idx;
    }

    template <typename T>
    mat_view<T> online_column_inverse<T>::subset_transposed()
    {
        assert(_A_sub_t.size() >= _n * dim<0>(_A));
        return as_span<2>(_A_sub_t.data(), { _n, dim<0>(_A) });
    }
}