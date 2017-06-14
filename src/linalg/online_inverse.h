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

#include <memory>
#include <cassert>

namespace ss { namespace detail
{
    /*
     *  Permutes the given matrix `A` such that the row and column `src`
     *  is moved to `dest`, with intermediate rows and columns shifted to
     *  account for this movement.
     */
    template <typename T>
    void square_permute(
        mat_view<T> A,
        const size_t src,
        const size_t dest
        )
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

    template <typename T>
    void erase_last_rowcol(
        std::vector<T>& v,
        const size_t M,
        const size_t N
        )
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

    template <typename T>
    void insert_last_rowcol(
        std::vector<T>& v,
        const size_t M,
        const size_t N,
        const T& val
        )
    {
        assert(v.size() == M * N);
        v.resize(v.size() + N + M + 1, val);

        ptrdiff_t i = (M * N) - 1;
        for (ptrdiff_t m = M-1; m > 0; m--)
        {
            /* fill last column */
            v[i + m + 1] = val;

            /* traversing backwards, shift values right such
             * that a column is appended */
            for (ptrdiff_t n = N-1; n >= 0; n--, i--)
                v[i + m] = v[i];
        }
    }

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
    /*
     *  Given a non-owning reference to a matrix `A`, maintains
     *  the inverse of column-wise subset of `A`.
     *
     *  Refer to ./docs/algorithms/online-matrix-inverse for more information
     */
    template <typename T>
    class online_column_inverse
    {
      public:
        online_column_inverse(const mat_view<T>& A)
            : _A{ A }
            , _indices(dim<1>(A), false)
            , _n{ 0 }
        {
            _inv.reserve(10 * 10);
            _A_sub_t.reserve(10 * dim<0>(A));
        }

        /*
         *  Inserts a column of `A` in to the inverse. Returns a
         *  view of the updated inverse.
         */
        const mat_view<T> insert(const uint32_t column_idx)
        {
            assert(column_idx < dim<1>(_A));

            if (_indices[column_idx]) {
                return inverse();
            }

            size_t const M = dim<0>(_A);
            if (_n == 0) {
                /* initialize */
// Py           A_gamma = helper.subset_array(A, lambda_indices)
                detail::insert_col_into_row(_A_sub_t, _A, column_idx, 0);

// Py           invAtA = 1.0 / (np.linalg.norm(A_gamma) * np.linalg.norm(A_gamma))
                T A_gamma_norm{ blas::xnrm2(M, _A_sub_t.data(), 1) };
                T inv_at_A { (T)1.0 / (A_gamma_norm * A_gamma_norm) };

                _inv.push_back(inv_at_A);
            }
            else {
                /* compute the inverse as if adding a column to the end */
                xt::xtensor<T, 1> vcol = xt::view(_A, xt::all(), column_idx);
                T vcol_dot = blas::xdot(vcol.size(), vcol.cbegin(), 1, vcol.cbegin(), 1);

// Py           u1 = np.dot(matA.T, vCol)
                auto u1 = std::make_unique<T[]>(_n);
                {
                    /* current view of sub_A_t */
                    mat_view<T> At = subset_transposed();

                    blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(At), dim<1>(At), 1.0,
                        At.cbegin(), dim<1>(At),
                        vcol.cbegin(), 1, 0.0,
                        u1.get(), 1);
                }

// Py           u2 = np.dot(current_inverse, u1)
                auto u2 = std::make_unique<T[]>(_n);
                blas::xgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0,
                    _inv.data(), _n,
                    u1.get(), 1, 0.0,
                    u2.get(), 1);

// Py           d = 1.0 / float(np.dot(vCol.T, vCol) - np.dot(u1.T, u2))
                T d = 1.0 / (vcol_dot - blas::xdot(_n, u1.get(), 1, u2.get(), 1));

                detail::insert_last_rowcol(_inv, _n, _n, T(0));
                uint32_t new_n{ _n + 1 };

// Py           A := alpha*x*y**T + A
                blas::xger(CblasRowMajor, _n, _n, d,
                    u2.get(), 1,
                    u2.get(), 1,
                    _inv.data(), new_n);

                auto new_inv = as_span<2>(_inv.data(), { new_n, new_n });
                {
                    /* assign u3 to bottom row/right-most column */
// Py               new_inverse[0:N, N] = -u3
// Py               new_inverse[N, 0:N] = -u3.T
                    for (size_t i{ 0 }; i < _n; ++i)
                    {
                        T u3{ -d * u2[i] };

                        new_inv(i, _n) = u3;
                        new_inv(_n, i) = u3;
                    }

                    /* assign u3 to bottom right */
// Py               new_inverse[N, N] = d
                    new_inv(_n, _n) = d;
                }

                /* permute to get the matrix corresponding to original X */
// Py           permute_order = np.hstack((np.arange(0, pos_vCol), N, np.arange(pos_vCol, N)))
// Py           new_inverse = new_inverse[:, permute_order]
// Py           new_inverse = new_inverse[permute_order, :]
                {
                    /* calculate destination column */
                    size_t idx{ insertion_index(column_idx) };

                    /* shift to position */
                    detail::square_permute(new_inv, _n, idx);

                    /* update A_sub_t */
                    detail::insert_col_into_row(_A_sub_t, _A, column_idx, idx);
                }
            }

            _indices[column_idx] = true;
            _n++;

            return inverse();
        }

        /*
         *  Removes a column of `A` from the inverse. Returns a
         *  view of the updated inverse.
         */
        const mat_view<T> remove(uint32_t column_idx)
        {
            assert(_n > 0);
            assert(column_idx < dim<1>(_A));

            if (!_indices[column_idx]) {
                return inverse();
            }

            if (_n == 1) {
                _inv.clear();
                _A_sub_t.clear();
            }
            else {
                /* permute to bring the column at the end in X */
                mat_view<T> inv = as_span<2>(_inv.data(), { _n, _n });

// Py           permute_order = np.hstack((np.arange(0, pos_vCol), np.arange(pos_vCol + 1, N), pos_vCol))
// Py           current_inverse = current_inverse[permute_order, :]
// Py           current_inverse = current_inverse[:, permute_order]
                {
                    /* calculate column to remove */
                    size_t idx{ insertion_index(column_idx) };

                    /* erase row from the transposed subset */
                    auto it = _A_sub_t.begin() + (idx * dim<0>(_A));
                    _A_sub_t.erase(it, it + dim<0>(_A));

                    /* shift to last column */
                    detail::square_permute(inv, idx, dim<1>(inv) - 1);
                }

                /* update the inverse by removing the last column */
                {
                    uint32_t new_n{ _n - 1 };

// Py               d = current_inverse[N - 1, N - 1]
                    T d{ inv(new_n, new_n) };

// Py               u2 = -(1.0 / d) * current_inverse[0:N - 1, N - 1]
                    blas::xscal(new_n, -(T(1) / d), &inv(0, new_n), _n);

// Py               F11inv = current_inverse[0:N - 1, 0 : N - 1]
// Py               new_inverse = F11inv - (d * np.outer(u2, u2.T))

                    /* A := alpha*x*y**T + A
                       note: A - d * x == -d * x + A
                     */
                    blas::xger(CblasRowMajor, new_n, new_n, -d,
                        &inv(0, new_n), _n,
                        &inv(0, new_n), _n,
                        &inv(0, 0),     _n);

                    /* resize and assign */
                    detail::erase_last_rowcol(_inv, _n, _n);
                }
            }

            _indices[column_idx] = false;
            _n--;

            return inverse();
        }

        /*
         *  Inverts the membership of the column at the given
         *  column index in the inverse. Returns a view of the
         *  updated inverse.
         */
        const mat_view<T> flip(const uint32_t index) {
            assert(index < _indices.size());
            if (_indices[index])
                return remove(index);
            else
                return insert(index);
        }

        const std::vector<bool>& indices() const {
            return _indices;
        }

        const uint32_t N() { return _n; }

        /* returns a view of the inverse */
        const mat_view<T> inverse()
        {
            assert(_inv.size() >= _n * _n);
            return as_span<2>(_inv.data(), { _n, _n });
        }

      private:
        mat_view<T> subset_transposed()
        {
            assert(_A_sub_t.size() >= _n * dim<0>(_A));
            return as_span<2>(_A_sub_t.data(), { _n, dim<0>(_A) });
        }

        /*
            Returns the index of the column in the inverse currently
            corresponding to the given column index of _A
         */
        size_t insertion_index(uint32_t column_idx)
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



        /* reference matrix */
        const mat_view<T> _A;
        /* A_gamma transposed */
        std::vector<T> _A_sub_t;
        /* the inverse of A_gamma */
        std::vector<T> _inv;
        /* number of columns of _A corresponding to the inverse */
        uint32_t _n;
        /* column indices of _A corresponding to the inverse */
        std::vector<bool> _indices;
    };
}