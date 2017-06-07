#pragma once

#include "blas_wrapper.h"
#include "linalg/common.h"

#include <ss/ndspan.h>
#include <xtensor/xview.hpp>

#include <memory>

namespace ss
{
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
                insert_col_into_row(_A_sub_t, _A, column_idx, 0);

// Py           invAtA = 1.0 / (np.linalg.norm(A_gamma) * np.linalg.norm(A_gamma))
                T A_gamma_norm{ blas::xnrm2(M, _A_sub_t.data(), 1) };
                T inv_at_A { (T)1.0 / (A_gamma_norm * A_gamma_norm) };

                _inv.push_back(inv_at_A);
            }
            else {
                /* compute the inverse as if adding a column to the end */
                xt::xtensor<T, 1> v_col = xt::view(_A, xt::all(), column_idx);

// Py           u1 = np.dot(matA.T, vCol)
                auto u1 = std::make_unique<T[]>(_n);
                {
                    /* current view of sub_A_t */
                    mat_view<T> At = subset_transposed();

                    blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(At), dim<1>(At), 1.0, At.cbegin(),
                        dim<1>(At), v_col.cbegin(), 1, 0.0,
                        u1.get(), 1);
                }

// Py           u2 = np.dot(current_inverse, u1)
                auto u2 = std::make_unique<T[]>(_n);
                blas::xgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0, _inv.data(),
                    _n, u1.get(), 1, 0.0, u2.get(), 1);

// Py           d = 1.0 / float(np.dot(vCol.T, vCol) - np.dot(u1.T, u2))
                T d {1.0};
                {
                    T a = blas::xdot(M, v_col.cbegin(), 1, v_col.cbegin(), 1);
                    T b = blas::xdot(_n, u1.get(), 1, u2.get(), 1);

                    d /= a - b;
                }

// Py           u3 = d * u2
                auto u3 = std::make_unique<T[]>(_n);

                std::copy_n(u2.get(), _n, u3.get());
                blas::xscal(_n, d, u3.get(), 1);

// Py           F11inv = current_inverse + (d * np.outer(u2, u2.T))
                mat<T> F11inv({ _n, _n });
                std::copy_n(_inv.data(), _inv.size(), F11inv.begin());

// Py           A := alpha*x*y**T + A
                blas::xger(CblasRowMajor, _n, _n, d,
                    u2.get(), 1,
                    u2.get(), 1,
                    F11inv.begin(), _n);

                /* assign F11inv */
                {
                    uint32_t new_n { _n + 1 };
                    _inv.assign(new_n * new_n, 0);

                    auto mut_inv = as_span<2>(_inv.data(), { new_n, new_n });
                    {
                        /* assign F11inv to top left */
// Py                   new_inverse[0:N, 0:N] = F11inv # [F11inv - u3 - u3' F22inv]
                        for (size_t r{ 0 }; r < _n; ++r) {
                            for (size_t c{ 0 }; c < _n; ++c) {
                                mut_inv(r, c) = F11inv(r, c);
                            }
                        }

                        /* assign u3 to right-most column */
// Py                   new_inverse[0:N, N] = -u3
                        for (size_t r{ 0 }; r < _n; ++r) {
                            mut_inv(r, _n) = -u3[r];
                        }

                        /* assign u3 to bottom row */
// Py                   new_inverse[N, 0:N] = -u3.T
                        for (size_t c{ 0 }; c < _n; ++c) {
                            mut_inv(_n, c) = -u3[c];
                        }

                        /* assign u3 to bottom row */
// Py                   new_inverse[N, N] = d
                        mut_inv(_n, _n) = d;
                    }

                    /* permute to get the matrix corresponding to original X */
// Py               permute_order = np.hstack((np.arange(0, pos_vCol), N, np.arange(pos_vCol, N)))
// Py               new_inverse = new_inverse[:, permute_order]
// Py               new_inverse = new_inverse[permute_order, :]
                    {
                        /* calculate destination column */
                        size_t idx{ insertion_index(column_idx) };

                        /* shift to position */
                        shift_column(mut_inv, _n, idx);
                        shift_row(mut_inv, _n, idx);

                        /* update A_sub_t */
                        insert_col_into_row(_A_sub_t, _A, column_idx, idx);
                    }
                }
            }

            _indices[column_idx] = true;
            _n++;

            return inverse();
        }

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
// Py           permute_order = np.hstack((np.arange(0, pos_vCol), np.arange(pos_vCol + 1, N), pos_vCol))
// Py           current_inverse = current_inverse[permute_order, :]
// Py           current_inverse = current_inverse[:, permute_order]
                {
                    mat_view<T> mut_inv = as_span<2>(_inv.data(), { _n, _n });

                    /* calculate column to remove */
                    size_t idx{ insertion_index(column_idx) };

                    {
                        mat_view<T> As_t{ subset_transposed() };
                        /* shift row to the bottom */
                        shift_row(As_t, idx, dim<0>(As_t) - 1);
                    }

                    /* shift to position */
                    shift_column(mut_inv, idx, dim<1>(mut_inv) - 1);
                    shift_row(mut_inv, idx, dim<1>(mut_inv) - 1);
                }

                uint32_t new_n{ _n - 1 };
                mat<T> F11inv({ new_n, new_n }, T(0));
                {
                    const mat_view<T> inv = as_span<2>(_inv.data(), { _n, _n });

// Py               #update the inverse by removing the last column
// Py               d = current_inverse[N - 1, N - 1]
                    T d{ inv(new_n, new_n) };

// Py               u3 = -1.0 * current_inverse[0:N - 1, N - 1]
                    /* copy last column */
                    auto u3 = std::make_unique<T[]>(new_n);
                    blas::xcopy(new_n, inv.cbegin() + new_n, new_n + 1, u3.get(), 1);
                    blas::xscal(new_n, -1, u3.get(), 1);

// py               u2 = (1.0 / d) * u3
                    auto u2 = std::make_unique<T[]>(new_n);
                    blas::xcopy(new_n, u3.get(), 1, u2.get(), 1);
                    blas::xscal(new_n, (T)1.0 / d, u2.get(), 1);

// Py               F11inv = current_inverse[0:N - 1, 0 : N - 1]
// Py               new_inverse = F11inv - (d* np.outer(u2, u2.T))

                    /* A := alpha*x*y**T + A,  */
                    blas::xger(CblasRowMajor, new_n, new_n, d,
                        u2.get(), 1,
                        u2.get(), 1,
                        F11inv.begin(), new_n);

                    /* slightly awkward.. */
                    for (size_t r{ 0 }; r < dim<0>(F11inv); ++r) {
                        for (size_t c{ 0 }; c < dim<1>(F11inv); ++c) {
                            F11inv(r, c) = inv(r, c) - F11inv(r, c);
                        }
                    }
                }

                /* resize and assign */
                _inv.assign(F11inv.size(), 0);
                std::copy_n(F11inv.cbegin(), F11inv.size(), _inv.data());
            }

            _indices[column_idx] = false;
            _n--;

            return inverse();
        }

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

        void insert_col_into_row(
            std::vector<T>& v,
            const mat_view<T>& A,
            const size_t src_col,
            const size_t dest_row
            )
        {
            auto m = dim<0>(A);
            auto it = v.insert(v.begin() + (dest_row * m), m, 0.0);

            blas::xcopy(m, A.cbegin() + src_col, dim<1>(A), &*it, 1);
        }

        /* original matrix */
        const mat_view<T>& _A;
        /* A_gamma transposed */
        std::vector<T> _A_sub_t;
        /* the inverse of A_gamma */
        std::vector<T> _inv;
        /* the indices of _A represented by the inverse */
        std::vector<bool> _indices;
        /* number of cols in _inv */
        uint32_t _n;
    };
}