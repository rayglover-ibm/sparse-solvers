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

#include "homotopy.h"
#include "blas_wrapper.h"

#include <xtensor/xview.hpp>

#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <assert.h>

namespace ss
{
    /* The following is a port:
          of ./tools/sparsity/src/Homotopy.py,
          at sha1 20b980c7804883d059896e04c3a0047615cbd984,
          committed 2015-11-09 14:08:24
    */
    using std::make_unique;

    template <typename T>
    using mat_view = ss::ndspan<T, 2>;

    template <typename T>
    using mat = xt::xtensor<T, 2>;

    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }

#pragma region helpers
    template<typename T>
    void columnwise_sum(
        mat_view<T>& A,
        T* x
        )
    {
        std::fill_n(x, dim<1>(A), (T)0.0);

        for (size_t r{ 0 }; r < dim<0>(A); ++r) {
            for (size_t c{ 0 }; c < dim<1>(A); ++c) {
                x[c] += A(r, c);
            }
        }
    }

    template<typename T>
    T inf_norm(
        const T* v,
        const size_t n,
        size_t* idx
        )
    {
        *idx = blas::ixamax(n, v, 1);
        return std::abs(v[*idx]);
    }

    template<typename T>
    T inf_norm(
        const T* v,
        const size_t n
        )
    {
        size_t idx;
        return inf_norm(v, n, &idx);
    }

    template<typename T>
    size_t mat_subset_cols(
        const mat<T>& A,
        const std::vector<bool>& col_indices,
        mat<T>& A_subset
        )
    {
        const size_t M = dim<0>(A_subset);
        const size_t N = dim<1>(A_subset);

        size_t i{ 0 }, n{ 0 };
        for (const bool val : col_indices) {
            if (val) {
                /* copy col */
                blas::xcopy(
                    M /* rows */,
                    A.raw_data() + i        /* src col */,  dim<1>(A),
                    A_subset.raw_data() + n /* dest col */, M);

                ++n;

                if (n == N)
                    break;
            }
            ++i;
        }
        return n;
    }

    template<typename T>
    void shift_column(
        mat_view<T>& A,
        const size_t col,
        const size_t dest_col
        )
    {
        assert(dim<1>(A) > col);
        assert(dim<1>(A) > dest_col);

        if (col == dest_col) { return; }
        const int32_t col_inc { dest_col < col ? -1 : 1 };

        /* for each row */
        for (size_t r{ 0 }; r < dim<0>(A); ++r) {
            /* take the value to move */
            T const val{ A(r, col) };

            /* for each col, starting at the src col */
            for (size_t c{ col }; c != dest_col; c += col_inc) {
                /* shift column left/right */
                A(r, c) = A(r, c + col_inc);
            }

            A(r, dest_col) = val;
        }
    }

    template<typename T>
    void shift_row(
        mat_view<T>& A,
        const size_t row,
        const size_t dest_row
        )
    {
        assert(dim<0>(A) > row);
        assert(dim<0>(A) > dest_row);

        if (row == dest_row) { return; }
        const int32_t row_inc{ dest_row < row ? -1 : 1 };

        /* store the src row */
        xt::xtensor<T, 1> x = xt::view(A, row);

        for (size_t r{ row }; r != dest_row; r += row_inc)
        {
            /* shift row upward/downward */
            xt::view(A, r) = xt::view(A, int(r) + row_inc);
        }

        /* copy row to destination */
        view(A, dest_row, xt::all()) = x;
    }

    template<typename T>
    void sign_vector(
        const size_t  n,
        const T* x,
        T* x_sign,
        const T tol
        )
    {
        for (size_t i{ 0 }; i < n; i++) {
            const T val{ x[i] };
            if (val > tol) {
                x_sign[i] = 1;
            }
            else if (val < -tol) {
                x_sign[i] = -1;
            }
            else {
                x_sign[i] = 0;
            }
        }
    }

    template<typename T>
    void zero_mask(
        const std::vector<bool>& mask,
        const T* x,
        T* y
        )
    {
        std::fill_n(y, mask.size(), (T)0.0);
        for (size_t i{ 0 }, off{ 0 }; i < mask.size(); i++) {
            if (mask[i]) {
                y[i] = x[off];
                ++off;
            }
        }
    }

#pragma endregion

#pragma region online_inverse

    template<typename T>
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
                xt::xtensor<T, 1> v_col = xt::view(_A, xt::all(), column_idx);  //make_unique<T[]>(M);

// Py           u1 = np.dot(matA.T, vCol)
                auto u1 = make_unique<T[]>(_n);

                {
                    /* current view of sub_A_t */
                    mat_view<T> At = subset_transposed();

                    blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(At), dim<1>(At), 1.0, At.raw_data(), dim<1>(At),
                        v_col.raw_data(), 1, 0.0,
                        u1.get(), 1);
                }

// Py           u2 = np.dot(current_inverse, u1)
                auto u2 = make_unique<T[]>(_n);
                blas::xgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0, _inv.data(),
                    _n, u1.get(), 1, 0.0, u2.get(), 1);

// Py           d = 1.0 / float(np.dot(vCol.T, vCol) - np.dot(u1.T, u2))
                T d {1.0};
                {
                    T a = blas::xdot(M, v_col.raw_data(), 1, v_col.raw_data(), 1);
                    T b = blas::xdot(_n, u1.get(), 1, u2.get(), 1);

                    d /= a - b;
                }

// Py           u3 = d * u2
                auto u3 = make_unique<T[]>(_n);

                std::copy_n(u2.get(), _n, u3.get());
                blas::xscal(_n, d, u3.get(), 1);

// Py           F11inv = current_inverse + (d * np.outer(u2, u2.T))
                mat<T> F11inv({ _n, _n });
                std::copy_n(_inv.data(), _inv.size(), F11inv.raw_data());

// Py           A := alpha*x*y**T + A
                blas::xger(CblasRowMajor, _n, _n, d,
                    u2.get(), 1,
                    u2.get(), 1,
                    F11inv.raw_data(), _n);

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
                    auto u3 = make_unique<T[]>(new_n);
                    blas::xcopy(new_n, inv.raw_data() + new_n, new_n + 1, u3.get(), 1);
                    blas::xscal(new_n, -1, u3.get(), 1);

// py               u2 = (1.0 / d) * u3
                    auto u2 = make_unique<T[]>(new_n);
                    blas::xcopy(new_n, u3.get(), 1, u2.get(), 1);
                    blas::xscal(new_n, (T)1.0 / d, u2.get(), 1);

// Py               F11inv = current_inverse[0:N - 1, 0 : N - 1]
//                    std::fill_n(F11inv.raw_data(), F11inv.size, (T)0.0);

// Py               new_inverse = F11inv - (d* np.outer(u2, u2.T))
                    /* A := alpha*x*y**T + A,  */
                    blas::xger(CblasRowMajor, new_n, new_n, d,
                        u2.get(), 1,
                        u2.get(), 1,
                        F11inv.raw_data(), new_n);

                    /* slightly awkward.. */
                    for (size_t r{ 0 }; r < dim<0>(F11inv); ++r) {
                        for (size_t c{ 0 }; c < dim<1>(F11inv); ++c) {
                            F11inv(r, c) = inv(r, c) - F11inv(r, c);
                        }
                    }
                }

                /* resize and assign */
                _inv.assign(F11inv.size(), 0);
                std::copy_n(F11inv.raw_data(), F11inv.size(), _inv.data());
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

            blas::xcopy(m, A.raw_data() + src_col, dim<1>(A), &*it, 1);
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

#pragma endregion

    template<typename T>
    void residual_vector(
        const mat_view<T>& A,
        const T* y,
        const T* x_previous,
        T* c
        )
    {
// Py   A_t = np.matrix.transpose(A)
        /*
            Port:

            The transpose is not evaluated here and is instead
            computed at the blas function when evaluating
            'np.dot(A_t, difference)'
        */

// Py   A_x = np.dot(A, x_previous)
        auto A_x = make_unique<T[]>(dim<0>(A));
        blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(A), dim<1>(A), 1.0, A.raw_data(),
            dim<1>(A), x_previous, 1, 0.0, A_x.get(), 1);

// Py   difference = np.zeros(len(y))
// Py   for i in range(0, len(y)) :
// Py       difference[i] = y[i] - A_x[i]

        /* A_x = y - A_x */
        for (size_t i = 0; i < dim<0>(A); i++) {
            A_x[i] = y[i] - A_x[i];
        }

// Py   return np.dot(A_t, difference)
        blas::xgemv(CblasRowMajor, CblasTrans, dim<0>(A), dim<1>(A), 1.0, A.raw_data(),
            dim<1>(A), A_x.get() /* difference */, 1, 0.0, c, 1);
    }

    template<typename T>
    std::pair<T, uint32_t> find_max_gamma(
        const mat_view<T>& A,
        const T* res_vec,
        const T* x,
        const T* dir_vec,
        const T c_inf,
        const std::vector<bool>& lambda_indices
        )
    {
        /* evaluate the eligible elements of transpose(A) * A * dir_vec */
        /* p = Ad */
        auto p = make_unique<T[]>(dim<0>(A));
        blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(A), dim<1>(A), 1.0, A.raw_data(),
            dim<1>(A), dir_vec, 1, 0.0, p.get(), 1);

        /* q = transpose(A)p */
        auto q = make_unique<T[]>(dim<1>(A));
        blas::xgemv(CblasRowMajor, CblasTrans, dim<0>(A), dim<1>(A), 1.0, A.raw_data(),
            dim<1>(A), p.get(), 1, 0.0, q.get(), 1);

        /* evaluate the competing lists of terms */
        T min{ std::numeric_limits<T>::max() };
        uint32_t idx{ 0u };

        /* find the minimum term and its index */
        for (uint32_t i{ 0u }; i < lambda_indices.size(); i++) {
            const T prev = min;
            if (lambda_indices[i]) {
                T minT = -x[i] / dir_vec[i];
                if (minT > 0.0 && minT < min) {
                    min = minT;
                }
            }
            else {
                T di_left{ 1.0f - q[i] }, di_right{ 1.0f + q[i] };

                if (std::abs(di_left) > 0.0) {
                    T leftT = (c_inf - res_vec[i]) / di_left;
                    if (leftT > 0.0 && leftT < min) {
                        min = leftT;
                    }
                }

                if (std::abs(di_right) > 0.0) {
                    T rightT = (c_inf + res_vec[i]) / di_right;
                    if (rightT > 0.0 && rightT < min) {
                        min = rightT;
                    }
                }
            }

            if (prev > min) {
                /* yield the index of first (left-most)
                   occurance of min */
                idx = i;
            }
        }

        return std::make_pair(min, idx);
    }

    template<typename T>
    homotopy_report run_solver(
        const mat_view<T>& A,
        const std::uint32_t max_iter,
        const T tolerance,
        const T* y,
        T* x
        )
    {
        assert(max_iter > 0);
        assert(y != nullptr);
        assert(x != nullptr);

        /* using a tolerance lt epsilon is generally not good */
        assert(tolerance >= std::numeric_limits<T>::epsilon()
            && tolerance < 1.0);

        /* initialise x to a vector of zeros */
// Py   x = np.zeros(N)
        std::fill_n(x, dim<1>(A), (T)0.0);

        /* initialise residual vector */
// Py   c_vec = residual_vector(A, y, x)
        mat<T> c_vec({ 1, dim<1>(A) });
        residual_vector(A, y, x, c_vec.raw_data());

        /* initialise lambda = || c_vec || _inf */
        // Py   c_inf = (np.linalg.norm(c_vec, np.inf))
        T c_inf{ 0.0 };

        auto direction_vec = make_unique<T[]>(dim<1>(A));
        online_column_inverse<T> inv(A);

        {
            size_t c_inf_i;
            c_inf = inf_norm(c_vec.raw_data(), dim<1>(A), &c_inf_i);

            T c_vec_gamma{ c_inf };
            T subsample_direction_vector{ 0.0 };
            auto inv_view = inv.insert((uint32_t)c_inf_i);

// Py       subsample_direction_vector = invAtA * helper.sign_vector(c_vec_gamma)
            sign_vector(1, &c_vec_gamma, &subsample_direction_vector, tolerance);
            subsample_direction_vector *= inv_view(0, 0);

// Py       direction_vec = helper.zero_mask(subsample_direction_vector, lambda_indices, N)
            zero_mask(inv.indices(), &subsample_direction_vector, direction_vec.get());
        }

        /* evaluate homotopy path segments in iterations */
        std::uint32_t iter{ 0u };
        while (iter < max_iter) {
            iter++;

            {
                auto const gamma = find_max_gamma(A, c_vec.raw_data(), x,
                    direction_vec.get(), c_inf, inv.indices());

                /* update inverse by inserting/removing the
                respective index from the inverse */
                inv.flip(gamma.second);

                /* update x */
                for (size_t i{ 0 }; i < dim<1>(A); i++) {
                    x[i] += gamma.first * direction_vec[i];
                }
            }

            /* update residual vector */
// Py       c_vec = residual_vector(A, y, x)
            residual_vector(A, y, x, c_vec.raw_data());

            /* update direction vector */
            {
// Py           c_vec_gamma = helper.subset_array(c_vec, lambda_indices)
                const uint32_t N{ inv.N() };

                mat<T> c_vec_gamma({1, N});
                mat_subset_cols(c_vec, inv.indices(), c_vec_gamma);

// Py           direction_vec = np.dot(invAtA, helper.sign_vector(c_vec_gamma))
                sign_vector(N, c_vec_gamma.raw_data(), c_vec_gamma.raw_data(), tolerance);
                auto dir_tmp = make_unique<T[]>(N);

                blas::xgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0,
                    inv.inverse().raw_data(), N,
                    c_vec_gamma.raw_data(), 1,
                    0.0,
                    dir_tmp.get(), 1);

// Py           direction_vec = helper.zero_mask(direction_vec, lambda_indices, N)
                zero_mask(inv.indices(), dir_tmp.get(), direction_vec.get());
            }

            /* find lambda(i.e., infinite norm of residual vector) */
            c_inf = inf_norm(c_vec.raw_data(), dim<1>(A));

            /* check if infinity norm of residual vector is within tolerance */
            if (c_inf < tolerance)
                break;
        }
        return{ iter, c_inf };
    }

    template<typename T>
    bool mat_norm_l1(
        mat_view<T>& A
        )
    {
        if (dim<1>(A) == 1 /* vector */) {
            T sum{ 0.0 };
            columnwise_sum(A, &sum);

            if (sum <= 0) { return false; }

            for (size_t i{ 0 }; i < dim<0>(A); ++i) {
                A(i, 0) /= sum;
            }
        }
        else {
            /* matrix */
            auto sums = make_unique<T[]>(dim<1>(A));
            columnwise_sum(A, sums.get());

            for (size_t i{ 0 }; i < dim<1>(A); ++i) {
                if (sums[i] <= 0) { return false; }
            }

            for (size_t r{ 0 }; r < dim<0>(A); ++r) {
                for (size_t c{ 0 }; c < dim<1>(A); ++c) {
                    A(r, c) /= sums[c];
                }
            }
        }

        return true;
    }

    template<typename T>
    void reconstruct_signal(
        const T* A,
        const std::uint32_t m,
        const std::uint32_t n,
        const T* x,
        T* y
        )
    {
        assert(m > 0
            && n > 0
            && A != nullptr
            && x != nullptr
            && y != nullptr);

        blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A,
            n, x, 1, 0.0, y, 1);
    }

    template<typename T>
    bool mat_norm_l1(
        T* A,
        const std::uint32_t m,
        const std::uint32_t n
        )
    {
        assert(m > 0
            && n > 0
            && A != nullptr);

        mat_view<T> A_mat{ m, n, A };
        return mat_norm_l1<T>(A_mat);
    }

/*  template homotopy_report solve_homotopy<float> (
        const float*, const std::uint32_t, const std::uint32_t, const std::uint32_t, const float, const float*, float*
        );

    template homotopy_report solve_homotopy<double>(
        const double*, const std::uint32_t, const std::uint32_t, const std::uint32_t, const double, const double*, double*
        );

    template bool mat_norm_l1<float>(
        float*, const std::uint32_t, const std::uint32_t
        );

    template bool mat_norm_l1<double>(
        double*, const std::uint32_t, const std::uint32_t
        );

    template void reconstruct_signal<float>(
        const float*, const std::uint32_t, const std::uint32_t, const float*, float*
        );

    template void reconstruct_signal<double>(
        const double*, const std::uint32_t, const std::uint32_t, const double*, double*
        );
*/

    template <> kernelpp::variant<homotopy_report, error_code> solve_homotopy::op<compute_mode::CPU>(
        const float* A_,
        const std::uint32_t A_m,
        const std::uint32_t A_n,
        const std::uint32_t max_iter,
        const float tolerance,
        const float* y,
        float* x
        )
    {
        assert(A_m > 0
            && A_n > 0
            && A_ != nullptr);

        const mat_view<float> A = ss::as_span<2, float>(const_cast<float*>(A_), { A_m, A_n });
        return run_solver(A, max_iter, tolerance, y, x);
    }
}