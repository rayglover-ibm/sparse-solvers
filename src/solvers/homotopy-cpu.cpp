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

#pragma region structs   
    /* A view on to a two-dimensional, contiguous, 
       row-major matrix of M rows, N cols */
    template<typename TBuff>
    class mat_view
    {
      public:
        inline TBuff* row(const size_t i) const { return &_buff[N * i]; }
        inline TBuff* col(const size_t j) const { return &_buff[j]; }
        inline TBuff* data()              const { return _buff; }
        
        inline TBuff& operator()(size_t row, size_t col) const {
            return _buff[N * row + col];
        }

        mat_view() = delete;
        mat_view(const uint32_t M, const uint32_t N, TBuff* buff)
            : M{ M }, N{ N }, size{ M * N }, _buff{ buff }
        {}

        const uint32_t M;
        const uint32_t N;
        const uint32_t size;

      protected:
        TBuff *const  _buff;
    };

    template<typename TBuff>
    class mat : public mat_view<TBuff> 
    {
      public:
        mat(const uint32_t M,
            const uint32_t N,
            std::unique_ptr<TBuff[]> ptr)
            : mat_view<TBuff>(M, N, ptr.get())
            , _ptr (std::move(ptr))
        {}
        
        mat(const uint32_t M,
            const uint32_t N) 
            : mat(M, N, {make_unique<TBuff[]>(M*N)})
        {}

        explicit mat(mat<TBuff>&& other) 
            : mat(other.M, other.N, std::move(other._ptr))
        {}

      protected:
        std::unique_ptr<TBuff[]> _ptr;
    };

    template<typename T>
    using const_mat_view = mat_view<const T>;

#pragma endregion

#pragma region helpers
    template<typename T>
    void columnwise_sum(
        mat_view<T>& A,
        T* x
        )
    {
        std::fill_n(x, A.N, (T)0.0);

        for (size_t r{ 0 }; r < A.M; ++r) {
            for (size_t c{ 0 }; c < A.N; ++c) {
                x[c] += A(r, c);
            }
        }
    }
    
    template<typename T>
    T inf_norm(
        const T* v,
        const uint32_t n,
        size_t* idx
        )
    {
        *idx = blas::ixamax(n, v, 1);
        return std::abs(v[*idx]);
    }

    template<typename T>
    T inf_norm(
        const T* v,
        const uint32_t n
        )
    {
        size_t idx;
        return inf_norm(v, n, &idx);
    }

    template<typename T>
    size_t mat_subset_cols(
        const mat_view<T>& A,
        const std::vector<bool>& col_indices,
        mat_view<T>& A_subset
        )
    {
        const uint32_t m { A_subset.M };
        
        size_t i{ 0 }, n{ 0 };
        for (const bool val : col_indices) {
            if (val) {
                /* copy col */
                blas::xcopy(
                    m /* n-rows */, 
                    A.col(i) /* src column */, A.N,
                    A_subset.col(n) /* dest column */, 
                    A_subset.N);

                ++n;

                if (n == A_subset.N)
                    break;
            }
            ++i;
        }
        return n;
    }

    template<typename T>
    void shift_column(
        const mat_view<T>& A,
        const size_t col,
        const size_t dest_col
        )
    {
        assert(A.N > col);
        assert(A.N > dest_col);

        if (col == dest_col) { return; }
        const int32_t col_inc { dest_col < col ? -1 : 1 };
        
        /* for each row */
        for (size_t r{ 0 }; r < A.M; ++r) {
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
        const mat_view<T>& A,
        const size_t row,
        const size_t dest_row
        )
    {
        assert(A.M > row);
        assert(A.M > dest_row);

        if (row == dest_row) { return; }
        const int32_t row_inc{ dest_row < row ? -1 : 1 };
        
        /* store the src row */
        auto x = make_unique<T[]>(A.N);
        std::copy_n(A.row(row), A.N, x.get());

        /* for each row, starting at the dest row */
        for (size_t r{ row }; r != dest_row; r += row_inc) {
            /* shift row downward */
            std::copy_n(A.row(r + row_inc), A.N, A.row(r));
        }
        
        /* copy row to destination */
        std::copy_n(x.get(), A.N, A.row(dest_row));
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
        online_column_inverse(const_mat_view<T>& A) 
            : _A{ A }
            , _indices(A.N, false)
            , _n{ 0 }
        {
            _inv.reserve(10 * 10);
            _A_sub_t.reserve(10 * A.M);
        }
        
        const_mat_view<T> insert(const uint32_t column_idx)
        {
            assert(column_idx < _A.N);
            
            if (_indices[column_idx]) {
                return inverse();
            }

            auto const M = _A.M;
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
                auto v_col = make_unique<T[]>(M);

                /* copy col from A */
                blas::xcopy(M, _A.col(column_idx), _A.N, v_col.get(), 1);

// Py           u1 = np.dot(matA.T, vCol)
                auto u1 = make_unique<T[]>(_n);

                {
                    /* current view of sub_A_t */
                    mat_view<T> At = subset_transposed();

                    blas::xgemv(CblasRowMajor, CblasNoTrans, At.M, At.N, 1.0, At.data(), At.N, 
                        v_col.get(), 1, 0.0, 
                        u1.get(), 1);
                }

// Py           u2 = np.dot(current_inverse, u1)
                auto u2 = make_unique<T[]>(_n);
                blas::xgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0, _inv.data(),
                    _n, u1.get(), 1, 0.0, u2.get(), 1);

// Py           d = 1.0 / float(np.dot(vCol.T, vCol) - np.dot(u1.T, u2))
                T d {1.0};
                {
                    T a = blas::xdot(M, v_col.get(), 1, v_col.get(), 1);
                    T b = blas::xdot(_n, u1.get(), 1, u2.get(), 1);

                    d /= a - b;
                }

// Py           u3 = d * u2
                auto u3 = make_unique<T[]>(_n);

                std::copy_n(u2.get(), _n, u3.get());
                blas::xscal(_n, d, u3.get(), 1);

// Py           F11inv = current_inverse + (d * np.outer(u2, u2.T))
                mat<T> F11inv(_n, _n);
                std::copy_n(_inv.data(), _inv.size(), F11inv.data());
                
// Py           A := alpha*x*y**T + A
                blas::xger(CblasRowMajor, _n, _n, d, 
                    u2.get(), 1, 
                    u2.get(), 1, 
                    F11inv.data(), _n);

                /* assign F11inv */
                {
                    uint32_t new_n { _n + 1 };
                    _inv.assign(new_n * new_n, 0);
                    mat_view<T> mut_inv { new_n, new_n, _inv.data() };

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

        const_mat_view<T> remove(uint32_t column_idx)
        {
            assert(_n > 0);
            assert(column_idx < _A.N);

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
                    mat_view<T> mut_inv{ _n, _n, _inv.data() };

                    /* calculate column to remove */
                    size_t idx{ insertion_index(column_idx) };

                    {
                        mat_view<T> As_t{ subset_transposed() };
                        /* shift row to the bottom */
                        shift_row(As_t, idx, As_t.M - 1);
                    }

                    /* shift to position */
                    shift_column(mut_inv, idx, mut_inv.N - 1);
                    shift_row(mut_inv, idx, mut_inv.N - 1);
                }

                uint32_t new_n{ _n - 1 };
                mat<T> F11inv(new_n, new_n);
                {
                    const_mat_view<T> inv{ _n, _n, _inv.data() };

// Py               #update the inverse by removing the last column
// Py               d = current_inverse[N - 1, N - 1]
                    T d{ inv(new_n, new_n) };

// Py               u3 = -1.0 * current_inverse[0:N - 1, N - 1]
                    /* copy last column */
                    auto u3 = make_unique<T[]>(new_n);
                    blas::xcopy(new_n, inv.col(new_n), new_n + 1, u3.get(), 1);
                    blas::xscal(new_n, -1, u3.get(), 1);

// py               u2 = (1.0 / d) * u3
                    auto u2 = make_unique<T[]>(new_n);
                    blas::xcopy(new_n, u3.get(), 1, u2.get(), 1);
                    blas::xscal(new_n, (T)1.0 / d, u2.get(), 1);

// Py               F11inv = current_inverse[0:N - 1, 0 : N - 1]
                    std::fill_n(F11inv.data(), F11inv.size, (T)0.0);

// Py               new_inverse = F11inv - (d* np.outer(u2, u2.T))
                    /* A := alpha*x*y**T + A,  */
                    blas::xger(CblasRowMajor, new_n, new_n, d,
                        u2.get(), 1,
                        u2.get(), 1,
                        F11inv.data(), new_n);

                    /* slightly awkward.. */
                    for (size_t r{ 0 }; r < F11inv.M; ++r) {
                        for (size_t c{ 0 }; c < F11inv.N; ++c) {
                            F11inv(r, c) = inv(r, c) - F11inv(r, c);
                        }
                    }
                }
                    
                /* resize and assign */
                _inv.assign(F11inv.size, 0);
                std::copy_n(F11inv.data(), F11inv.size, _inv.data());
            }

            _indices[column_idx] = false;
            _n--;

            return inverse();
        }

        const_mat_view<T> flip(const uint32_t index) {
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
        const_mat_view<T> inverse() 
        {
            assert(_inv.size() >= _n * _n);
            return { _n, _n, _inv.data() };
        }

    private:

        mat_view<T> subset_transposed()
        {
            assert(_A_sub_t.size() >= _n * _A.M);
            return { _n, _A.M, _A_sub_t.data() };
        }

        size_t insertion_index(uint32_t column_idx) 
        {
            assert(column_idx < _A.N);

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
            const const_mat_view<T>& A,
            const size_t src_col,
            const size_t dest_row
            )
        {
            auto n = A.M;
            auto it = v.insert(v.begin() + (dest_row * n), n, 0.0);
            
            blas::xcopy(n, A.col(src_col), A.N, &*it, 1);
        }

        /* original matrix */
        const_mat_view<T>& _A;
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
        const const_mat_view<T>& A,
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
        auto A_x = make_unique<T[]>(A.M);
        blas::xgemv(CblasRowMajor, CblasNoTrans, A.M, A.N, 1.0, A.data(), 
            A.N, x_previous, 1, 0.0, A_x.get(), 1);

// Py   difference = np.zeros(len(y))
// Py   for i in range(0, len(y)) :
// Py       difference[i] = y[i] - A_x[i]
        
        /* A_x becomes y - A_x */
        for (size_t i = 0; i < A.M; i++) {
            A_x[i] = y[i] - A_x[i];
        }

// Py   return np.dot(A_t, difference)
        blas::xgemv(CblasRowMajor, CblasTrans, A.M, A.N, 1.0, A.data(),
            A.N, A_x.get() /* difference */, 1, 0.0, c, 1);
    }

    template<typename T>
    std::pair<T, uint32_t> find_max_gamma(
        const const_mat_view<T>& A,
        const T* res_vec, 
        const T* x,
        const T* dir_vec, 
        const T c_inf, 
        const std::vector<bool>& lambda_indices
        )
    {
        /* evaluate the eligible elements of transpose(A) * A * dir_vec */
        /* p = Ad */
        auto p = make_unique<T[]>(A.M);
        blas::xgemv(CblasRowMajor, CblasNoTrans, A.M, A.N, 1.0, A.data(),
            A.N, dir_vec, 1, 0.0, p.get(), 1);

        /* q = transpose(A)p */
        auto q = make_unique<T[]>(A.N);
        blas::xgemv(CblasRowMajor, CblasTrans, A.M, A.N, 1.0, A.data(),
            A.N, p.get(), 1, 0.0, q.get(), 1);

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
        const_mat_view<T>& A,
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
        std::fill_n(x, A.N, (T)0.0);

        /* initialise residual vector */
// Py   c_vec = residual_vector(A, y, x)
        mat<T> c_vec(1, A.N);
        residual_vector(A, y, x, c_vec.data());

        /* initialise lambda = || c_vec || _inf */
        // Py   c_inf = (np.linalg.norm(c_vec, np.inf))
        T c_inf{ 0.0 };

        auto direction_vec = make_unique<T[]>(A.N);
        online_column_inverse<T> inv(A);

        {
            size_t c_inf_i;
            c_inf = inf_norm(c_vec.data(), A.N, &c_inf_i);

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
                auto const gamma = find_max_gamma(A, c_vec.data(), x,
                    direction_vec.get(), c_inf, inv.indices());

                /* update inverse by inserting/removing the
                respective index from the inverse */
                inv.flip(gamma.second);

                /* update x */
                for (size_t i{ 0 }; i < A.N; i++) {
                    x[i] += gamma.first * direction_vec[i];
                }
            }

            /* update residual vector */
// Py       c_vec = residual_vector(A, y, x)
            residual_vector(A, y, x, c_vec.data());

            /* update direction vector */
            {
// Py           c_vec_gamma = helper.subset_array(c_vec, lambda_indices)
                const uint32_t N{ inv.N() };

                mat<T> c_vec_gamma(1, N);
                mat_subset_cols(c_vec, inv.indices(), c_vec_gamma);

                // Py           direction_vec = np.dot(invAtA, helper.sign_vector(c_vec_gamma))
                sign_vector(N, c_vec_gamma.data(), c_vec_gamma.data(), tolerance);
                auto dir_tmp = make_unique<T[]>(N);

                blas::xgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0,
                    inv.inverse().data(), N,
                    c_vec_gamma.data(), 1,
                    0.0,
                    dir_tmp.get(), 1);

// Py           direction_vec = helper.zero_mask(direction_vec, lambda_indices, N)
                zero_mask(inv.indices(), dir_tmp.get(), direction_vec.get());
            }

            /* find lambda(i.e., infinite norm of residual vector) */
            c_inf = inf_norm(c_vec.data(), A.N);

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
        if (A.N == 1 /* vector */) {
            T sum{ 0.0 };
            columnwise_sum(A, &sum);

            if (sum <= 0) { return false; }

            for (size_t i{ 0 }; i < A.M; ++i) {
                A(i, 0) /= sum;
            }
        }
        else {
            /* matrix */
            auto sums = make_unique<T[]>(A.N);
            columnwise_sum(A, sums.get());

            for (size_t i{ 0 }; i < A.N; ++i) {
                if (sums[i] <= 0) { return false; }
            }

            for (size_t r{ 0 }; r < A.M; ++r) {
                for (size_t c{ 0 }; c < A.N; ++c) {
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

        frutils::blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A,
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

        const_mat_view<float> A(A_m, A_n, A_);
        return run_solver(A, max_iter, tolerance, y, x);
    }
}