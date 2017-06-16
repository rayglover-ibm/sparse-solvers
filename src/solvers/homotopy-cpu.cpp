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

#include "linalg/common.h"
#include "linalg/online_inverse.h"

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
    size_t vec_subset(
        const xt::xtensor<T, 1>& X,
        const std::vector<bool>& indices,
        xt::xtensor<T, 1>& X_subset
        )
    {
        size_t i{ 0 }, n{ 0 };
        for (const bool val : indices) {
            if (val) {
                X_subset[n] = X[i];
                ++n;
            }
            ++i;
        }
        return n;
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
        for (size_t i{ 0 }, off{ 0 }; i < mask.size(); i++) {
            y[i] = mask[i] ? x[off++] : T(0);
        }
    }

#pragma endregion

    template<typename T>
    void residual_vector(
        const mat_view<T>& A,
        const T* y,
        const T* x_previous,
        T* c
        )
    {
        size_t m = dim<0>(A), n = dim<1>(A);

// Py   A_t = np.matrix.transpose(A)
        /* note(port): The transpose is not evaluated here and
           is instead computed at the blas function when evaluating
           'np.dot(A_t, difference)'
         */
// Py   A_x = np.dot(A, x_previous)
        auto A_x = make_unique<T[]>(m);
        blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
            A.cbegin(), n,
            x_previous, 1, 0.0,
            A_x.get(), 1);

// Py   difference = np.zeros(len(y))
// Py   for i in range(0, len(y)) :
// Py       difference[i] = y[i] - A_x[i]

        /* A_x = y - A_x */
        for (size_t i = 0; i < m; i++) {
            A_x[i] = y[i] - A_x[i];
        }

// Py   return np.dot(A_t, difference)
        blas::xgemv(CblasRowMajor, CblasTrans, m, n, 1.0,
            A.cbegin(), n,
            A_x.get() /* difference */, 1, 0.0,
            c, 1);
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
        blas::xgemv(CblasRowMajor, CblasNoTrans, dim<0>(A), dim<1>(A), 1.0,
            A.cbegin(), dim<1>(A),
            dir_vec, 1, 0.0,
            p.get(), 1);

        /* q = transpose(A) p */
        auto q = make_unique<T[]>(dim<1>(A));
        blas::xgemv(CblasRowMajor, CblasTrans, dim<0>(A), dim<1>(A), 1.0,
            A.cbegin(), dim<1>(A),
            p.get(), 1, 0.0,
            q.get(), 1);

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

                if (di_left != 0.0) {
                    T leftT = (c_inf - res_vec[i]) / di_left;
                    if (leftT > 0.0 && leftT < min) {
                        min = leftT;
                    }
                }

                if (di_right != 0.0) {
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
        const ndspan<T> y,
        ndspan<T> x
        )
    {
        assert(max_iter > 0);
        assert(y.size() == dim<0>(A));
        assert(x.size() == dim<1>(A));

        /* using a tolerance lt epsilon is generally not good */
        assert(tolerance >= std::numeric_limits<T>::epsilon()
            && tolerance < 1.0);

        /* initialise x */
        xt::view(x) = T(0);

        auto c         = xt::xtensor<T, 1>::from_shape({ dim<1>(A) });
        auto c_gamma   = xt::xtensor<T, 1>::from_shape({ dim<1>(A) });
        T    c_inf     = T(0);

        auto direction = xt::xtensor<T, 1>::from_shape({ dim<1>(A) });
        online_column_inverse<T> inv(A);

        /* initialise residual vector */
// Py   c_vec = residual_vector(A, y, x)
        residual_vector(A, y.begin(), x.begin(), c.begin());

        /* initialise lambda = || c_vec || _inf */
        {
            size_t c_inf_i;
// Py       c_inf = (np.linalg.norm(c_vec, np.inf))
            c_inf = inf_norm(c.cbegin(), dim<1>(A), &c_inf_i);

            T c_vec_gamma{ c_inf };
            T subsample_direction_vector{ 0.0 };
            inv.insert((uint32_t)c_inf_i);

// Py       subsample_direction_vector = invAtA * helper.sign_vector(c_vec_gamma)
            sign_vector(1, &c_vec_gamma, &subsample_direction_vector, tolerance);
            subsample_direction_vector *= inv.inverse()(0, 0);

// Py       direction_vec = helper.zero_mask(subsample_direction_vector, lambda_indices, N)
            zero_mask(inv.indices(), &subsample_direction_vector, direction.begin());
        }

        /* evaluate homotopy path segments in iterations */
        std::uint32_t iter{ 0u };
        while (iter++ < max_iter)
        {
            auto const gamma = find_max_gamma(A, c.cbegin(), x.begin(),
                direction.begin(), c_inf, inv.indices());

            /* update inverse by inserting/removing the
                respective index from the inverse */
            inv.flip(gamma.second);

            /* update x */
            xt::view(x) += gamma.first * direction;

            /* update residual vector */
// Py       c_vec = residual_vector(A, y, x)
            residual_vector(A, y.begin(), x.begin(), c.begin());

            /* update direction vector */
            {
                const std::vector<bool>& mask = inv.indices();

// Py           c_vec_gamma = helper.subset_array(c_vec, lambda_indices)
                const uint32_t N = vec_subset(c, mask, c_gamma);

// Py           direction_vec = np.dot(invAtA, helper.sign_vector(c_vec_gamma))
                sign_vector(N, c_gamma.begin(), c_gamma.begin(), tolerance);

                auto dir_tmp = make_unique<T[]>(N);
                blas::xgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0,
                    inv.inverse().cbegin(), N,
                    c_gamma.begin(), 1, 0.0,
                    dir_tmp.get(), 1);

// Py           direction_vec = helper.zero_mask(direction_vec, lambda_indices, N)
                zero_mask(mask, dir_tmp.get(), direction.begin());
            }

            /* find lambda(i.e., infinite norm of residual vector) */
            c_inf = inf_norm(c.cbegin(), dim<1>(A));

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

    template <> kernelpp::variant<homotopy_report, error_code>
    solve_homotopy::op<compute_mode::CPU>(
        const float* A_,
        const std::uint32_t A_m,
        const std::uint32_t A_n,
        const std::uint32_t max_iter,
        const float tolerance,
        const float* y_,
        float* x_
        )
    {
        assert(A_m > 0
            && A_n > 0
            && A_ != nullptr);

        const ndspan<float, 2> A = ss::as_span<2, float>(A_, { A_m, A_n });
        const ndspan<float>    y = ss::as_span<1, float>(y_, { A_m });
        ndspan<float>          x = ss::as_span<1, float>(x_, { A_n });

        return run_solver(A, max_iter, tolerance, y, x);
    }
}