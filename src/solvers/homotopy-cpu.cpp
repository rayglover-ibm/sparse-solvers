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

#include "linalg/common.h"
#include "linalg/blas_wrapper.h"
#include "linalg/online_inverse.h"
#include "linalg/rank_index.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <tuple>
#include <assert.h>

namespace ss
{
    template <typename T>
    T inf_norm(const ndspan<T> v, size_t* idx)
    {
        *idx = blas::ixamax(dim<0>(v), v.storage_cbegin(), 1);
        return std::abs(v[*idx]);
    }

    template <typename T>
    T inf_norm(const ndspan<T> v)
    {
        size_t idx;
        return inf_norm(v, &idx);
    }

    template <typename T>
    void vec_subset(
        const xt::xtensor<T, 1>& X,
        const rank_index<uint32_t>& indices,
        xt::xtensor<T, 1>& X_subset
        )
    {
        size_t n = 0;
        for (const uint32_t i : indices) {
            X_subset[n] = X[i];
            ++n;
        }
    }

    template <typename T>
    void sign(ndspan<T> x, const T tol) {
        for (T& val : x) {
            if      (val > tol)  { val =  1; }
            else if (val < -tol) { val = -1; }
            else                 { val =  0; }
        }
    }

    template <typename T>
    void expand(
        xt::xtensor<T, 1>& direction,
        const rank_index<uint32_t>& indices)
    {
        assert(indices.size() <= dim<0>(direction));

        int i = dim<0>(direction) - 1,
            j = indices.size() - 1;

        for (auto it = indices.crbegin(); it != indices.crend(); ++it)
        {
            while (i > *it) { direction[i--] = T(0); }
            direction[i--] = direction[j--];
        }
        while (i >= 0) { direction[i--] = T(0); }
    }

    template <typename T>
    void residual_vector(
        const mat_view<T> A,
        const ndspan<T> y,
        const ndspan<T> x_previous,
        ndspan<T> c
        )
    {
        xt::xtensor<T, 1> A_x = y;

        blas::xgemv<T>(CblasNoTrans, -1.0, A, x_previous, 1.0, A_x);
        blas::xgemv<T>(CblasTrans,    1.0, A, A_x,        0.0, c);
    }

    template <typename T>
    std::pair<T, size_t> find_max_gamma(
        const mat_view<T> A,
        const ndspan<T> c,
        const ndspan<T> x,
        const ndspan<T> direction,
        const T c_inf,
        const rank_index<uint32_t>& lambda_indices
        )
    {
        assert(lambda_indices.size() <= dim<1>(A));

        /* evaluate the eligible elements of transpose(A) * A * dir_vec */
        const size_t m = dim<0>(A), n = dim<1>(A);

        /* p = Ad */
        auto p = xt::xtensor<T, 1>::from_shape({ m });
        blas::xgemv<T>(CblasNoTrans, 1.0, A, direction, 0.0, p);

        /* q = transpose(A) p */
        auto q = xt::xtensor<T, 1>::from_shape({ n });
        blas::xgemv<T>(CblasTrans, 1.0, A, p, 0.0, q);

        /* evaluate the competing lists of terms */
        T min{ std::numeric_limits<T>::max() };
        size_t idx{ 0u };

        /* find the minimum term and its index */
        auto ldx = std::begin(lambda_indices);
        auto end = std::end(lambda_indices);

        for (size_t i{ 0u }; i < n; i++) {
            const T prev = min;

            if (ldx != end && *ldx == i) {
                T minT = -x[i] / direction[i];
                if (minT > 0.0 && minT < min) {
                    min = minT;
                }
                ldx++;
            }
            else {
                T di_left{ T(1) - q[i] }, di_right{ T(1) + q[i] };

                if (di_left != 0.0) {
                    T leftT = (c_inf - c[i]) / di_left;
                    if (leftT > 0.0 && leftT < min) {
                        min = leftT;
                    }
                }
                if (di_right != 0.0) {
                    T rightT = (c_inf + c[i]) / di_right;
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

    template <typename T>
    void inverse_add_or_remove(
        const mat_view<T>&        A,
        size_t                    A_col,
        rank_index<uint32_t>&     lambda_indices,
        online_column_inverse<T>& inv)
    {
        int rank = lambda_indices.rank_of(A_col);
        if (rank >= 0) {
            lambda_indices.erase(A_col);
            inv.remove(rank);
        }
        else {
            rank = lambda_indices.insert(A_col);
            auto col = xt::view(A, xt::all(), A_col);
            inv.insert(rank, col.cbegin(), col.cend());
        }
    }

    template <typename T>
    homotopy_report run_solver(
        const ndspan<T, 2> A,
        const std::uint32_t max_iter,
        const T tolerance,
        const ndspan<T> y,
        ndspan<T> x
        )
    {
        assert(max_iter > 0
            && y.size() == dim<0>(A)
            && x.size() == dim<1>(A));

        /* using a tolerance lt epsilon is generally not good */
        assert(tolerance >= std::numeric_limits<T>::epsilon()
            && tolerance < 1.0);

        const size_t N = dim<1>(A);

        /* initialise x */
        view(x) = T(0);

        auto direction = xt::xtensor<T, 1>::from_shape({ N });
        auto c         = xt::xtensor<T, 1>::from_shape({ N });
        auto c_gamma   = xt::xtensor<T, 1>::from_shape({ N });
        T    c_inf     = T(0);

        rank_index<uint32_t> lambda_indices;
        online_column_inverse<T> inv(dim<0>(A), size_t(log(N)));

        /* initialise residual vector */
        residual_vector(A, y, x, as_span(c));

        {   /* initialise lambda = || c_vec || _inf */
            size_t idx;
            c_inf = inf_norm(as_span(c), &idx);

            inverse_add_or_remove(A, idx, lambda_indices, inv);

            T c_gamma{ c_inf };
            sign(as_span(&c_gamma, { 1 }), tolerance);

            /* initialize direction */
            direction[0] = c_gamma * inv.inverse()(0, 0);
            expand(direction, lambda_indices);
        }

        /* evaluate homotopy path segments in iterations, stopping if
             - the infinity norm of residual vector is within tolerance
             - the residual vector length reaches zero 
         */
        std::uint32_t iter{ 0u };
        do {
            iter++;

            T min; size_t idx;

            std::tie(min, idx) = find_max_gamma(A, as_span(c), x,
                as_span(direction), c_inf, lambda_indices);

            /* update inverse by inserting/removing the
               respective index from the inverse */
            inverse_add_or_remove(A, idx, lambda_indices, inv);

            auto K = lambda_indices.size();
            if (K == 0) { break; }

            /* update x */
            ss::view(x) += min * direction;

            /* update residual vector */
            residual_vector(A, y, x, as_span(c));

            {   /* update direction vector */
                /* produce a subset of c and map to -1,0,+1 */
                vec_subset(c, lambda_indices, c_gamma);
                sign(as_span(c_gamma.storage_begin(), K), tolerance);

                /* update */
                blas::xgemv<T>(CblasNoTrans, 1.0, inv.inverse(), c_gamma, 0.0, direction);

                /* expand the direction vector, filling with 0's where mask[i] == false */
                expand(direction, lambda_indices);
            }

            /* find lambda (i.e., infinity norm of residual vector) */
            c_inf = inf_norm(as_span(c));
        }
        while (iter < max_iter && c_inf > tolerance);
        
        return{ iter, c_inf };
    }

    template <> kernelpp::variant<homotopy_report, error_code>
    solve_homotopy::op<compute_mode::CPU, float>(
        const ndspan<float, 2> A,
        const ndspan<float> y,
        float tolerance,
        std::uint32_t max_iterations,
        ndspan<float> x)
    {
        return run_solver<float>(A, max_iterations, tolerance, y, x);
    }

    template <> kernelpp::variant<homotopy_report, error_code>
    solve_homotopy::op<compute_mode::CPU, double>(
        const ndspan<double, 2> A,
        const ndspan<double> y,
        double tolerance,
        std::uint32_t max_iterations,
        ndspan<double> x)
    {
        return run_solver<double>(A, max_iterations, tolerance, y, x);
    }
}