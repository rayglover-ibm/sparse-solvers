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

#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <tuple>
#include <assert.h>

namespace ss
{
    template<typename T>
    T inf_norm(const ndspan<T> v, size_t* idx)
    {
        *idx = blas::ixamax(dim<0>(v), v.cbegin(), 1);
        return std::abs(v[*idx]);
    }

    template<typename T>
    T inf_norm(const ndspan<T> v)
    {
        size_t idx;
        return inf_norm(v, &idx);
    }

    template<typename T>
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

    template<typename T>
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

    template<typename T>
    void residual_vector(
        const mat_view<T> A,
        const ndspan<T> y,
        const ndspan<T> x_previous,
        ndspan<T> c
        )
    {
        const size_t m = dim<0>(A), n = dim<1>(A);
        xt::xtensor<T, 1> A_x = y;

        /* A_x = y - A_x */
        blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, -1.0,
            A.cbegin(), n,
            x_previous.cbegin(), 1, 1.0,
            A_x.begin(), 1);

        blas::xgemv(CblasRowMajor, CblasTrans, m, n, 1.0,
            A.cbegin(), n,
            A_x.cbegin() /* difference */, 1, 0.0,
            c.begin(), 1);
    }

    template<typename T>
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
        blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0,
            A.cbegin(), n,
            direction.cbegin(), 1, 0.0,
            p.begin(), 1);

        /* q = transpose(A) p */
        auto q = xt::xtensor<T, 1>::from_shape({ n });
        blas::xgemv(CblasRowMajor, CblasTrans, m, n, 1.0,
            A.cbegin(), n,
            p.cbegin(), 1, 0.0,
            q.begin(), 1);

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

    template<typename T>
    homotopy_report run_solver(
        const mat_view<T>& A,
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

        online_column_inverse<T> inv(A.shape());

        /* initialise residual vector */
        residual_vector(A, y, x, as_span(c));

        /* initialise lambda = || c_vec || _inf */
        {
            size_t idx;
            c_inf = inf_norm(as_span(c), &idx);

            auto col = xt::view(A, xt::all(), idx);
            inv.insert(idx, col.cbegin(), col.cend());

            T c_gamma{ c_inf };
            sign(as_span(&c_gamma, { 1 }), tolerance);

            /* initialize direction */
            direction[0] = c_gamma * inv.inverse()(0, 0);
            expand(direction, inv.indices());
        }

        /* evaluate homotopy path segments in iterations */
        std::uint32_t iter{ 0u };
        while (iter < max_iter)
        {
            iter++;
            {
                T min; size_t idx;

                std::tie(min, idx) = find_max_gamma(A, as_span(c), x,
                    as_span(direction), c_inf, inv.indices());

                /* update inverse by inserting/removing the
                   respective index from the inverse */
                if (inv.indices().rank_of(idx) >= 0)
                    inv.remove(idx);
                else {
                    auto col = xt::view(A, xt::all(), idx);
                    inv.insert(idx, col.cbegin(), col.cend());
                }

                /* update x */
                ss::view(x) += min * direction;
            }

            /* update residual vector */
            residual_vector(A, y, x, as_span(c));

            /* update direction vector */
            {
                const rank_index<uint32_t>& mask = inv.indices();
                size_t K = mask.size();

                vec_subset(c, mask, c_gamma);
                sign(as_span(c_gamma.begin(), K), tolerance);

                blas::xgemv(CblasRowMajor, CblasNoTrans, K, K, 1.0,
                    inv.inverse().cbegin(), K,
                    c_gamma.begin(), 1, 0.0,
                    direction.begin(), 1);

                /* expand the direction vector, filling with 0's where mask[i] == false */
                expand(direction, mask);
            }

            /* find lambda(i.e., infinite norm of residual vector) */
            c_inf = inf_norm(as_span(c));

            /* check if infinity norm of residual vector is within tolerance */
            if (c_inf < tolerance)
                break;
        }
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