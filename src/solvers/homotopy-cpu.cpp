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
#include <assert.h>

namespace ss
{
    /* The following is a port:
          of ./tools/sparsity/src/Homotopy.py,
          at sha1 20b980c7804883d059896e04c3a0047615cbd984,
          committed 2015-11-09 14:08:24
    */
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
        size_t cardinality,
        const std::vector<bool>& indices
        )
    {
        size_t i = indices.size() - 1, j = cardinality - 1;

        for (auto it = indices.crbegin(); it != indices.crend(); ++it, --i) {
            direction[i] = (*it) ? direction[j--] : T(0);
        }
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

// Py   A_x = np.dot(A, x_previous)
        xt::xtensor<T, 1> A_x = y;

        /* A_x = y - A_x */
        blas::xgemv(CblasRowMajor, CblasNoTrans, m, n, -1.0,
            A.cbegin(), n,
            x_previous.cbegin(), 1, 1.0,
            A_x.begin(), 1);

// Py   return np.dot(np.matrix.transpose(A), difference)
        blas::xgemv(CblasRowMajor, CblasTrans, m, n, 1.0,
            A.cbegin(), n,
            A_x.cbegin() /* difference */, 1, 0.0,
            c.begin(), 1);
    }

    template<typename T>
    std::pair<T, size_t> find_max_gamma(
        const mat_view<T>& A,
        const ndspan<T> c,
        const ndspan<T> x,
        const ndspan<T> direction,
        const T c_inf,
        const std::vector<bool>& lambda_indices
        )
    {
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
        for (size_t i{ 0u }; i < lambda_indices.size(); i++) {
            const T prev = min;
            if (lambda_indices[i]) {
                T minT = -x[i] / direction[i];
                if (minT > 0.0 && minT < min) {
                    min = minT;
                }
            }
            else {
                T di_left{ 1.0f - q[i] }, di_right{ 1.0f + q[i] };

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
        xt::view(x) = T(0);

        auto direction = xt::xtensor<T, 1>::from_shape({ N });
        auto c         = xt::xtensor<T, 1>::from_shape({ N });
        auto c_gamma   = xt::xtensor<T, 1>::from_shape({ N });
        T    c_inf     = T(0);

        online_column_inverse<T> inv(A);

        /* initialise residual vector */
        residual_vector(A, y, x, as_span(c));

        /* initialise lambda = || c_vec || _inf */
        {
            size_t idx;
            c_inf = inf_norm(as_span(c), &idx);

            inv.insert((uint32_t)idx);

            T c_gamma{ c_inf };
            sign(as_span(&c_gamma, { 1 }), tolerance);

            /* initialize direction */
            direction[0] = c_gamma * inv.inverse()(0, 0);
            expand(direction, 1, inv.indices());
        }

        /* evaluate homotopy path segments in iterations */
        std::uint32_t iter{ 0u };
        while (iter < max_iter)
        {
            iter++;
            auto const gamma = find_max_gamma(A, as_span(c), x,
                as_span(direction), c_inf, inv.indices());

            /* update inverse by inserting/removing the
                respective index from the inverse */
            inv.flip(gamma.second);

            /* update x */
            xt::view(x) += gamma.first * direction;

            /* update residual vector */
            residual_vector(A, y, x, as_span(c));

            /* update direction vector */
            {
                const std::vector<bool>& mask = inv.indices();
                const size_t K = vec_subset(c, mask, c_gamma);

                sign(as_span(c_gamma.begin(), K), tolerance);

                blas::xgemv(CblasRowMajor, CblasNoTrans, K, K, 1.0,
                    inv.inverse().cbegin(), K,
                    c_gamma.begin(), 1, 0.0,
                    direction.begin(), 1);

                /* expand the direction vector, filling with 0's where mask[i] == false */
                expand(direction, K, mask);
            }

            /* find lambda(i.e., infinite norm of residual vector) */
            c_inf = inf_norm(as_span(c));

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
            auto sums = std::make_unique<T[]>(dim<1>(A));
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

    template <> kernelpp::variant<homotopy_report, error_code>
    solve_homotopy::op<compute_mode::CPU>(
        const ndspan<float, 2> A,
        const ndspan<float> y,
        float tolerance,
        std::uint32_t max_iterations,
        ndspan<float> x)
    {
        return run_solver<float>(A, max_iterations, tolerance, y, x);
    }
}