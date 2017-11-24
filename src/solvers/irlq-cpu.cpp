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

#include "irlq.h"

#include "linalg/common.h"
#include "linalg/blas_wrapper.h"

#include <assert.h>

namespace ss
{
    template <typename T>
    irlq_report run_solver(
        const ndspan<T, 2> Q,
        const std::uint32_t max_iter,
        const T tolerance,
        const ndspan<T> y,
        ndspan<T> x)
    {
        return { 0, 1 };       
    }

    template <> kernelpp::variant<irlq_report, error_code>
    solve_irlq::op<compute_mode::CPU, float>(
        const ndspan<float, 2> Q,
        const ndspan<float> y,
        float tolerance,
        std::uint32_t max_iterations,
        ndspan<float> x)
    {
        return run_solver<float>(A, max_iterations, tolerance, y, x);
    }

    template <> kernelpp::variant<irlq_report, error_code>
    solve_irlq::op<compute_mode::CPU, double>(
        const ndspan<double, 2> Q,
        const ndspan<double> y,
        double tolerance,
        std::uint32_t max_iterations,
        ndspan<double> x)
    {
        return run_solver<double>(A, max_iterations, tolerance, y, x);
    }
}