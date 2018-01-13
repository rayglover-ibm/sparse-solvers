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

#include "ss/fwd.h"
#include "ss/ndspan.h"
#include "ss/policies.h"

#include <kernelpp/types.h>

namespace ss
{
    /* Solver base --------------------------------------------------------- */

    template <typename T, typename SolverPolicy>
    struct solver
    {   
        using report_type  = typename SolverPolicy::report_type;
        using state_type   = typename SolverPolicy::template state_type<T>;
        using solve_result = kernelpp::maybe<report_type>;

        /* A : non-owning view of a sensing matrix */
        solver(const ndspan<T, 2> A);

        ~solver() = default;
        
        /*  Uses the SolverPolicy to solve the equation
         *    min || x || _1  subject to A x = y
         *
         *                 y : signal vector of length m
         *    max_iterations : maximum number of iterations
         *               tol : sparsity budget
         *                 x : the output sparse representation vector
         *                     of length n
         *
         *    returns : an instance of report_type, or an error
         */
        solve_result solve(const ndspan<T> y, T tol, std::uint32_t max_iterations, ndspan<T> x);

        solver(solver<T, SolverPolicy>&& other) : m{ std::move(other.m) } {}

      private:
        std::unique_ptr<state_type> m;
    };

    /* Solver types  ------------------------------------------------------- */

    template <typename T>
    using homotopy = solver<T, homotopy_policy>;

    template <typename T>
    using irls = solver<T, irls_policy>;


    /* Utilities ----------------------------------------------------------- */

    /*  computes A x
     *  Reconstructs a signal given the sparse representation
     *  of that signal.
     *
     *    A : input matrix A used to construct x
     *    x : the sparse representation vector,
     *        of length n
     *    y : the output of the reconstructed signal,
     *        of length m
     */
    void reconstruct_signal(
        const ndspan<float, 2> A, const ndspan<float> x, ndspan<float> y);

    void reconstruct_signal(
        const ndspan<double, 2> A, const ndspan<double> x, ndspan<double> y);


    /*  Normalizes the columns of a given matrix in-place according
     *  to the L1-norm of each column.
     *
     *    A : input matrix A to be normalized
     */
    void norm_l1(ndspan<float, 2> A);

    void norm_l1(ndspan<double, 2> A);


    /* Definitions --------------------------------------------------------- */
    
    template <typename T, typename S>
    solver<T, S>::solver(const ndspan<T, 2> A)
        : m(std::make_unique<state_type>(A))
    {
        static_assert(
            detail::is_solver<S, T>::value,
            "The specified solver policy does not implment the required interface");
    }

    template <typename T, typename S>
    typename solver<T, S>::solve_result solver<T, S>::solve(
        const ndspan<T>     y,
              T             tolerance,
              std::uint32_t max_iterations,
              ndspan<T>     x)
    {
        return S::run(*m, y, tolerance, max_iterations, x);
    }
}