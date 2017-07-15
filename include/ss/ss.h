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

#include "ss/ndspan.h"
#include <kernelpp/types.h>

namespace ss
{
    /* Solvers ------------------------------------------------------------- */

    struct homotopy_report
    {
        /* the number of iterations performed. */
        std::uint32_t iter;

        /* the solution error */
        double solution_error;
    };

    /* make std::variant happy */
    inline bool operator== (const homotopy_report&, const homotopy_report&) { return false; }

    /*  Uses the homotopy method to solve the equation
        min || x || _1  subject to A x = y

                     A : sensing matrix of row-major order
                     y : signal vector of length m
        max_iterations : maximum number of iterations
             tolerance : sparsity budget
                     x : the output sparse representation vector
                         of length n

        returns : an instance of homotopy_report, or an error
    */
    struct homotopy
    {
        homotopy();
        ~homotopy();

        kernelpp::maybe<homotopy_report> solve(
            const ndspan<float, 2> A,
            const ndspan<float>    y,
                  float            tolerance,
                  std::uint32_t    max_iterations,
                  ndspan<float>    x);

        kernelpp::maybe<homotopy_report> solve(
            const ndspan<double, 2> A,
            const ndspan<double>    y,
                  double            tolerance,
                  std::uint32_t     max_iterations,
                  ndspan<double>    x);

      private:
        struct state;
        std::unique_ptr<state> m;
    };

    /* Utilities ----------------------------------------------------------- */

    /*  computes A x
        Reconstructs a signal given the sparse representation
        of that signal.

        A : input matrix A used to construct x
        x : the sparse representation vector,
            of length n
        y : the output of the reconstructed signal,
            of length m
    */
    void reconstruct_signal(
        const ndspan<float, 2> A, const ndspan<float> x, ndspan<float> y);

    void reconstruct_signal(
        const ndspan<double, 2> A, const ndspan<double> x, ndspan<double> y);


    /*  Normalizes the columns of a given matrix in-place according
        to the L1-norm of each column.

        A : input matrix A to be normalized
    */
    void norm_l1(ndspan<float, 2> A);

    void norm_l1(ndspan<double, 2> A);
}