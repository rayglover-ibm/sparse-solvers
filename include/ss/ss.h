#pragma once

#include "ss/ndspan.h"

#include <kernelpp/types.h>
#include <gsl.h>
#include <array>

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

    struct homotopy
    {
        homotopy();
		~homotopy();

        /*
            homotopy::solve
            --------------------
            uses the homotopy method to solve the equation
            min || x || _1  subject to A x = y

                    A : sensing matrix of row-major order
                    m : the number of rows in matrix A
                    n : the number of columns in matrix A
             max_iter : maximum number of iterations
            tolerance : sparsity budget
                    y : signal vector of length m
                    
                    x : the output sparse representation vector
                        of length n

            returns : an instance of homotopy_report, or an error
        */
        kernelpp::maybe<ss::homotopy_report> solve(
            const ndspan<float, 2> A,
            const gsl::span<float> y,
                  float            tolerance,
                  std::uint32_t    max_iterations,
                  gsl::span<float> x);
        
		
		struct state;
        std::unique_ptr<state> m;
    };
}