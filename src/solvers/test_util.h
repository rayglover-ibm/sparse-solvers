#include <ss/ss.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

using xt::xtensor;
using ss::as_span;

namespace
{
    template <typename Report>
    void check_report(kernelpp::maybe<Report>& result,
        float tolerance, uint32_t max_iterations
        )
    {
        EXPECT_TRUE(result.template is<Report>());
        auto r = result.template get<Report>();

        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, max_iterations);

        if (r.iter < max_iterations) {
            EXPECT_LE(r.solution_error, tolerance);
        }
    }

    template <template <typename> class Solver, typename T>
    void smoke_test()
    {
        const uint32_t N = 5;

        xtensor<T, 2> identity = xt::eye(N);
        xtensor<T, 1> signal   = xt::zeros<T>({N});
        xtensor<T, 1> x        = xt::zeros<T>({N});

        Solver<T> solver(as_span(identity));

        /* for each column in the identity matrix */
        for (uint32_t n = 0; n < N; n++)
        {
            /* construct signal */
            ss::view(signal) = 0.0f;
            signal[n] = 1.0f;

            /* sparse representation */
            ss::view(x) = 0.0f;

            auto result = solver.solve(as_span(signal), .001f, N, as_span(x));
            ::check_report(result, .001f, N);

            /* resulting sparse respresentation should be exactly
            equal to the input signal */
            EXPECT_EQ(x, signal);
        }
    }
}
