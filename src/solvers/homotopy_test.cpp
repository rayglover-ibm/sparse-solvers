#include "ss/ss.h"

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

using xt::xtensor;
using ss::as_span;

TEST(homotopy, smoke_test)
{
    const uint32_t N = 5;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal */
        view(signal) = 0.0f;
        signal[n] = 1.0f;

        /* sparse representation */
        view(x) = 0.0f;

        ss::homotopy h;
        auto result = h.solve(as_span(identity), as_span(signal), 0.001, 10, as_span(x));

        /* resulting sparse respresentation should be exactly
           equal to the input signal */
        EXPECT_TRUE(result.is<ss::homotopy_report>());
        EXPECT_EQ(x, signal);
    }
}

TEST(homotopy, noisy_signal)
{
    const uint32_t N = 50;
    const double NOISE = 0.01;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal with some noise.
           TODO(rayg): normalize? */
        signal = xt::random::rand({N}, 0.0, NOISE);
        signal[n] += 1.0 - (0.5 * NOISE);

        /* sparse representation */
        view(x) = 0.0f;

        ss::homotopy h;
        auto result = h.solve(as_span(identity), as_span(signal), NOISE, 100, as_span(x));

        ss::homotopy_report r = result.get<ss::homotopy_report>();
        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, 100);
        EXPECT_LE(r.solution_error, NOISE);

        /* The solution should be sparse.
           Furthermore, since the noise level in the input signal
           is equal to the solver tolerance, we should expect the
           resulting approximation has sparsity of N-1 / N */
        auto x2 = xt::xarray<float>(x);
        EXPECT_EQ(xt::nonzero(x2).size(), 1);
    }
}