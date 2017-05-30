#define GSL_THROW_ON_CONTRACT_VIOLATION 1

#include "ss/ss.h"

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <array>
#include <vector>

TEST(homotopy, smoke_test)
{
    const uint32_t N = 5;

    xt::xtensor<float, 2> identity_mat = xt::eye(N);
    std::array<float, N> signal;
    std::array<float, N> x;

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal */
        signal.fill(0);
        signal[n] = 1;

        /* sparse representation */
        x.fill(0);

        ss::homotopy h;
        auto result = h.solve(ss::as_span(identity_mat), ss::as_span(signal), 0.001, 10, ss::as_span(x));

        /* resulting sparse respresentation should be exactly
           equal to the input signal */
        EXPECT_TRUE(result.is<ss::homotopy_report>());
        EXPECT_EQ(x, signal);
    }
}