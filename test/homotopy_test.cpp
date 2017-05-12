#define GSL_THROW_ON_CONTRACT_VIOLATION 1

#include "ss/ss.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

TEST(homotopy, smoke_test)
{
    const uint32_t N = 5;

    std::array<float, N * N> identity_mat;
    std::array<float, N> signal;
    std::array<float, N> x;

    /* identity matrix */
    identity_mat.fill(0);
    for (uint32_t i = 0, j = 0; i < identity_mat.size(); i += N, j++)
        identity_mat[i + j] = 1.0;

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal */
        signal.fill(0);
        signal[n] = 1;

        /* sparse representation */
        x.fill(0);

        ss::homotopy h;
        auto result = h.solve({ identity_mat, { N, N } }, { signal }, 0.001, 10, x);

        /* resulting sparse respresentation should be exactly
           equal to the input signal */
        EXPECT_TRUE(result.is<ss::homotopy_report>());
        EXPECT_EQ(x, signal);
    }
}