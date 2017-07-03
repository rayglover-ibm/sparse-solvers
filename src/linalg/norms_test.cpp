#include <ss/ndspan.h>
#include <linalg/norms.h>

#include <gtest/gtest.h>

using xt::xtensor;
using ss::mat;

TEST(norms, l1_matrix)
{
    mat<float> A{
        {1, 2, 0},
        {3, 4, 1}
    };

    mat<float> expect{
        {0.25, 0.3333, 0},
        {0.75, 0.6667, 1}
    };

    ss::l1(ss::as_span(A));
    EXPECT_TRUE(xt::isclose(A, expect, 1.0, 0.001)());
}

TEST(norms, l1_vector)
{
    xtensor<float, 1> x{
        {1, 2, 3, 4, 5, 0},
    };

    xtensor<float, 1> expect{
        {0.06667, 0.1333, 0.2, 0.2666, 0.3333, 0}
    };

    ss::l1(ss::as_span(x));
    EXPECT_TRUE(xt::isclose(x, expect, 1.0, 0.001)());
}