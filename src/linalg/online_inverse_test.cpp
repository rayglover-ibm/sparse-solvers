#include <linalg/online_inverse.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmissing.hpp>

using xt::xtensor;
using ss::mat;

TEST(online_inverse, square_permute_2)
{
    const ss::mat<float> A{
        {1, 2},
        {3, 4}
    };

    { /* shift 0 -> 1 */
        ss::mat<float> expect {
            {4, 3},
            {2, 1}
        };
        ss::mat<float> A_tmp = A;

        ss::detail::square_permute(ss::as_span(A_tmp), 0, 1);
        EXPECT_EQ(A_tmp, expect);

        ss::detail::square_permute(ss::as_span(A_tmp), 1, 0);
        EXPECT_EQ(A_tmp, A);
    }
}

TEST(online_inverse, square_permute_3)
{
    const ss::mat<float> A{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    { /* shift 1 -> 2 */
        ss::mat<float> expect {
            {1, 3, 2},
            {7, 9, 8},
            {4, 6, 5}
        };
        ss::mat<float> A_tmp = A;

        ss::detail::square_permute(ss::as_span(A_tmp), 1, 2);
        EXPECT_EQ(A_tmp, expect);

        ss::detail::square_permute(ss::as_span(A_tmp), 2, 1);
        EXPECT_EQ(A_tmp, A);
    }

    { /* shift 0 -> 2 */
        ss::mat<float> expect {
            {5, 6, 4},
            {8, 9, 7},
            {2, 3, 1}
        };
        ss::mat<float> A_tmp = A;

        ss::detail::square_permute(ss::as_span(A_tmp), 0, 2);
        EXPECT_EQ(A_tmp, expect);

        ss::detail::square_permute(ss::as_span(A_tmp), 2, 0);
        EXPECT_EQ(A_tmp, A);
    }
}

TEST(online_inverse, square_permute_4)
{
    const ss::mat<float> A{
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9,  10, 11, 12},
        {13, 14, 15, 16}
    };

    { /* shift 1 -> 3 */
        const ss::mat<float> expect{
            {1,  3,  4,  2},
            {9,  11, 12, 10},
            {13, 15, 16, 14},
            {5,  7,  8,  6},
        };
        ss::mat<float> A_tmp = A;

        ss::detail::square_permute(ss::as_span(A_tmp), 1, 3);
        EXPECT_EQ(A_tmp, expect);

        ss::detail::square_permute(ss::as_span(A_tmp), 3, 1);
        EXPECT_EQ(A_tmp, A);
    }

    { /* shift 1 -> 2 */
        const ss::mat<float> expect{
            {1,  3,  2,  4},
            {9,  11, 10, 12},
            {5,  7,  6,  8},
            {13, 15, 14, 16},
        };
        ss::mat<float> A_tmp = A;

        ss::detail::square_permute(ss::as_span(A_tmp), 1, 2);
        EXPECT_EQ(A_tmp, expect);

        ss::detail::square_permute(ss::as_span(A_tmp), 2, 1);
        EXPECT_EQ(A_tmp, A);
    }
}