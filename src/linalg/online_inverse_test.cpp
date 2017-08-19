#include <linalg/online_inverse.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

using xt::xtensor;
using ss::mat;
using kernelpp::run;

TEST(online_inverse, square_permute_2)
{
    using op = ::ss::detail::square_permute;

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

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 0, 1));
        EXPECT_EQ(A_tmp, expect);

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 1, 0));
        EXPECT_EQ(A_tmp, A);
    }
}

TEST(online_inverse, square_permute_3)
{
    using op = ::ss::detail::square_permute;

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

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 1, 2));
        EXPECT_EQ(A_tmp, expect);

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 2, 1));
        EXPECT_EQ(A_tmp, A);
    }

    { /* shift 0 -> 2 */
        ss::mat<float> expect {
            {5, 6, 4},
            {8, 9, 7},
            {2, 3, 1}
        };
        ss::mat<float> A_tmp = A;

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 0, 2));
        EXPECT_EQ(A_tmp, expect);

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 2, 0));
        EXPECT_EQ(A_tmp, A);
    }
}

TEST(online_inverse, square_permute_4)
{
    using op = ::ss::detail::square_permute;

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

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 1, 3));
        EXPECT_EQ(A_tmp, expect);

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 3, 1));
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

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 1, 2));
        EXPECT_EQ(A_tmp, expect);

        EXPECT_FALSE(run<op>(ss::as_span(A_tmp), 2, 1));
        EXPECT_EQ(A_tmp, A);
    }
}

TEST(online_inverse, erase_last_rowcol)
{
    using op = ::ss::detail::erase_last_rowcol;
    {
        ss::aligned_vector<float> A{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        EXPECT_FALSE(run<op>(A, 3, 3));
        EXPECT_EQ(A, (ss::aligned_vector<float>{ 1, 2, 4, 5 }));

        EXPECT_FALSE(run<op>(A, 2, 2));
        EXPECT_EQ(A, (ss::aligned_vector<float>{ 1 }));
    }
    {
        ss::aligned_vector<float> A{
            1, 2, 3,
            4, 5, 6
        };

        EXPECT_FALSE(run<op>(A, 2, 3));
        EXPECT_EQ(A, (ss::aligned_vector<float>{ 1, 2 }));
    }
}

TEST(online_inverse, insert_last_rowcol)
{
    using op = ::ss::detail::insert_last_rowcol;
    {
        ss::aligned_vector<float> A{ 1 };

        EXPECT_FALSE(run<op>(A, 1, 1, 0.0f));
        EXPECT_EQ(A, (ss::aligned_vector<float>{ 1, 0, 0, 0 }));
    }
    {
        ss::aligned_vector<float> A{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        EXPECT_FALSE(run<op>(A, 3, 3, 0.0f));
        EXPECT_EQ(A, (ss::aligned_vector<float>{
            1, 2, 3, 0,
            4, 5, 6, 0,
            7, 8, 9, 0,
            0, 0, 0, 0
        }));
    }
    {
        ss::aligned_vector<float> A{
            1, 2, 3
        };

        EXPECT_FALSE(run<op>(A, 1, 3, 0.0f));
        EXPECT_EQ(A, (ss::aligned_vector<float>{
            1, 2, 3, 0,
            0, 0, 0, 0
        }));
    }
}

TEST(online_inverse, identity)
{
    const uint64_t K = 10;

    xt::xtensor<float, 2> A = xt::eye(K);
    ss::online_column_inverse<float> inv(K);

    uint64_t k=0;

    /* insert */
    for (; k < K; k++)
    {
        auto col = xt::view(A, xt::all(), k);
        inv.insert(k, col.cbegin(), col.cend());

        xt::xarray<float> b = xt::eye({k + 1u, k + 1u});
        EXPECT_TRUE(xt::isclose(inv.inverse(), b, 1.0, .0)());
    }

    k = K-1;

    /* remove */
    for (; int(k) > 0; k--)
    {
        inv.remove(k);

        xt::xarray<float> b = xt::eye({k, k});
        EXPECT_TRUE(xt::isclose(inv.inverse(), b, 1.0, .0)());
    }

    inv.remove(0);
    EXPECT_EQ(0, inv.N());
}