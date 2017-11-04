#include <ss/ndspan.h>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xnoalias.hpp>

#include <gtest/gtest.h>
#include <array>
#include <vector>

TEST(ndspan, 2d_constructors)
{
    std::array<float, 8> data;
    ss::ndspan<float, 2> mat = ss::as_span<2>(data, { 4, 2 });
    ss::ndspan<float, 2> mat2 = ss::as_span<2>(data, { 8, 1 });

    /* 4 * 1 != 8 */
    EXPECT_ANY_THROW(( ss::as_span<2>(data, { 4, 1 }) ));
    /* n > 0 */
    EXPECT_ANY_THROW(( ss::as_span<2>(data, { 8, 0 }) ));

    EXPECT_EQ(&data.at(0), mat.raw_data());
    EXPECT_EQ(&data.at(0), mat2.raw_data());
}

TEST(ndspan, 1d_constructors)
{
    std::array<float, 8> data{ 1, 2, 3, 4, 5, 6, 7, 8 };
    ss::ndspan<float> spanA = ss::as_span(data);

    EXPECT_EQ(data.size(), spanA.shape()[0]);
    EXPECT_EQ(data[1], spanA(1));

    ss::ndspan<float> spanB = ss::as_span(&data[4], 1);

    EXPECT_EQ(1, spanB.shape()[0]);
    EXPECT_EQ(spanB(0), 5);
}

TEST(ndspan, constness)
{
    std::array<float, 8> data{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ss::ndspan<float> spanA = ss::as_span(data);
    const ss::ndspan<float> spanB = ss::as_span(spanA);

    EXPECT_EQ(spanB.size(), spanA.size());
    EXPECT_EQ(spanB, spanA);
}

TEST(ndspan, strides)
{
    std::array<float, 9> data{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    {
        /* full strides */
        auto span = ss::as_span<2>(data.data(), { 3, 3 }, { 3, 1 });
        EXPECT_EQ(9, span.size());
    }
    {
        /* row spans */
        auto span = ss::as_span<2>(data.data(), { 3, 2 }, { 3, 2 });
        xt::xtensor<float, 2> expect {{1, 3}, {4, 6}, {7, 9}};

        EXPECT_EQ(6, span.size());
        EXPECT_EQ(expect, span);
    }
    {
        /* row and column spans */
        auto span = ss::as_span<2>(data.data(), { 2, 2 }, { 6, 2 });
        xt::xtensor<float, 2> expect {{1, 3}, {7, 9}};

        EXPECT_EQ(4, span.size());
        EXPECT_EQ(expect, span);
    }
}

TEST(ndspan, xtensor_compatibility)
{
    xt::xarray<float> x{ 1, 2, 3, 4 };

    auto d = xt::diag(x);
    auto&& e = xt::eval(d);

    ss::ndspan<float, 2> spanA { ss::as_span(e) };

    EXPECT_EQ(1, spanA(0, 0));
    EXPECT_EQ(4, spanA(3, 3));

    xt::noalias(spanA) = xt::ones<float>({ 4u, 4u });
    EXPECT_EQ(1, spanA(3, 3));
}