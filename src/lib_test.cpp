#define GSL_THROW_ON_CONTRACT_VIOLATION 1

#include "ss/ss.h"

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

    ss::ndspan<float> spanB({ &data[4], 1}, { 1 });

    EXPECT_EQ(1, spanB.shape()[0]);
    EXPECT_EQ(spanB(0), 5);
}