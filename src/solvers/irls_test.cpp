#include "test_util.h"
#include <ss/ss.h>

#include <gtest/gtest.h>

namespace
{
    template <> void check_report<ss::irls_report>(
        kernelpp::maybe<ss::irls_report>& result,
        float tolerance,
        uint32_t max_iterations)
    {
        ASSERT_TRUE(result.is<ss::irls_report>());
        auto r = result.get<ss::irls_report>();

        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, max_iterations);

        if (r.iter < max_iterations && !r.spd_failure) {
            EXPECT_LE(r.solution_error, tolerance);
        }
    }
}

TEST(irls, smoke_test)
{
    ::smoke_test<ss::irls, float>();
    ::smoke_test<ss::irls, double>();
}

TEST(irls, smoke_test_column_subset)
{
    ::smoke_test_column_subset<ss::irls, float>();
    ::smoke_test_column_subset<ss::irls, double>();
}

TEST(irls, noisy_signal)
{
    ::noisy_signal_test<ss::irls, float>();
    ::noisy_signal_test<ss::irls, double>();
}

TEST(irls, permutations)
{
    ::permutations_test<ss::irls, float>(4, 4, 0.1f, 0.1f, 10);
    ::permutations_test<ss::irls, double>(5, 5, 0.1f, 0.1f, 10);
}