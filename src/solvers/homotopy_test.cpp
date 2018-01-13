#include <ss/ss.h>
#include "test_util.h"

#include <gtest/gtest.h>

namespace
{
    template <> void check_report<ss::homotopy_report>(
        kernelpp::maybe<ss::homotopy_report>& result,
        float tolerance, uint32_t max_iterations)
    {
        ASSERT_TRUE(result.is<ss::homotopy_report>());
        auto r = result.get<ss::homotopy_report>();

        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, max_iterations);

        if (r.iter < max_iterations) {
            EXPECT_LE(r.solution_error, tolerance);
        }
    }
}

TEST(homotopy, smoke_test)
{
    ::smoke_test<ss::homotopy, float>();
    ::smoke_test<ss::homotopy, double>();
}

TEST(homotopy, smoke_test_column_subset)
{
    ::smoke_test_column_subset<ss::homotopy, float>();
    ::smoke_test_column_subset<ss::homotopy, double>();
}

TEST(homotopy, noisy_signal)
{
    ::noisy_signal_test<ss::homotopy, float>();
    ::noisy_signal_test<ss::homotopy, double>();
}

TEST(homotopy, noisy_patterns_test)
{
    ::noisy_patterns_test<ss::homotopy, float>(100, 25, .1f, 1.0f);
    ::noisy_patterns_test<ss::homotopy, float>(25, 100, .1f, 1.0f);
}

TEST(homotopy, permutations)
{
    /* square */
    ::permutations_test<ss::homotopy, float>(10, 10, .1f, .1f, 10);
    ::permutations_test<ss::homotopy, double>(10, 10, .1f, .1f, 10);
    
    /* overdetermined */
    ::permutations_test<ss::homotopy, float>(25, 10, .1f, .1f, 25);
    ::permutations_test<ss::homotopy, double>(25, 10, .1f, .1f, 25);

    /* underdetermined */
    ::permutations_test<ss::homotopy, float>(10, 25, .05f, .05f, 25);
    ::permutations_test<ss::homotopy, double>(10, 25, .05f, .05f, 25);
}