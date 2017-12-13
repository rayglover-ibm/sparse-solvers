#include <ss/ss.h>
#include "test_util.h"

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xeval.hpp>

using xt::xtensor;
using ss::as_span;

namespace
{
    void check_report(kernelpp::maybe<ss::homotopy_report>& result,
        float tolerance, uint32_t max_iterations
        )
    {
        EXPECT_TRUE(result.is<ss::homotopy_report>());
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
    ::noisy_patterns_test<ss::homotopy, float>(100, 25);
    ::noisy_patterns_test<ss::homotopy, float>(25, 100);
}