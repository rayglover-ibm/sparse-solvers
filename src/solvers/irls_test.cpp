#include "test_util.h"
#include <ss/ss.h>

#include <gtest/gtest.h>

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