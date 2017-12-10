#include "test_util.h"
#include <ss/ss.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

TEST(irls, smoke_test)
{
    ::smoke_test<ss::irls, float>();
    ::smoke_test<ss::irls, double>();
}