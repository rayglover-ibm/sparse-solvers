#include <linalg/cholesky_decomposition.h>

#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace ss::blas;
using xt::xtensor;
using ss::mat;
using ss::dim;

TEST(cholesky_decomposition, isspd)
{
    const xtensor<float, 2> A{
        {0, 1},
        {1, 0}
    };

    ss::cholesky_decomposition<float> chol{ ss::as_span(A) };
    EXPECT_FALSE(chol.isspd());
}

TEST(cholesky_decomposition, 2x2)
{
    const xtensor<float, 2> A{
        {2, 1},
        {1, 2}
    };

    const xtensor<float, 1> b{1, -1};
    const xtensor<float, 1> x_expect{1, -1};

    ss::cholesky_decomposition<float> chol{ ss::as_span(A) };
    EXPECT_TRUE(chol.isspd());
    {
        SCOPED_TRACE("A = LL*");

        auto LLT = xgemm(CblasNoTrans, CblasTrans, float(1.0), chol.l(), chol.l());

        EXPECT_TRUE(xt::allclose(A, LLT, 0.0f, 1e-4));
    }
    {
        SCOPED_TRACE("Ax = b");

        const xtensor<float, 1> x{0, 0};
        chol.solve(b, x);

        EXPECT_TRUE(xt::allclose(x, x_expect, 0.0f, 1e-4));
    }
}

namespace
{
    template <typename T>
    void test_decomposition(ss::ndspan<T, 2> A, T absolute_error)
    {
        using namespace ss::blas;

        ss::cholesky_decomposition<T> chol{ A };
        EXPECT_TRUE(chol.isspd());
        {
            SCOPED_TRACE("A = LL*");

            auto LLT = xgemm(CblasNoTrans, CblasTrans, T{1.0}, chol.l(), chol.l());
            EXPECT_TRUE(xt::allclose(A, LLT, T{0}, absolute_error));
        }
    }

    template <typename T>
    void test_random(int N)
    {
        xtensor<T, 2> noise = xt::random::randn({ N, N }, 10.0f, 5.0f);
        auto spd_noise = ss::blas::xgemm(CblasNoTrans, CblasTrans, T{1}, noise, noise);

        ::test_decomposition(ss::as_span(spd_noise), T(1e-3f));
    }
}

TEST(cholesky_decomposition, random_inputs)
{
    xt::random::seed(0);

    test_random<float>(1);
    test_random<float>(2);
    test_random<float>(4);
    test_random<float>(5);
    test_random<float>(6);
    test_random<float>(7);

    test_random<double>(50);
    test_random<double>(100);
}