#include <linalg/qr_decomposition.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

using xt::xtensor;
using ss::mat;
using ss::dim;

TEST(qr_decomposition, 2x2)
{
    const xtensor<float, 2> A{
        { 1, -1},
        {-1,  1}
    };

    const xtensor<float, 1> b{1, -1};
    const xtensor<float, 1> x_expect{0, -1};

    {
        SCOPED_TRACE("Ax = b");

        ss::qr_decomposition<float> QR{ ss::as_span(A) };

        xtensor<float, 1> x{ 0, 0 };
        QR.solve(b, x);

        EXPECT_TRUE(xt::allclose(x, x_expect, 0.0f, 1e-4));
    }
}

namespace
{
    template <typename T>
    void test_decomposition(ss::ndspan<T, 2> A, T absolute_error)
    {
        using namespace ss::blas;

        ss::qr_decomposition<T> QR{ A };

        auto q = QR.q();
        auto r = QR.r();

        EXPECT_EQ(q.shape(), A.shape());
        EXPECT_THAT(r.shape(), testing::ElementsAre(dim<1>(A), dim<1>(A)));

        {
            SCOPED_TRACE("QR = A");

            auto qr = xgemm(CblasNoTrans, CblasNoTrans, T{1}, q, r);
            EXPECT_TRUE(xt::allclose(A, qr, T{0} /* rel */, absolute_error));
        }
        {
            SCOPED_TRACE("transpose(Q)Q = I");

            auto qTq = xgemm(CblasTrans, CblasNoTrans, T{1}, q, q);
            EXPECT_TRUE(xt::allclose(qTq, xt::eye(dim<0>(qTq)), T{0} /* rel */, absolute_error));
        }

    }

    template <typename T>
    void test_random(int M, int N)
    {
        xtensor<T, 2> noise = xt::random::randn({ M, N }, 10.0f, 2.5f);
        ::test_decomposition(ss::as_span(noise), T(1e-4f));
    }
}

TEST(qr_decomposition, random_inputs)
{
    test_random<float>(1, 1);
    test_random<float>(2, 2);
    test_random<float>(4, 4);
    test_random<float>(5, 4);
    test_random<float>(6, 4);
    test_random<float>(7, 4);

    test_random<double>(50, 50);
    test_random<double>(100, 20);
}