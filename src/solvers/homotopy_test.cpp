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

namespace
{
    /* TODO(rayg) replace with xt::norm_l1 */
    template <typename T>
    void l1(ss::ndspan<T, 1> x) {
        xt::view(x, xt::all()) /= xt::sum(xt::abs(x));
    }
}

TEST(homotopy, noisy_patterns)
{
    const uint32_t M = 25, N = 100;
    const int PATTERN = 3;
    const float NOISE = 0.01f;

    xt::random::seed(0);

    /* make some noise */
    xtensor<float, 2> noise = xt::random::randn({ M, N }, .5f, .1f);

    /* construct a noisy signal with a pattern */
    xtensor<float, 1> signal = xt::random::randn({ M }, .5f, .1f);
    xt::view(signal, xt::range(0, int(M), PATTERN /* step */)) += 1.0f;

    ss::ndspan<float, 1> s = as_span(signal);
    ::l1(s);

    int failures = 0;
    for (uint32_t n = 0; n < N; n++)
    {
        xt::xtensor<float, 2> haystack = noise;

        /* insert a representation of the signal to search for in the current column */
        auto needle = xt::view(haystack, xt::range(0, int(M), PATTERN /* step */), n);
        needle += 1.0f;

        /* normalize the columns such that the sum of each column equals 1 */
        ss::norm_l1(as_span(haystack));

        /* solve */
        xtensor<float, 1> x = xt::zeros<float>({ N });
        {
            auto result = ss::homotopy<float>(as_span(haystack))
                .solve(as_span(s), NOISE, N, as_span(x));

            ::check_report(result, NOISE, N);
        }

        /* argmax(x) == n */
        EXPECT_EQ(xt::amax(x)(), x[n]);
        /* 95% sparsity */
        EXPECT_LT(xt::nonzero(x).size(), int(N * 0.05));

        {
            /* reconstruct the signal given the sparse representation */
            xtensor<float, 1> y = xt::zeros<float>({ M });
            ss::reconstruct_signal(as_span(haystack), as_span(x), as_span(y));

            if (!xt::allclose(y, s, 0.0 /* relative */, 5 * NOISE /* absolute */)) {
                std::cout << "Reconstruction of signal " << n << " failed.\n";
                failures++;
            }
        }
    }
    EXPECT_EQ(failures, 0);
}