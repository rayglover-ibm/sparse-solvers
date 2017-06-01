#include <ss/ss.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

using xt::xtensor;
using ss::as_span;

TEST(homotopy, smoke_test)
{
    const uint32_t N = 5;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    ss::homotopy solver;

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal */
        view(signal) = 0.0f;
        signal[n] = 1.0f;

        /* sparse representation */
        view(x) = 0.0f;

        auto result = solver.solve(
            as_span(identity), as_span(signal), 0.001f, 10, as_span(x));

        /* resulting sparse respresentation should be exactly
           equal to the input signal */
        EXPECT_TRUE(result.is<ss::homotopy_report>());
        EXPECT_EQ(x, signal);
    }
}

TEST(homotopy, noisy_signal)
{
    const uint32_t N = 50;
    const float NOISE = 0.01f;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    ss::homotopy solver;

    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal with some noise.
           TODO(rayg): normalize? */
        signal = xt::random::rand({N}, 0.0f, NOISE);
        signal[n] += 1.0f - (0.5f * NOISE);

        /* sparse representation */
        view(x) = 0.0f;

        auto result = solver.solve(
            as_span(identity), as_span(signal), NOISE, 100, as_span(x));

        auto r = result.get<ss::homotopy_report>();
        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, 100);
        EXPECT_LE(r.solution_error, NOISE);

        /* The solution should be sparse.
           Furthermore, since the noise level in the input signal
           is equal to the solver tolerance, we should expect the
           resulting approximation has sparsity of N-1 / N */
        auto x2 = xt::xarray<float>(x);
        EXPECT_EQ(xt::nonzero(x2).size(), 1);
    }
}

TEST(homotopy, noisy_patterns)
{
    const uint32_t M = 25, N = 100;
    const float NOISE = 0.05f;

    /* make some noise */
    xtensor<float, 2> haystack = xt::random::randn({ M, N }, NOISE * 2 /* mean */, NOISE /* stddev */);
    haystack = xt::clip(haystack, 0.0, 1.0);

    /* construct a noisy signal with a pattern */
    xtensor<float, 1> signal = xt::random::randn({ M }, NOISE * 2, NOISE);
    xt::view(signal, xt::range(0, int(M), 2 /* step */)) += 1.0f;

    ss::homotopy solver;

    for (uint32_t n = 0; n < N; n++)
    {
        /* insert a representation of the signal to search for */
        auto needle = xt::view(haystack, xt::range(0, int(M), 2 /* step */), n);
        needle += 1.0f;

        xtensor<float, 1> x = xt::zeros<float>({N});
        auto result = solver.solve(as_span(haystack), as_span(signal), 0.01f, 100, as_span(x));

        auto r = result.get<ss::homotopy_report>();
        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, 100);
        EXPECT_LE(r.solution_error, NOISE);

        /* argmax(x) == n */
        EXPECT_EQ(xt::amax(x)(), x[n]);

        /* remove the signal */
        needle -= 1.0f;
    }
}