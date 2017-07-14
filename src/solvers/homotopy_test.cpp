#include <ss/ss.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

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
    const uint32_t N = 5;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    ss::homotopy solver;

    /* for each column in the identity matrix */
    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal */
        ss::view(signal) = 0.0f;
        signal[n] = 1.0f;

        /* sparse representation */
        ss::view(x) = 0.0f;

        auto result = solver.solve(
            as_span(identity), as_span(signal), .001f, N, as_span(x));

        ::check_report(result, .001f, N);

        /* resulting sparse respresentation should be exactly
           equal to the input signal */
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
        ss::view(x) = 0.0f;

        auto result = solver.solve(
            as_span(identity), as_span(signal), NOISE, N, as_span(x));

        ::check_report(result, NOISE, N);

        /* The solution should be sparse.
           Furthermore, since the noise level in the input signal
           is equal to the solver tolerance, we should expect the
           resulting approximation has sparsity of N-1 / N */
        {
            auto x2 = xt::xarray<float>(x);
            EXPECT_EQ(xt::nonzero(x2).size(), 1);
        }
    }
}

TEST(homotopy, noisy_patterns)
{
    const uint32_t M = 25, N = 100;
    const int PATTERN = 3;
    const float TOL = 0.115f;

    /* make some noise */
    xtensor<float, 2> haystack = xt::random::randn({ M, N }, .5f, .1f);

    /* construct a noisy signal with a pattern */
    xtensor<float, 1> signal = xt::random::randn({ M }, .5f, .1f);
    xt::view(signal, xt::range(0, int(M), PATTERN /* step */)) += 1.0f;

    ss::homotopy solver;

    int failures = 0;
    for (uint32_t n = 0; n < N; n++)
    {
        /* insert a representation of the signal to search for */
        auto needle = xt::view(haystack, xt::range(0, int(M), PATTERN /* step */), n);
        needle += 1.0f;

        /* solve */
        xtensor<float, 1> x = xt::zeros<float>({ N });
        auto result = solver.solve(as_span(haystack), as_span(signal), TOL, 25, as_span(x));

        /* check solver statistics */
        ::check_report(result, TOL, 25);

        /* argmax(x) == n */
        EXPECT_EQ(xt::amax(x)(), x[n]);
        {
            /* evaluate the sparse representation. */
            xtensor<float, 1> expect = xt::zeros<float>({ N });
            expect[n] = 1.0;

            if (!xt::isclose(x, expect, 1.0 /* relative */, TOL /* absolute */)()) {
                std::cout << "Solution for signal " << n << " failed a sparisty test:\n" << x << "\n\n";
                failures++;
            }
        }
        {
            /* reconstruct the signal given the sparse representation */
            xt::xtensor<float, 1> y = xt::zeros<float>({ M });
            ss::reconstruct_signal(as_span(haystack), as_span(x), as_span(y));

            if (!xt::isclose(x, y, 1.0 /* relative */, TOL /* absolute */)()) {
                std::cout << "Reconstruction of signal " << n << " failed.\n";
                failures++;
            }
        }

        /* remove the signal */
        needle -= 1.0f;
    }
    EXPECT_EQ(failures, 0);
}