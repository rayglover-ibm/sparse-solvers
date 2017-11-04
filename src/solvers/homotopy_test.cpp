#include <ss/ss.h>

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

    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }
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

TEST(homotopy, sparse_signal_test)
{
    const uint32_t N = 10;

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    ss::homotopy solver;

    auto result = solver.solve(
        as_span(identity), as_span(signal), .001f, N, as_span(x));

    ::check_report(result, .001f, N);
}

TEST(homotopy, smoke_test_column_subset)
{
    const int N = 10;
    const int M = 5;

    xt::random::seed(0);

    xtensor<float, 2> data = xt::zeros<float>({ M, N });

    /* columns 0-4 */
    auto noise    = xt::view(data, xt::all(), xt::range(0, M-1));
    /* columns 5-9 */
    auto identity = xt::view(data, xt::all(), xt::range(M, N));

    /* insert noise in the columns were not interested */
    noise = xt::random::rand(noise.shape(), 0.0f, 0.1f);
    /* insert identity in the columns we are interested */
    identity = xt::eye(M);

    xtensor<float, 1> signal = xt::zeros<float>({ M });
    xtensor<float, 1> x      = xt::zeros<float>({ M });

    ss::homotopy solver;

    /* for each column we are interested in */
    for (uint32_t n = 0; n < dim<1>(identity); n++)
    {
        /* construct inputs */
        ss::view(signal) = xt::view(identity, xt::all(), n);
        ss::view(x) = 0.0f;

        auto result = solver.solve(
            as_span(identity), as_span(signal), .001f, N, as_span(x));

        EXPECT_EQ(x, signal);
    }
}

TEST(homotopy, noisy_signal)
{
    const uint32_t N = 50;
    const float NOISE = 0.01f;

    xt::random::seed(0);

    xtensor<float, 2> identity = xt::eye(N);
    xtensor<float, 1> signal   = xt::zeros<float>({N});
    xtensor<float, 1> x        = xt::zeros<float>({N});

    ss::homotopy solver;

    for (uint32_t n = 0; n < N; n++)
    {
        /* construct signal with some noise. */
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

namespace
{
    /* TODO(rayg) replace with xt::norm_l1 */
    template <typename T>
    void l1(ss::ndspan<T, 1> x)
    {
        T sum = xt::sum(xt::abs(x))();
        xt::view(x, xt::all()) /= sum;
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

    ss::homotopy solver;

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
            auto result = solver.solve(as_span(haystack), as_span(s), NOISE, N, as_span(x));

            /* check solver statistics */
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