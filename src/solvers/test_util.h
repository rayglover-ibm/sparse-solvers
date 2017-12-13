#include <ss/ss.h>

#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>

using xt::xtensor;
using ss::as_span;

namespace
{
    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }
    
    template <typename Report>
    void check_report(kernelpp::maybe<Report>& result,
        float tolerance, uint32_t max_iterations
        )
    {
        EXPECT_TRUE(result.template is<Report>());
        auto r = result.template get<Report>();

        EXPECT_GE(r.iter, 1);
        EXPECT_LE(r.iter, max_iterations);

        if (r.iter < max_iterations) {
            EXPECT_LE(r.solution_error, tolerance);
        }
    }

    template <template <typename> class Solver, typename T>
    void smoke_test()
    {
        const uint32_t N = 5;

        xtensor<T, 2> identity = xt::eye(N);
        xtensor<T, 1> signal   = xt::zeros<T>({N});
        xtensor<T, 1> x        = xt::zeros<T>({N});

        Solver<T> solver(as_span(identity));

        /* for each column in the identity matrix */
        for (uint32_t n = 0; n < N; n++)
        {
            /* construct signal */
            ss::view(signal) = 0.0f;
            signal[n] = 1.0f;

            /* sparse representation */
            ss::view(x) = 0.0f;

            auto result = solver.solve(as_span(signal), .001f, N, as_span(x));
            ::check_report(result, .001f, N);

            /* resulting sparse respresentation should be exactly
               equal to the input signal */
            EXPECT_EQ(x, signal);
        }
    }

    template <template <typename> class Solver, typename T>
    void smoke_test_column_subset()
    {
        const int N = 10;
        const int M = 5;

        xt::random::seed(0);

        xtensor<T, 2> data = xt::zeros<T>({ M, N });

        /* columns 0-4 */
        auto noise    = xt::view(data, xt::all(), xt::range(0, M-1));
        /* columns 5-9 */
        auto identity = xt::view(data, xt::all(), xt::range(M, N));

        /* insert noise in the columns were not interested */
        noise = xt::random::rand(noise.shape(), 0.0f, 0.1f);
        /* insert identity in the columns we are interested */
        identity = xt::eye(M);

        xtensor<T, 1> signal = xt::zeros<T>({ M });
        xtensor<T, 1> x      = xt::zeros<T>({ M });

        Solver<T> solver(as_span(identity));

        /* for each column we are interested in */
        for (uint32_t n = 0; n < dim<1>(identity); n++)
        {
            /* construct inputs */
            ss::view(signal) = xt::view(identity, xt::all(), n);
            ss::view(x) = 0.0f;

            auto result = solver.solve(as_span(signal), .001f, N, as_span(x));
            EXPECT_EQ(x, signal);
        }
    }

    template <template <typename> class Solver, typename T>
    void noisy_signal_test()
    {
        const uint32_t N = 50;
        const T NOISE = 0.01f;

        xt::random::seed(0);

        xtensor<T, 2> identity = xt::eye(N);
        xtensor<T, 1> signal   = xt::zeros<T>({N});
        xtensor<T, 1> x        = xt::zeros<T>({N});

        Solver<T> solver(as_span(identity));

        for (uint32_t n = 0; n < N; n++)
        {
            /* construct signal with some noise. */
            signal = xt::random::rand({N}, T{0}, NOISE);
            signal[n] += T{1} - (0.5f * NOISE);

            /* sparse representation */
            ss::view(x) = T{0};

            auto result = solver.solve(as_span(signal), NOISE, N, as_span(x));
            ::check_report(result, NOISE, N);

            /* The solution should be sparse.
               Furthermore, since the noise level in the input signal
               is equal to the solver tolerance, we should expect the
               resulting approximation has sparsity of N-1 / N */
            EXPECT_EQ(xt::filter(x, x > NOISE).size(), 1);
        }
    }

    /* TODO(rayg) replace with xt::norm_l1 */
    template <typename T>
    void l1(ss::ndspan<T, 1> x) {
        xt::view(x, xt::all()) /= xt::sum(xt::abs(x));
    }

    template <template <typename> class Solver, typename T>
    void noisy_patterns_test(uint32_t M, uint32_t N)
    {
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
            EXPECT_LE(xt::nonzero(x).size(), int(N * 0.05));

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
}
