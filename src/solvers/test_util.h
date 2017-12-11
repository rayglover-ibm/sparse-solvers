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
}
