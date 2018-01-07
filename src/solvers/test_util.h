#include <ss/ss.h>

#include <gtest/gtest.h>
#include <algorithm>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xindex_view.hpp>

using xt::xtensor;
using ss::as_span;

namespace
{
    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }

    template <typename Report>
    void check_report(
        kernelpp::maybe<Report>& result,
        float tolerance,
        uint32_t max_iterations);

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

    template <typename T>
    void l1(ss::ndspan<T, 1> x)
    {
        T norm = xt::sum(xt::abs(x))();
        ASSERT_GT(norm, 0.0);
        xt::view(x, xt::all()) /= norm;
    }

    template <template <typename> class Solver, typename T>
    void noisy_patterns_test(
        uint32_t M,
        uint32_t N,
        T noise_level,
        T signal_level)
    {
        const int PATTERN = 2;
        const T   ERROR = 0.1 * noise_level;

        xt::random::seed(0);

        /* make some noise */
        xtensor<T, 2> noise = xt::random::randn({ M, N }, .5f, noise_level);

        /* construct a noisy signal with a pattern */
        xtensor<T, 1> signal = xt::random::randn({ M }, .5f, noise_level);
        xt::view(signal, xt::range(0, int(M), PATTERN /* step */)) += signal_level;

        ss::ndspan<T, 1> s = as_span(signal);
        ::l1(s);

        for (uint32_t n = 0; n < N; n++)
        {
            xt::xtensor<T, 2> haystack = noise;

            /* insert a representation of the signal to search for in the current column */
            auto needle = xt::view(haystack, xt::range(0, int(M), PATTERN /* step */), n);
            needle = signal_level;

            /* normalize the columns such that the sum of each column equals 1 */
            ss::norm_l1(as_span(haystack));

            /* solve */
            xtensor<T, 1> x = xt::zeros<T>({ N });
            {
                auto result = Solver<T>(as_span(haystack))
                    .solve(s, ERROR, N, as_span(x));

                ::check_report(result, ERROR, N);
            }
            
            /* argmax(x) == n */
            EXPECT_EQ(xt::argmax(x)(), n);

            /* expect a single element above the ERROR level */
            EXPECT_EQ(xt::filter(x, x > ERROR).size(), 1);

            {
                /* reconstruct the signal given the sparse representation */
                xtensor<T, 1> y = xt::zeros<T>({ M });
                ss::reconstruct_signal(as_span(haystack), as_span(x), as_span(y));

                if (!xt::allclose(y, s, 0.0 /* relative */, 5 * ERROR /* absolute */)) {
                    ADD_FAILURE() << "Reconstruction of signal " << n << " failed:"
                        << "\n  y=" << y
                        << "\n  s=" << s
                        << '\n';
                }
            }
        }
    }

    template <typename T>
    void permute(std::vector<T>& v, int n) {
        while (n > 0) { std::next_permutation(v.begin(), v.end()); n--; }
    }

    template <template <typename> class Solver, typename T>
    void permutations_test(
        uint32_t M,
        uint32_t N,
        T signal_noise = 0.0,
        T sensing_noise = 0.0,
        int skip = 1)
    {
        xt::print_options::set_precision(4);
        xt::print_options::set_line_width(300);
        xt::random::seed(0);

        /* error level for the solver */
        T ERROR = signal_noise + sensing_noise;

        /* initialize the first permutation */
        std::vector<T> colbuff(M, T{0});
        std::iota(colbuff.begin(), colbuff.end(), 1);
        ::permute(colbuff, skip);

        /* sensing matrix of permutations */
        xtensor<T, 2> A = xt::random::randn({ M, N }, T{ 0 }, sensing_noise);
        {
            /* given m numbers, produce n permutations, one
               in each column of the sensing matrix */
            std::vector<T> col = colbuff;
            for (int n{0}; n < N; n++, ::permute(col, skip)) {
                xt::view(A, xt::all(), n) += as_span(col);
            }
        }

        Solver<T> solver(as_span(A));
        
        /* find each permutation */
        for (int n{0}; n < N; n++, ::permute(colbuff, skip))
        {
            xtensor<T, 1> signal = as_span(colbuff) + xt::random::randn({ M }, T{ 0 }, signal_noise);
            xtensor<T, 1> x = xt::zeros<T>({ N });

            auto result = solver.solve(as_span(signal), ERROR, N, as_span(x));
            ::check_report(result, ERROR, N);

            /* argmax(x) == n */
            if (xt::argmax(x)() != n) {
                xtensor<T, 1> b = xt::zeros<T>({ M });
                ss::reconstruct_signal(as_span(A), as_span(x), as_span(b));

                ADD_FAILURE() << "Solution for signal " << n << " failed:"
                    << "\n     x =" << x
                    << "\n  Ax-b =" << (b-signal)
                    << '\n';
            }
        }
    }
}
