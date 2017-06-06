#include <ss/ss.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <benchmark/benchmark.h>

using xt::xtensor;
using ss::as_span;

namespace
{
    inline void homotopy_bench(benchmark::State& state)
    {
        xt::random::seed(0);

        const uint32_t M = state.range(0);
        const uint32_t N = state.range(1);

        const int PATTERN = 2;
        const float TOL = 0.1f;

        /* make some noise */
        xtensor<float, 2> haystack = xt::random::randn({ M, N }, .5f, .1f);

        /* construct a noisy signal with a pattern */
        xtensor<float, 1> signal = xt::random::randn({ M }, .5f, .1f);
        xt::view(signal, xt::range(0, int(M), PATTERN /* step */)) += 1.0f;

        ss::homotopy solver;
        int iters = 0, i = 0;

        while (state.KeepRunning())
        {
            int n = i % N;

            /* insert a representation of the signal to search for */
            auto needle = xt::view(haystack, xt::range(0, int(M), PATTERN /* step */), n);
            needle += 1.0f;

            xtensor<float, 1> x = xt::zeros<float>({ N });
            auto result = solver.solve(as_span(haystack), as_span(signal), TOL, N, as_span(x));

            iters += result.get_unchecked<ss::homotopy_report>().iter;

            /* remove the signal */
            needle -= 1.0f;

            i++;
        }

        state.counters["Mean iterations"] = double(iters) / i;
    }
}

BENCHMARK(homotopy_bench)
    ->RangeMultiplier(4)
    ->Unit(benchmark::kMillisecond)
    ->Ranges({ { 16, 8 << 6 } /* M */, { 16, 8 << 8 } } /* N */);

BENCHMARK_MAIN();