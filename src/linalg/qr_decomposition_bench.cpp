#include <linalg/qr_decomposition.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>

#include <benchmark/benchmark.h>

using xt::xtensor;
using ss::as_span;

namespace
{
    inline void qr_decomposition_bench(benchmark::State& state)
    {
        xt::random::seed(0);
        const uint32_t M = state.range(0);

        /* make some noise */
        xtensor<float, 2> A = xt::random::randn({ M, M }, .5f, .1f);

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(ss::qr_decomposition<float>(as_span(A)));
        }
    }

    inline void qr_decomposition_solve_bench(benchmark::State& state)
    {
        xt::random::seed(0);
        const uint32_t M = state.range(0);

        /* make some noise */
        xtensor<float, 2> A = xt::random::randn({ M, M }, .5f, .1f);
        xtensor<float, 1> b = xt::random::randn({ M }, .5f, .1f);
        xtensor<float, 1> x = xt::zeros<float>({ M });
        
        ss::qr_decomposition<float> QR(as_span(A));

        while (state.KeepRunning()) {
            QR.solve(b, x);
            ss::view(x) += 1e-4;

            benchmark::ClobberMemory();
        }
    }
}

BENCHMARK(qr_decomposition_bench)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond)
    ->Ranges({ { 32, 8 << 8 } /* M */ });

BENCHMARK(qr_decomposition_solve_bench)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond)
    ->Ranges({ { 32, 8 << 8 } /* M */ });