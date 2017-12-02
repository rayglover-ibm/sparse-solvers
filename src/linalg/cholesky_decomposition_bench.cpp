#include <linalg/cholesky_decomposition.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>

#include <benchmark/benchmark.h>

using xt::xtensor;
using ss::as_span;

namespace
{
    inline void cholesky_decomposition_bench(benchmark::State& state)
    {
        xt::random::seed(0);
        const uint32_t M = state.range(0);

        /* make some spd noise */
        xtensor<float, 2> noise = xt::random::randn({ M, M }, 10.0f, 5.0f);
        auto A = ss::blas::xgemm(CblasNoTrans, CblasTrans, float{1}, noise, noise);

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(ss::cholesky_decomposition<float>(as_span(A)));
        }
    }
}

BENCHMARK(cholesky_decomposition_bench)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond)
    ->Ranges({ { 32, 8 << 8 } /* M */ });
