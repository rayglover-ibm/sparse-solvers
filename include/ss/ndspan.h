/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#pragma once

#include <gsl/gsl>

#include <array>
#include <numeric>

namespace ss
{
    /*
        N-dimensional span of `T`
     */
    template <typename T, size_t N>
    struct ndspan
    {
        explicit ndspan()
            : span{ nullptr, 0 }, shape()
        {
            static_assert(N == 0, "An empty ndspan must be dimensionless");
        }

        ndspan(gsl::span<T> s)
            : span{ s }, shape{ s.size() }
        {
            static_assert(N == 1, "Shape must be specified");
        }

        ndspan(gsl::span<T> s, std::array<ptrdiff_t, N> l)
            : span{ s }, shape{ l }
        {
            Ensures(std::accumulate(shape.begin(), shape.end(), ptrdiff_t{ 1u }, std::multiplies<ptrdiff_t>())
                == span.size());
        }

        gsl::span<T> span;
        const std::array<ptrdiff_t, N> shape;
    };
}