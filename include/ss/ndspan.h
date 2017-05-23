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
#include <cstddef>

namespace ss
{
    namespace detail
    {
        template <size_t NDims>
        size_t total_span(std::array<size_t, NDims> dims) {
            return std::accumulate(dims.begin(), dims.end(), size_t{ 1u }, std::multiplies<size_t>());
        }
    }

    /*
        N-dimensional span of `T`
        TODO(rayg): replace with xtensor view
     */
    template <typename T, size_t NDims = 1>
    struct ndspan
    {
        explicit ndspan()
            : span{ nullptr, 0 }, shape()
        {
            static_assert(NDims == 0, "An empty ndspan must be dimensionless");
        }

        ndspan(gsl::span<T> s)
            : span{ s }, shape{ s.size() }
        {
            static_assert(NDims == 1, "Shape must be specified");
        }

        ndspan(gsl::span<T> s, std::array<size_t, NDims> l)
            : span{ s }, shape{ l }
        {
            Ensures(detail::total_span<NDims>(l) == span.size());
        }

        ndspan(T* data, std::array<size_t, NDims> l)
            : span{ data, detail::total_span<NDims>(l) }, shape{ l }
        {}

		T* data() { return span.data(); }
		const T* data() const { return span.data(); }

        gsl::span<T> span;
        const std::array<size_t, NDims> shape;
    };
}