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
#include <xtensor/xtensor.hpp>
#include <xtensor/xbuffer_adaptor.hpp>
#include <array>

namespace ss
{
    template <typename T, size_t NDim = 1>
    using ndspan = xt::xtensor_adaptor<
        xt::xbuffer_adaptor<T>, NDim, xt::layout_type::row_major
        >;

    /* helpers ------------------------------------------------------------- */

    template <size_t N, typename C, typename T = typename C::value_type>
    ndspan<T, N> as_span(C& container, std::array<size_t, N> shape) {
        return { { container.data(), container.size() }, shape };
    }

    template <typename C, typename T = typename C::value_type>
    ndspan<T, 1> as_span(C& container) {
        return { { container.data(), container.size() }, { container.size() } };
    }
}