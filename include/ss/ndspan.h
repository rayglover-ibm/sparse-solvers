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

    /* stl container as 1-d non-owning view */
    template <typename C>
    inline auto as_span(C& container)
        -> std::enable_if_t<!xt::has_raw_data_interface<C>::value,
                ndspan<typename C::value_type, 1>
                >
    {
        xt::xbuffer_adaptor<typename C::value_type> buff{ container.data(), container.size() };
        return { buff, { container.size() } };
    }

    /* stl container as n-d non-owning view */
    template <size_t N, typename C, typename T = typename C::value_type>
    ndspan<T, N> as_span(C& container, std::array<size_t, N> shape)
    {
        xt::xbuffer_adaptor<T> buff{ container.data(), container.size() };
        return { buff, shape };
    }

    /* tensor-like container as n non-owning view */
    template <class T>
    inline auto as_span(T& t)
        -> std::enable_if_t<xt::has_raw_data_interface<T>::value,
                ndspan<typename T::value_type, std::tuple_size<typename T::shape_type>::value>
                >
    {
        xt::xbuffer_adaptor<typename T::value_type> buff{ t.raw_data() + t.raw_data_offset(), t.size() };
        return { buff, t.shape() };
    }
}