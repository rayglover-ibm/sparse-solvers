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

#include <xtensor/xbuffer_adaptor.hpp>
#include <xtensor/xadapt.hpp>
#include <array>

namespace ss
{
    /*
     *  A non-owning, row-major n-dimensional view over a contiguous
     *  (ptr, size) representation.
     */
    template <typename T, size_t NDim = 1>
    using ndspan = xt::xtensor_adaptor<
        xt::xbuffer_adaptor<T>, NDim, xt::layout_type::row_major
        >;

    /* as_span ------------------------------------------------------------- */

    /*
     *  constructs a 1-d non-owning view of an stl-like container
     */
    template <typename C>
    inline auto as_span(C& container)
        -> std::enable_if_t<!xt::has_raw_data_interface<C>::value,
                ndspan<typename C::value_type, 1>
                >
    {
        auto* buf = container.data();
        size_t len = container.size();

        return xt::xadapt(buf, len, xt::no_ownership(), std::array<size_t, 1>{ len },
            xt::layout_type::row_major);
    }

    /*
     *  constructs a n-d non-owning view of the given shape
     *  of an stl-like container
     */
    template <size_t N, typename C>
    ndspan<typename C::value_type, N> as_span(C& container, std::array<size_t, N> shape)
    {
        auto* buf = container.data();
        size_t len = container.size();

        return xt::xadapt(buf, len, xt::no_ownership(), shape,
            xt::layout_type::row_major);
    }

    /*
     *  constructs a n-d non-owning view of a xtensor-like container
     */
    template <class T>
    inline auto as_span(T& t)
        -> std::enable_if_t<xt::has_raw_data_interface<T>::value,
                ndspan<typename T::value_type, std::tuple_size<typename T::shape_type>::value>
                >
    {
        auto* buf = t.raw_data() + t.raw_data_offset();
        size_t len = t.size();

        return xt::xadapt(buf, len, xt::no_ownership(), t.shape(),
            xt::layout_type::row_major);
    }
}