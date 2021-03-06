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

#include "ss/ndspan.h"
#include <algorithm>

namespace ss
{
    template <typename T>
    using mat_view = ss::ndspan<T, 2>;

    template <typename T, class A = std::allocator<T>>
    using mat = xt::xtensor<T, 2, xt::layout_type::row_major, A>;

    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }

    template <size_t D, typename M>
    size_t stride(const M& mat) { return mat.strides()[D]; }

    template <typename T>
    using aligned_vector = std::vector<T>;
}