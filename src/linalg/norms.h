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

#include <memory>
#include "linalg/common.h"

namespace ss
{
    template<typename T>
    void columnwise_sum(ndspan<T, 2> A, T* x)
    {
        std::fill_n(x, dim<1>(A), (T)0.0);

        for (size_t r{ 0 }; r < dim<0>(A); ++r) {
            for (size_t c{ 0 }; c < dim<1>(A); ++c) {
                x[c] += A(r, c);
            }
        }
    }

    template<typename T>
    bool l1(ndspan<T, 1> A)
    {
        T sum{ 0.0 };
        columnwise_sum(A, &sum);

        if (sum <= 0) { return false; }

        for (size_t i{ 0 }; i < dim<0>(A); ++i) {
            A(i, 0) /= sum;
        }
        return true;
    }

    template<typename T>
    bool l1(ndspan<T, 2> A)
    {
        /* matrix */
        auto sums = std::make_unique<T[]>(dim<1>(A));
        columnwise_sum(A, sums.get());

        for (size_t i{ 0 }; i < dim<1>(A); ++i) {
            if (sums[i] <= 0) { return false; }
        }

        for (size_t r{ 0 }; r < dim<0>(A); ++r) {
            for (size_t c{ 0 }; c < dim<1>(A); ++c) {
                A(r, c) /= sums[c];
            }
        }
        return true;
    }
}