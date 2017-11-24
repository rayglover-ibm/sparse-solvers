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

#include <kernelpp/types.h>
#include <kernelpp/kernel.h>

#include "ss/ss.h"
#include "linalg/qr_decomposition.h"

namespace ss
{
    using kernelpp::compute_mode;
    using kernelpp::error_code;

    KERNEL_DECL(solve_irlq,
        compute_mode::CPU)
    {
        template <compute_mode, typename T>
        static kernelpp::variant<irlq_report, error_code> op(
            const ndspan<T, 2> Q,
            const ndspan<T> y,
            T tolerance,
            std::uint32_t max_iterations,
            ndspan<T> x
            );
    };
}