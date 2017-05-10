#pragma once

#include <kernelpp/types.h>
#include <kernelpp/kernel.h>

#include "ss/ss.h"

namespace ss
{
    using kernelpp::compute_mode;
    using kernelpp::error_code;

    KERNEL_DECL(solve_homotopy,
        compute_mode::CPU)
    {
        template <compute_mode> static kernelpp::variant<homotopy_report, error_code> op(
            const float* A,
            const std::uint32_t m,
            const std::uint32_t n,
            const std::uint32_t max_iter,
            const float tolerance,
            const float* y,
            float* x
            );
    };
}