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
#include "linalg/blas_wrapper.h"
#include "linalg/blas_prelude.h"

#include "kernelpp/kernel.h"
#include "kernelpp/kernel_invoke.h"

namespace ss {
namespace blas
{
    /* cblas --------------------------------------------------------------- */

    using namespace kernelpp;

    /* Kernel to perform feature detection for the runtime cpu/gpu */
    struct cblas::loader : kernelpp::impl<loader, compute_mode::AVX, compute_mode::CPU>
    {
        static constexpr const char* name = "cblas::loader";
        template<compute_mode> static error_code op();
    };

    cblas* cblas::get()
    {
        if (!m) { configure(); }
        return m.get();
    }

    void cblas::configure()
    {
        /* if an error occured when loading blas, abort */
        if (kernelpp::run<loader>()) {
            const char* msg = m && m->error() ?
                m->error().value().c_str() : "Failed to load cblas";

            fprintf(stderr, "%s\n", msg);
            abort();
        }
    }

    /* static cblas instance */
    std::unique_ptr<cblas> cblas::m;

    /* avx cpu blas */
    template <> error_code cblas::loader::op<compute_mode::AVX>()
    {
        cblas::m.reset(new cblas(BLAS_AVX_RUNTIME_FILE));
        return cblas::m->error() ? error_code::KERNEL_FAILED : error_code::NONE;
    }

    /* standard cpu blas */
    template <> error_code cblas::loader::op<compute_mode::CPU>()
    {
        cblas::m.reset(new cblas(BLAS_RUNTIME_FILE));
        return cblas::m->error() ? error_code::KERNEL_FAILED : error_code::NONE;
    }
}
}