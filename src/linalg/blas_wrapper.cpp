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

namespace ss {
namespace blas
{
    namespace detail {
        handle_base::handle_base(std::string path)
        {
            h.resolve_policy(dlibxx::resolve::lazy);
            h.set_options(dlibxx::options::local);
            h.load(path);

            if (!h.loaded()) {
                fprintf(stderr, "Failed to load OpenBLAS: %s\n", h.error().c_str());
                abort();
            }
        }
    }

    cblas* cblas::get()
    {
        if (!m) { configure(); }
        return m.get();
    }

    void cblas::configure() {
        m.reset(new cblas("./libopenblas.so"));
    }

    std::unique_ptr<cblas> cblas::m;
}
}