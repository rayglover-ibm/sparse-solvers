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

namespace ss {
namespace blas
{
    cblas* cblas::get()
    {
        if (!m) { configure(); }
        return m.get();
    }

    void cblas::configure() {
        m.reset(new cblas(BLAS_RUNTIME_FILE));
        /* if an error occured, abort */
        if (m->error()) {
            fprintf(stderr, "%s\n", m->error().value().c_str());
            abort();
        }
    }

    std::unique_ptr<cblas> cblas::m;
}
}