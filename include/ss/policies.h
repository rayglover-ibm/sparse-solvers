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
#include <kernelpp/types.h>
#include <xtl/xany.hpp>

namespace ss
{
    /* Homotopy ------------------------------------------------------------ */

    struct homotopy_report
    {
        /* The number of iterations performed. */
        uint32_t iter;

        /* The solution error */
        double solution_error;
    };

    /* Make std::variant happy */
    inline bool operator== (const homotopy_report&, const homotopy_report&) { return false; }

    /* A solver policy which implements the homotopy method */
    struct homotopy_policy
    {
        using report_type = homotopy_report;

        template <typename T> using state_type = const ndspan<T, 2>;

        static kernelpp::maybe<homotopy_report> run(
            state_type<float>&, const ndspan<float>, float, uint32_t, ndspan<float>);

        static kernelpp::maybe<homotopy_report> run(
            state_type<double>&, const ndspan<double>, double, uint32_t, ndspan<double>);

        homotopy_policy();
        homotopy_policy(homotopy_policy&&);

        ~homotopy_policy();
    };

    /* IRLS ---------------------------------------------------------------- */

    struct irls_report
    {
        /* The number of iterations performed. */
        uint32_t iter;

        /* The solution error */
        double solution_error;

        /*  Whether the IRLS failed because an iteration was evaluating
         *  a matrix which is not symmetric positive definite, i.e. it doesn't
         *  have a full cholesky decomposition.
         */
        bool spd_failure;
    };

    /* make std::variant happy */
    inline bool operator== (const irls_report&, const irls_report&) { return false; }

    /* */
    struct irls_state
    {
        irls_state(const ndspan<float, 2>);
        irls_state(const ndspan<double, 2>);

        ~irls_state();

        xtl::any QR;
    };

    /* A solver policy which implements the Iteratively Reweighted Least Squares method */
    struct irls_policy
    {
        using report_type = irls_report;

        template <typename> using state_type = irls_state;

        static kernelpp::maybe<irls_report> run(
            state_type<float>&, const ndspan<float>, float, uint32_t, ndspan<float>);

        static kernelpp::maybe<irls_report> run(
            state_type<double>&, const ndspan<double>, double, uint32_t, ndspan<double>);
    };
}