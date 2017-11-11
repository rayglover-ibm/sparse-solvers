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
#include <utility>

namespace ss {
    namespace detail
    {
        using std::declval;

        /*  helper type which is well formed  when the solver policy supports
         *  solutions paramaterized by T
         */
        template <typename P, typename T>
        using solvable = decltype(
            declval<P>().run(
                declval<typename P::template state_type<T>&>(),
                declval<ndspan<T>>(), T{0}, std::size_t{0},
                declval<ndspan<T>>())
            );
        
        template <typename P, typename T, typename = void>
        struct is_solvable : std::false_type {};

        template <typename P, typename T>
        struct is_solvable <P, T, xt::void_t<solvable<P, T>> > : std::true_type {};    
    }
}