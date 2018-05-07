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

#include "linalg/common.h"

#include <memory>
#include <vector>

namespace ss
{
    /*  Poor-mans order-statistic tree with rank_of
     *  and rank_at operations.
     */
    template <typename T>
    class rank_index
    {
      public:
        explicit rank_index();

        auto begin() const { return _index.begin(); }
        auto end() const { return _index.end(); }

        auto crbegin() const { return _index.crbegin(); }
        auto crend() const { return _index.crend(); }

        int insert(T item);
        bool erase(const T& item);

        size_t size() const;

        int rank_of(const T& item) const;
        const T& rank_at(size_t index) const;

      private:
        std::vector<T> _index;
    };


    /* Implementation ------------------------------------------------------ */

    template <typename T>
    rank_index<T>::rank_index()
        : _index()
    {}

    template <typename T>
    size_t rank_index<T>::size() const
    {
        return _index.size();
    }

    template <typename T>
    int rank_index<T>::insert(T item)
    {
        auto bound = std::lower_bound(_index.begin(), _index.end(), item);
        size_t idx = bound - _index.begin();

        if (bound == _index.end() || *bound != item) {
            _index.insert(bound, item);
        }
        return idx;
    }

    template <typename T>
    int rank_index<T>::rank_of(const T& item) const
    {
        auto bound = std::lower_bound(_index.begin(), _index.end(), item);
        return bound == _index.end() || *bound != item
            ? -1 : std::distance(_index.begin(), bound);
    }

    template <typename T>
    const T& rank_index<T>::rank_at(size_t index) const {
        return _index[index];
    }

    template <typename T>
    bool rank_index<T>::erase(const T& item)
    {
        auto bound = std::lower_bound(_index.begin(), _index.end(), item);
        if (bound != _index.end()) {
            _index.erase(bound);
            return true;
        }
        return false;
    }
}