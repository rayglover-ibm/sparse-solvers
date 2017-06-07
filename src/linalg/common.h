#pragma once

namespace ss
{
    /* The following is a port:
          of ./tools/sparsity/src/Homotopy.py,
          at sha1 20b980c7804883d059896e04c3a0047615cbd984,
          committed 2015-11-09 14:08:24
    */
    using std::make_unique;

    template <typename T>
    using mat_view = ss::ndspan<T, 2>;

    template <typename T>
    using mat = xt::xtensor<T, 2>;

    template <size_t D, typename M>
    size_t dim(const M& mat) { return mat.shape()[D]; }
}