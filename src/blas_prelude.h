#pragma once
#include "lib_config.h"

#if defined(BLAS_OpenBLAS)
# include <openblas_config.h>
# include <cblas.h>
#else
static_assert(false, "Couldn't determine which BLAS to use!");
#endif