cmake_minimum_required (VERSION 3.3)
cmake_policy (SET CMP0048 NEW) # project versioning

project (core LANGUAGES C CXX VERSION 0.0.1)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# main library name
set (ss "sparsesolvers" CACHE STRING "native library name")

# options ---------------------------------------------------------------------
option ("${ss}_WITH_TESTS"  "Enable ${ss} unit tests" ON)
option ("${ss}_WITH_PYTHON" "Enable ${ss} python binding" OFF)
# -----------------------------------------------------------------------------

list (APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/cmake"
    "${CMAKE_CURRENT_LIST_DIR}/third_party/DownloadProject"
)

# dependencies
add_subdirectory (third_party/kernelpp)
include (BlasUtils)

# -- core library
add_library (${ss} STATIC
    "src/lib.cpp"
    "src/solvers/homotopy-cpu.cpp"
)
blas_init (${ss} blas_target blas_vendor)
configure_file (
    "include/ss/ss_config.h.in"
    "include/ss/ss_config.h"
)

target_include_directories (${ss}
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            "$<TARGET_PROPERTY:kernelpp,INTERFACE_INCLUDE_DIRECTORIES>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries (${ss} kernelpp)

# -- bindings
if (${ss}_WITH_PYTHON)
    add_subdirectory (third_party/pybind11)
    add_subdirectory (bindings/python)
endif ()

# -- tests
if (${ss}_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()