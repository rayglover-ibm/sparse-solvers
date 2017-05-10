cmake_minimum_required (VERSION 3.3)

project (core LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (ss "sparse-solver" CACHE STRING "native library name")
option ("${ss}_WITH_TESTS" "Enable ${ss} unit tests" ON)

list (APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/cmake"
    "${CMAKE_CURRENT_LIST_DIR}/third_party/DownloadProject"
)

# kernelpp, gtest
add_subdirectory (third_party/kernelpp)

add_library (${ss} STATIC
    "src/lib.cpp"
    "src/solvers/homotopy-cpu.cpp"
)

include (BlasUtils)
blas_init (${ss} blas_target blas_vendor)

# configurable header
configure_file (
    "src/lib_config.h.in"
    "src/lib_config.h"
)
target_include_directories (${ss}
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "$<TARGET_PROPERTY:kernelpp,INTERFACE_INCLUDE_DIRECTORIES>"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries (${ss} kernelpp)

# tests
if (${ss}_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()