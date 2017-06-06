cmake_minimum_required (VERSION 3.3)
cmake_policy (SET CMP0048 NEW) # project versioning

project (core LANGUAGES C CXX VERSION 0.0.1)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# main library name
set (ss "sparsesolvers" CACHE STRING "native library name")

# options ---------------------------------------------------------------------
option ("${ss}_WITH_TESTS"   "Enable ${ss} unit tests" ON)
option ("${ss}_WITH_BENCHES" "Enable ${ss} benchmarks" OFF)
option ("${ss}_WITH_PYTHON"  "Enable ${ss} python binding" OFF)
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
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include"
            "$<TARGET_PROPERTY:kernelpp,INTERFACE_INCLUDE_DIRECTORIES>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries (${ss} kernelpp)

# -- compiler specifics
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options (${ss} PUBLIC
        -Wall -Wno-sign-compare
        -pedantic
        -fdiagnostics-color
        -fno-omit-frame-pointer
        -fmax-errors=5
    )

    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties (${ss}
            # strip binary
            PROPERTIES LINK_FLAGS "-Wl,--strip-all,--gc-sections"
        )
    endif ()
endif ()

# -- main tests
if (${ss}_WITH_TESTS)
    add_executable ("${ss}_test"
        "src/lib_test.cpp"
        "src/solvers/homotopy_test.cpp"
    )
    target_link_libraries ("${ss}_test" "${ss}" gtest gmock_main)
    copy_target_files (${ss}_test ${ss})

    enable_testing ()
    add_test (
        NAME ${ss}_test_suite
        COMMAND ${ss}_test
    )
endif ()

# -- benches
if (${ss}_WITH_BENCHES)
    # note: DownloadProject is a submodule of kernelpp
    include ("${PROJECT_SOURCE_DIR}/third_party/kernelpp/third_party/DownloadProject/DownloadProject.cmake")

    # download Google Benchmark targets
    download_project (
        PROJ            googlebench
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_TAG         master
        UPDATE_DISCONNECTED 1
    )
    add_subdirectory ("${googlebench_SOURCE_DIR}" "${googlebench_BINARY_DIR}")
    add_executable ("${ss}_benches" "src/solvers/homotopy_bench.cpp")
    target_link_libraries ("${ss}_benches" "${ss}" benchmark)

    enable_testing ()
    add_test (
        NAME ${ss}_benches
        COMMAND ${ss}_benches
    )
endif ()

# -- bindings
if (${ss}_WITH_PYTHON)
    add_subdirectory (third_party/pybind11)
    add_subdirectory (bindings/python)
endif ()