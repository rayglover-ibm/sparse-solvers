cmake_minimum_required (VERSION 3.2)
cmake_policy (SET CMP0048 NEW) # project versioning

project (core LANGUAGES C CXX VERSION 0.0.2)

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
)

# dependencies
add_subdirectory (third_party/kernelpp)
include (BlasUtils)

# -- core library
list (APPEND src
    "src/lib.cpp"
    "src/solvers/homotopy-cpu.cpp"
    "src/linalg/blas_wrapper.cpp"
    "third_party/dlibxx/src/dlibxx.unix.cxx"
)

add_library (${ss} STATIC ${src})
blas_init (${ss} blas_target)

configure_file (
    "include/ss/ss_config.h.in"
    "include/ss/ss_config.h.processed"
)
file (GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/ss/ss_config.h"
    INPUT  "${CMAKE_CURRENT_BINARY_DIR}/include/ss/ss_config.h.processed"
)

target_include_directories (${ss}
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtl/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include"
            "$<TARGET_PROPERTY:kernelpp,INTERFACE_INCLUDE_DIRECTORIES>"
            "$<TARGET_PROPERTY:${blas_target},INTERFACE_INCLUDE_DIRECTORIES>"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlibxx/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries (${ss} PUBLIC kernelpp dl)

# -- general compiler/linker settings
target_compile_options (${ss} PUBLIC
    $<$<CONFIG:Debug>:-g -DXTENSOR_ENABLE_ASSERT>
)
# -- compiler/linker specifics (clang/gcc)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options (${ss} PUBLIC
        -Wall -pedantic -Wno-missing-braces -Wno-deprecated-declarations
        -fdiagnostics-color -fno-omit-frame-pointer
    )
    target_compile_options (${ss} PUBLIC
        $<$<CONFIG:Release>:-O3>
    )
endif ()

# -- main tests
if (${ss}_WITH_TESTS)
    if (NOT TARGET gtest)
        # note: google test should be enabled by default by kernelpp
        message (SEND_ERROR "Tests could not be built: Google Test not configured correctly.")
    endif ()
    add_executable ("${ss}_test"
        "src/lib_test.cpp"
        "src/solvers/homotopy_test.cpp"
        "src/linalg/rank_index_test.cpp"
        "src/linalg/online_inverse_test.cpp"
        "src/linalg/qr_decomposition_test.cpp"
        "src/linalg/cholesky_decomposition_test.cpp"
        "src/linalg/norms_test.cpp"
    )
    target_include_directories ("${ss}_test"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    set_rpath ("${ss}_test" "$ORIGIN/")
    copy_target_files ("${ss}_test" "${ss}")
    target_link_libraries ("${ss}_test" "${ss}" gtest gmock_main)

    enable_testing ()
    add_test (
        NAME ${ss}_test_suite
        COMMAND ${ss}_test
    )
endif ()

# -- benches
if (${ss}_WITH_BENCHES)
    # retrieve DownloadProject
    set (src "https://raw.githubusercontent.com/Crascit/DownloadProject/master")
    set (dest "${CMAKE_CURRENT_BINARY_DIR}/tmp.DownloadProject")

    foreach (file
        "DownloadProject.cmake"
        "DownloadProject.CMakeLists.cmake.in")
        file (DOWNLOAD "${src}/${file}" "${dest}/${file}" STATUS "retrieving ${file}")
    endforeach ()

    # download Google Benchmark targets
    include ("${dest}/DownloadProject.cmake")
    download_project (
        PROJ            googlebench
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_TAG         master
        UPDATE_DISCONNECTED 1
    )
    set (BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable Googlebench test suite" FORCE)
    add_subdirectory ("${googlebench_SOURCE_DIR}" "${googlebench_BINARY_DIR}")
    add_executable ("${ss}_benches"
        "src/linalg/qr_decomposition_bench.cpp"
        "src/linalg/cholesky_decomposition_bench.cpp"
        "src/solvers/homotopy_bench.cpp"
        "src/lib_bench.cpp"
    )
    target_include_directories ("${ss}_benches"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    set_rpath ("${ss}_benches" "$ORIGIN/")
    copy_target_files ("${ss}_benches" "${ss}")
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