cmake_minimum_required (VERSION 3.1)

# -- python binding name
set (binding "py")

project ("${ss}-${binding}" VERSION "${PROJECT_VERSION}")

set (src      "${ss}/binding.cpp")
set (src_test "${ss}/binding_test.py")

pybind11_add_module (${binding} ${src})
configure_file ("${ss}/__init__.py" "${ss}/__init__.py" COPYONLY)
copy_target_files (${binding} ${ss})

# -- override the output name and output location
set_target_properties (${binding} PROPERTIES 
    OUTPUT_NAME "binding"
    LIBRARY_OUTPUT_DIRECTORY "${ss}"
)
foreach (config ${CMAKE_CONFIGURATION_TYPES})
    string (TOUPPER ${config} config)
    set_target_properties (${binding} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config} "${ss}")
endforeach()
target_link_libraries (${binding}
    PRIVATE ${ss}
)

# -- binding tests
if (${ss}_WITH_TESTS)
    include (CTest)
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND python "-u" "${src_test}"
    )
    set_tests_properties (${PROJECT_NAME}_test_suite
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif ()