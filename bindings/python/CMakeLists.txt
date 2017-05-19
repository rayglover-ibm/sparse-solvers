cmake_minimum_required (VERSION 3.1)

# capture the name of the core c++ library
set (binding "py")

project ("${ss}-${binding}" VERSION "${PROJECT_VERSION}")

set (src      "${ss}/binding.cpp")
set (src_test "${ss}/binding_test.py")

pybind11_add_module (${binding} ${src})
configure_file ("${ss}/__init__.py" "${ss}/__init__.py" COPYONLY)

# override the output name
set_target_properties (${binding} PROPERTIES 
    OUTPUT_NAME "binding"
)
target_link_libraries (${binding}
    PRIVATE ${ss}
)
add_custom_command (TARGET ${binding} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${binding}>" "${ss}"
)

# tests
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