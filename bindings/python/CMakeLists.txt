cmake_minimum_required (VERSION 3.1)

# -- python binding name
set (binding "py")

project ("${ss}-${binding}" VERSION "${PROJECT_VERSION}")

set (src "${ss}/binding.cpp")
# set (src_test "${ss}/binding_test.py")

pybind11_add_module (${binding} ${src})
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

# -- setup.py
configure_file (setup.py.in setup.py)

# -- copy .py
configure_file (${ss}/__init__.py     ${ss} COPYONLY)
configure_file (tests/test_binding.py tests COPYONLY)

# -- .whl creation
add_custom_target (bdist_wheel
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# -- binding tests
if (${ss}_WITH_TESTS)
    include (CTest)
    add_test (
        NAME ${PROJECT_NAME}_test_suite
        COMMAND ${PYTHON_EXECUTABLE} setup.py test
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif ()