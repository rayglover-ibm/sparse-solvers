# main test suite
add_executable ("${ss}_test"
    "lib_test.cpp"
    "homotopy_test.cpp"
)

target_link_libraries ("${ss}_test" "${ss}" gtest gmock_main)
copy_target_files (${ss}_test ${ss})

add_test (
	NAME ${ss}_test_suite
	COMMAND ${ss}_test
)