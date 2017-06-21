define_property(TARGET
    PROPERTY REQUIRED_RUNTIME_FILES
    BRIEF_DOCS "Runtime taget files required for this target"
    FULL_DOCS "Runtime taget files required for this target"
)

function (append_target_files target)
    set_property (TARGET ${target} APPEND PROPERTY REQUIRED_RUNTIME_FILES ${ARGN})
endfunction ()

function (copy_target_files dest_target target)
    get_target_property (tgts ${target} REQUIRED_RUNTIME_FILES)
    foreach (tgt ${tgts})
        add_custom_command (TARGET ${dest_target} POST_BUILD
            COMMENT "Copying $<TARGET_FILE:${tgt}>" VERBATIM
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${tgt}>
                $<TARGET_FILE_DIR:${dest_target}>
        )
    endforeach ()
endfunction ()

macro (blas_init target pkg vendor)
    file (DOWNLOAD
        "https://raw.githubusercontent.com/rayglover-ibm/openblas-ci/master/OpenBLASBootstrap.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp.OpenBLASBootstrap.cmake"
    )
    include ("${CMAKE_CURRENT_BINARY_DIR}/tmp.OpenBLASBootstrap.cmake")
    OpenBLAS_find_archive (
        RELEASE_NAME "LATEST" OS ${CMAKE_SYSTEM_NAME} BUILD_URL url
    )
    OpenBLAS_init (
        BUILD_URL "${url}" PROJ OpenBLAS
    )

    # OpenBLAS library needs to be copied to binary directory
    append_target_files (${target} OpenBLAS)
    
    target_link_libraries (${target} OpenBLAS)
    target_include_directories (${target} PUBLIC
        $<TARGET_PROPERTY:OpenBLAS,INTERFACE_INCLUDE_DIRECTORIES>
    )

    set (BLAS_OpenBLAS 1)
    set (${pkg} "OpenBLAS")
    set (${vendor} "OpenBLAS")
endmacro ()
