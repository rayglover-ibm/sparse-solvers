define_property (TARGET
    PROPERTY REQUIRED_RUNTIME_FILES
    BRIEF_DOCS "Runtime files required for this target"
    FULL_DOCS "Runtime files required for this target"
)

function (copy_target_files dest_target target)
    get_target_property (tgts ${target} REQUIRED_RUNTIME_FILES)
    if (NOT ${tgts} MATCHES "NOTFOUND")
        foreach (tgt ${tgts})
            add_custom_command (TARGET ${dest_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying $<TARGET_SONAME_FILE_NAME:${tgt}>"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${tgt}>
                    $<TARGET_FILE_DIR:${dest_target}>
            )
        endforeach ()
    endif ()
endfunction ()

macro (blas_init target pkg vendor)
    # download OpenBLAS bootsrap
    set (bootstrap "${CMAKE_CURRENT_BINARY_DIR}/tmp.OpenBLASBootstrap.cmake")
    if (NOT EXISTS "${bootstrap}")
        file (DOWNLOAD
            "https://raw.githubusercontent.com/rayglover-ibm/openblas-ci/master/OpenBLASBootstrap.cmake"
            "${bootstrap}")
    endif ()

    # Get the latest OpenBLAS build
    include ("${bootstrap}")
    OpenBLAS_find_archive (BUILD_URL url)
    OpenBLAS_init (BUILD_URL "${url}" PROJ OpenBLAS)

    set (BLAS_OpenBLAS 1)
    set (${pkg} "OpenBLAS")
    set (${vendor} "OpenBLAS")
endmacro ()
