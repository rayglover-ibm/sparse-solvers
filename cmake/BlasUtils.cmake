define_property (TARGET PROPERTY REQUIRED_RUNTIME_FILES
    BRIEF_DOCS "Runtime files required for this target"
    FULL_DOCS  "Runtime files required for this target"
)

# for each required runtime file in 'target', copy it to
# the target directory of 'dest_target' post-build
function (copy_target_files dest_target target)
    get_target_property (tgts ${target} REQUIRED_RUNTIME_FILES)
    if (NOT "${tgts}" MATCHES "NOTFOUND")
        foreach (file ${tgts})
            add_custom_command (TARGET ${dest_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "[copy] ${file}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${file}" "$<TARGET_FILE_DIR:${dest_target}>"
            )
        endforeach ()
    endif ()
endfunction ()

macro (blas_init target blas_target)
    set (dest ${CMAKE_CURRENT_BINARY_DIR})

    # download OpenBLAS bootsrap
    set (bootstrap "${dest}/tmp.OpenBLASBootstrap.cmake")
    if (NOT EXISTS "${bootstrap}")
        file (DOWNLOAD
            "https://raw.githubusercontent.com/rayglover-ibm/openblas-ci/master/OpenBLASBootstrap.cmake"
            "${bootstrap}")
    endif ()

    # get the latest OpenBLAS builds
    include ("${bootstrap}")
    OpenBLAS_find_archive (BUILD_URL url)
    OpenBLAS_init (BUILD_URL "${url}" COMPONENTS NEHALEM HASWELL)

    add_dependencies (${target}
        "OpenBLAS::NEHALEM" "OpenBLAS::HASWELL"
    )
    set_target_properties (${target} PROPERTIES
        REQUIRED_RUNTIME_FILES "$<TARGET_FILE:OpenBLAS::NEHALEM>;$<TARGET_FILE:OpenBLAS::HASWELL>"
    )
    set (BLAS_OpenBLAS 1)
    set (${blas_target} OpenBLAS::NEHALEM)
endmacro ()

macro (set_rpath target rpath)
    set_target_properties (${target} PROPERTIES 
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH ${rpath}
    )
endmacro ()
