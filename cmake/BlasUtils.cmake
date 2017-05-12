include (DownloadProject)

define_property(TARGET
	PROPERTY
	REQUIRED_RUNTIME_FILES
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
    if (WIN32)
        set (downloads_root "https://github.com/rayglover-ibm/openblas-ci/releases/download")
        set (version "v0.2.19")
        set (platform "win")

        download_project (
            PROJ  OpenBLAS
            URL   ${downloads_root}/${version}/${platform}.zip
            UPDATE_DISCONNECTED 1
        )
        find_package (OpenBLAS REQUIRED
            PATHS "${OpenBLAS_SOURCE_DIR}"
        )
        
        # OpenBLAS dll needs to be copied to binary directory
        append_target_files (${target} OpenBLAS)
        
		target_link_libraries (${target} OpenBLAS)
        target_include_directories (${target} PUBLIC
            $<TARGET_PROPERTY:OpenBLAS,INTERFACE_INCLUDE_DIRECTORIES>
        )

        set (BLAS_OpenBLAS 1)
        set (${pkg} "OpenBLAS")
        set (${vendor} "OpenBLAS")
    else ()
        message (SEND_ERROR "TODO(rayg) non-windows blas")
    endif ()
endmacro ()
