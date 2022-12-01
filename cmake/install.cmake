function(install_headers_from_dir srcdir)
    #Note: install(DIRECTORY ...) causes empty directories being created
    get_filename_component(cname "${srcdir}" NAME)
    foreach (subdir ${ARGN})
        set(abssrcdir "${CMAKE_SOURCE_DIR}/../${srcdir}/${subdir}")
        file(GLOB_RECURSE files RELATIVE "${abssrcdir}" "${abssrcdir}/*.h")
        foreach (f ${files})
            get_filename_component(d "${f}" DIRECTORY)
            install(FILES "${abssrcdir}/${f}" DESTINATION "include/onnxruntime/${cname}/${subdir}/${d}")
        endforeach()
    endforeach()
endfunction()

install_headers_from_dir(include/onnxruntime/core platform)
install_headers_from_dir(onnxruntime/core
        common
        framework
        platform
        session
        util
        providers/shared_library
        providers/shared
        providers/cpu
        )

install_headers_from_dir(orttraining/orttraining
        models/runner
        training_ops/cpu
        core/framework
        core/graph
        core/optimizer
        core/session
        )

install(FILES
        ${CMAKE_SOURCE_DIR}/../include/onnxruntime/core/providers/providers.h
        ${CMAKE_SOURCE_DIR}/../onnxruntime/core/providers/op_kernel_type_control.h
        ${CMAKE_SOURCE_DIR}/../onnxruntime/core/framework/op_kernel_type_control_utils.h
        ${CMAKE_SOURCE_DIR}/../onnxruntime/core/providers/common.h
        ${CMAKE_SOURCE_DIR}/../onnxruntime/core/providers/utils.h
        ${CMAKE_SOURCE_DIR}/../onnxruntime/core/providers/get_execution_providers.h

        DESTINATION include/onnxruntime/core/providers
)

# Build Generated Headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h
        DESTINATION include/onnxruntime)
