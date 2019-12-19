# Copyright(C) 2019 Intel Corporation
# Licensed under the MIT License

include (ExternalProject)

set(ngraph_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/ngraph)
set(ngraph_INSTALL_DIR ${ngraph_ROOT_DIR})
set(ngraph_INCLUDE_DIRS ${ngraph_INSTALL_DIR}/include)
set(ngraph_LIBRARIES ${ngraph_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
set(ngraph_SRC ${CMAKE_CURRENT_BINARY_DIR}/ngraph/src/project_ngraph)
set(prebuilt_ONNX_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/onnx")
set(prebuilt_ONNX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(ngraph_URL "https://github.com/NervanaSystems/ngraph.git")
set(ngraph_TAG "v0.26.0")

# Libraries for python package.
if (WIN32)
    set(NGRAPH_SHARED_LIB ngraph.dll)
    set(NGRAPH_CPU_BACKEND_SHARED_LIB cpu_backend.dll)
    set(NGRAPH_IOMP5MD_SHARED_LIB libiomp5md.dll)
    set(NGRAPH_MKLDNN_SHARED_LIB mkldnn.dll)
    set(NGRAPH_MKLML_SHARED_LIB mklml.dll)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(NGRAPH_TBB_SHARED_LIB tbb_debug.dll)
    else()
        set(NGRAPH_TBB_SHARED_LIB tbb.dll)
    endif()
else()
    set(NGRAPH_SHARED_LIB libngraph.so)
    set(NGRAPH_CODEGEN_SHARED_LIB libcodegen.so)
    set(NGRAPH_CPU_BACKEND_SHARED_LIB libcpu_backend.so)
    set(NGRAPH_IOMP5MD_SHARED_LIB libiomp5.so)
    set(NGRAPH_MKLDNN_SHARED_LIB libmkldnn.so)
    set(NGRAPH_MKLML_SHARED_LIB libmklml_intel.so)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(NGRAPH_TBB_SHARED_LIB libtbb_debug.so)
        set(NGRAPH_TBB_SHARED_LIB_2 libtbb_debug.so.2)
    else()
        set(NGRAPH_TBB_SHARED_LIB libtbb.so)
        set(NGRAPH_TBB_SHARED_LIB_2 libtbb.so.2)
    endif()
endif()

# discard prior changes due to unblock incremental builds.
set(NGRAPH_PATCH_DISCARD_COMMAND cd ${ngraph_SRC} && git reset HEAD --hard && git clean -fx)

if (MSVC)
    # For the moment, Windows does not support codegen, it works on DEX-only mode
    ExternalProject_Add(project_ngraph
            PREFIX ngraph
            GIT_REPOSITORY ${ngraph_URL}
            GIT_TAG ${ngraph_TAG}
            GIT_SHALLOW TRUE
            GIT_CONFIG core.autocrlf=input
            PATCH_COMMAND ${NGRAPH_PATCH_DISCARD_COMMAND}
            COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/patches/ngraph/ngraph_onnx.cmake ${ngraph_SRC}/cmake/external_onnx.cmake
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/ngraph/ngraph_protobuf.patch
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/ngraph/dnnl_v1.patch
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DNGRAPH_DEX_ONLY=ON
                -DNGRAPH_USE_SYSTEM_PROTOBUF=FALSE
                -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
                -DNGRAPH_INTERPRETER_ENABLE=FALSE
                -DNGRAPH_ONNXIFI_ENABLE=FALSE
                -DNGRAPH_UNIT_TEST_ENABLE=FALSE
                -DNGRAPH_TOOLS_ENABLE=FALSE
                -DNGRAPH_USE_LEGACY_MKLDNN=FALSE
                -DCMAKE_INSTALL_PREFIX=${ngraph_INSTALL_DIR}
                -Dprebuilt_ONNX_BINARY_DIR=${prebuilt_ONNX_BINARY_DIR}
                -Dprebuilt_ONNX_SOURCE_DIR=${prebuilt_ONNX_SOURCE_DIR}
            DEPENDS onnx
        )
    add_library(ngraph SHARED IMPORTED)
    set_property(TARGET ngraph PROPERTY IMPORTED_LOCATION ${ngraph_LIBRARIES}/${NGRAPH_SHARED_LIB})
    set_property(TARGET ngraph PROPERTY IMPORTED_IMPLIB ${ngraph_LIBRARIES}/ngraph.lib)
else()
    ExternalProject_Add(project_ngraph
            PREFIX ngraph
            GIT_REPOSITORY ${ngraph_URL}
            GIT_TAG ${ngraph_TAG}
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${NGRAPH_PATCH_DISCARD_COMMAND}
            # Here we use onnx and protobuf built by onnxruntime to avoid linking with incompatible libraries. This might change in future.
            COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/patches/ngraph/ngraph_onnx.cmake ${ngraph_SRC}/cmake/external_onnx.cmake
            # TODO: Use cmake.file+copy as above.
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/ngraph/ngraph_protobuf.patch
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/ngraph/dnnl_v1.patch
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DNGRAPH_USE_PREBUILT_LLVM=TRUE
                -DNGRAPH_USE_SYSTEM_PROTOBUF=FALSE
                -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
                -DNGRAPH_INTERPRETER_ENABLE=FALSE
                -DNGRAPH_ONNXIFI_ENABLE=FALSE
                -DNGRAPH_UNIT_TEST_ENABLE=FALSE
                -DNGRAPH_TOOLS_ENABLE=FALSE
                -DNGRAPH_USE_LEGACY_MKLDNN=FALSE
                -DCMAKE_INSTALL_PREFIX=${ngraph_INSTALL_DIR}
                -Dprebuilt_ONNX_BINARY_DIR=${prebuilt_ONNX_BINARY_DIR}
                -Dprebuilt_ONNX_SOURCE_DIR=${prebuilt_ONNX_SOURCE_DIR}
            DEPENDS onnx
        )

    add_library(ngraph SHARED IMPORTED)
    set_property(TARGET ngraph PROPERTY IMPORTED_LOCATION ${ngraph_LIBRARIES}/${NGRAPH_SHARED_LIB})
endif()
add_dependencies(ngraph project_ngraph)
include_directories(${ngraph_INCLUDE_DIRS})
