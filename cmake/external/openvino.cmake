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
set(ngraph_TAG "edc65ca0111f86a7e63a98f62cb17d153cc2535c")

# Libraries for python package.
if (WIN32)
    set(NGRAPH_SHARED_LIB ovep_ngraph.dll)
else()
    set(NGRAPH_SHARED_LIB libovep_ngraph.so)
endif()

# discard prior changes due to unblock incremental builds.
set(NGRAPH_CHANGE_DIR_COMMAND cd ${ngraph_SRC})

if (MSVC)
    # For the moment, Windows does not support codegen, it works on DEX-only mode
    ExternalProject_Add(project_ngraph
            PREFIX ngraph
            GIT_REPOSITORY ${ngraph_URL}
            GIT_TAG ${ngraph_TAG}
            GIT_SHALLOW TRUE
            GIT_CONFIG core.autocrlf=input
            PATCH_COMMAND ${NGRAPH_CHANGE_DIR_COMMAND}
            COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/patches/openvino/ngraph_onnx.cmake ${ngraph_SRC}/cmake/external_onnx.cmake
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/openvino/ngraph_protobuf.patch
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/openvino/lib_name.patch
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
                -DNGRAPH_JSON_ENABLE=FALSE
                -DNGRAPH_USE_SYSTEM_PROTOBUF=FALSE
                -DNGRAPH_UNIT_TEST_ENABLE=FALSE
                -DNGRAPH_TEST_UTIL_ENABLE=FALSE
                -DNGRAPH_TOOLS_ENABLE=FALSE
                -DNGRAPH_CPU_ENABLE=FALSE
                -DNGRAPH_INTERPRETER_ENABLE=FALSE
                -DNGRAPH_NOP_ENABLE=FALSE
                -DNGRAPH_GPU_ENABLE=FALSE
                -DNGRAPH_GENERIC_CPU_ENABLE=FALSE
                -DNGRAPH_PLAIDML_ENABLE=FALSE
                -DNGRAPH_ENABLE_CPU_CONV_AUTO=FALSE
                -DNGRAPH_PYTHON_BUILD_ENABLE=FALSE
                -DNGRAPH_FAST_MATH_ENABLE=FALSE
                -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=FALSE
                -DNGRAPH_NATIVE_ARCH_ENABLE=FALSE
                -DCMAKE_INSTALL_PREFIX=${ngraph_INSTALL_DIR}
                -Dprebuilt_ONNX_BINARY_DIR=${prebuilt_ONNX_BINARY_DIR}
                -Dprebuilt_ONNX_SOURCE_DIR=${prebuilt_ONNX_SOURCE_DIR}
            DEPENDS onnx
        )
    add_library(ovep_ngraph SHARED IMPORTED)
    set_property(TARGET ovep_ngraph PROPERTY IMPORTED_LOCATION ${ngraph_LIBRARIES}/${NGRAPH_SHARED_LIB})
    set_property(TARGET ovep_ngraph PROPERTY IMPORTED_IMPLIB ${ngraph_LIBRARIES}/ovep_ngraph.lib)

else()
    ExternalProject_Add(project_ngraph
            PREFIX ngraph
            GIT_REPOSITORY ${ngraph_URL}
            GIT_TAG ${ngraph_TAG}
            GIT_SHALLOW TRUE
            PATCH_COMMAND ${NGRAPH_CHANGE_DIR_COMMAND}
            # Here we use onnx and protobuf built by onnxruntime to avoid linking with incompatible libraries. This might change in future.
            COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/patches/openvino/ngraph_onnx.cmake ${ngraph_SRC}/cmake/external_onnx.cmake
            # TODO: Use cmake.file+copy as above.
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/openvino/ngraph_protobuf.patch
            COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/openvino/lib_name.patch
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
                -DNGRAPH_JSON_ENABLE=FALSE
                -DNGRAPH_USE_SYSTEM_PROTOBUF=FALSE
                -DNGRAPH_UNIT_TEST_ENABLE=FALSE
                -DNGRAPH_TEST_UTIL_ENABLE=FALSE
                -DNGRAPH_TOOLS_ENABLE=FALSE
                -DNGRAPH_CPU_ENABLE=FALSE
                -DNGRAPH_INTERPRETER_ENABLE=FALSE
                -DNGRAPH_NOP_ENABLE=FALSE
                -DNGRAPH_GPU_ENABLE=FALSE
                -DNGRAPH_GENERIC_CPU_ENABLE=FALSE
                -DNGRAPH_PLAIDML_ENABLE=FALSE
                -DNGRAPH_ENABLE_CPU_CONV_AUTO=FALSE
                -DNGRAPH_PYTHON_BUILD_ENABLE=FALSE
                -DNGRAPH_FAST_MATH_ENABLE=FALSE
                -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=FALSE
                -DNGRAPH_NATIVE_ARCH_ENABLE=FALSE
                -DCMAKE_INSTALL_PREFIX=${ngraph_INSTALL_DIR}
                -Dprebuilt_ONNX_BINARY_DIR=${prebuilt_ONNX_BINARY_DIR}
                -Dprebuilt_ONNX_SOURCE_DIR=${prebuilt_ONNX_SOURCE_DIR}
            DEPENDS onnx
        )

    add_library(ovep_ngraph SHARED IMPORTED)
    set_property(TARGET ovep_ngraph PROPERTY IMPORTED_LOCATION ${ngraph_LIBRARIES}/${NGRAPH_SHARED_LIB})
endif()
add_dependencies(ovep_ngraph project_ngraph)
include_directories(${ngraph_INCLUDE_DIRS})
