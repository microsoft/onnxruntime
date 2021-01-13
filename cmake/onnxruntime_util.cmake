# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_util_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/util/*.h"
    "${ONNXRUNTIME_ROOT}/core/util/*.cc"
    "${ONNXRUNTIME_ROOT}/core/profile/*.h"
    "${ONNXRUNTIME_ROOT}/core/profile/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_util_srcs})

add_library(onnxruntime_util ${onnxruntime_util_srcs})
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
   #TODO: fix the warnings, they are dangerous
   target_compile_options(onnxruntime_util PRIVATE "/wd4244")
endif()
target_include_directories(onnxruntime_util PRIVATE ${ONNXRUNTIME_ROOT} ${MKLML_INCLUDE_DIR} PUBLIC ${eigen_INCLUDE_DIRS})
if (onnxruntime_USE_CUDA)
 target_include_directories(onnxruntime_util PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
onnxruntime_add_include_to_target(onnxruntime_util onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
if(UNIX)
    target_compile_options(onnxruntime_util PUBLIC "-Wno-error=comment")
endif()
set_target_properties(onnxruntime_util PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_util PROPERTIES FOLDER "ONNXRuntime")
add_dependencies(onnxruntime_util ${onnxruntime_EXTERNAL_DEPENDENCIES})
if (WIN32)
    target_compile_definitions(onnxruntime_util PRIVATE _SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(onnxruntime_framework PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()
