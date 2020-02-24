# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_optimizer_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_optimizer_srcs})

add_library(onnxruntime_optimizer ${onnxruntime_optimizer_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/optimizer  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_optimizer onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
   target_compile_options(onnxruntime_optimizer PRIVATE "/wd4244")   
endif()
target_include_directories(onnxruntime_optimizer PRIVATE ${ONNXRUNTIME_ROOT})
add_dependencies(onnxruntime_optimizer ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_optimizer PROPERTIES FOLDER "ONNXRuntime")
