# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set (CXXOPTS ${PROJECT_SOURCE_DIR}/external/cxxopts/include)

# python dependent files for torch interop
file(GLOB onnxruntime_interop_torch_srcs
    "${ORTTRAINING_SOURCE_DIR}/core/training_ops/cpu/torch/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/training_ops/cpu/torch/*.cc"
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.h"
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.cc"
)

onnxruntime_add_static_library(onnxruntime_interop_torch ${onnxruntime_interop_torch_srcs})
add_dependencies(onnxruntime_interop_torch onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnxruntime_interop_torch onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)

onnxruntime_add_include_to_target(onnxruntime_interop_torch Python::Module)
target_include_directories(onnxruntime_interop_torch PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} PUBLIC ${onnxruntime_graph_header})

target_link_libraries(onnxruntime_interop_torch PRIVATE onnxruntime_python_interface)

# DLPack is a header-only dependency
set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/dlpack/include)
target_include_directories(onnxruntime_interop_torch PRIVATE ${ORTTRAINING_ROOT} ${DLPACK_INCLUDE_DIR})
