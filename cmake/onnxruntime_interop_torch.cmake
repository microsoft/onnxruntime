# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(onnxruntime_python_interface.cmake)

# Python dependent files for torch interop
file(GLOB onnxruntime_interop_torch_srcs
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.h"
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.cc"
)

onnxruntime_add_static_library(onnxruntime_interop_torch ${onnxruntime_interop_torch_srcs})
add_dependencies(onnxruntime_interop_torch onnxruntime_python_interface onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnxruntime_interop_torch onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Python::Module)
target_include_directories(onnxruntime_interop_torch PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${ORTTRAINING_ROOT} ${DLPACK_INCLUDE_DIR})
set_target_properties(onnxruntime_interop_torch PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_interop_torch PROPERTIES FOLDER "ONNXRuntime")
