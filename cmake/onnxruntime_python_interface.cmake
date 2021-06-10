# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_python_interface_cc_srcs
"${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_python.cc"
"${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_python.h"
"${ONNXRUNTIME_ROOT}/core/dlpack/python_common.h"
)

onnxruntime_add_static_library(onnxruntime_python_interface ${onnxruntime_python_interface_cc_srcs})
add_dependencies(onnxruntime_python_interface onnx  ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnxruntime_python_interface onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Python::Module)

# DLPack is a header-only dependency
set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/dlpack/include)
target_include_directories(onnxruntime_python_interface PUBLIC ${DLPACK_INCLUDE_DIR})
set_target_properties(onnxruntime_python_interface PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_python_interface PROPERTIES FOLDER "ONNXRuntime")
