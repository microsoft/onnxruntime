# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pyop.cc")
onnxruntime_add_static_library(onnxruntime_pyop ${onnxruntime_pyop_srcs})
add_dependencies(onnxruntime_pyop onnxruntime_graph)
onnxruntime_add_include_to_target(onnxruntime_pyop onnxruntime_common onnxruntime_graph onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
target_include_directories(onnxruntime_pyop PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
onnxruntime_add_include_to_target(onnxruntime_pyop Python::Module Python::NumPy)
target_link_libraries(onnxruntime_pyop PRIVATE Python::Python)


