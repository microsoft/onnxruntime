# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pyop.cc")
onnxruntime_add_static_library(onnxruntime_pyop ${onnxruntime_pyop_srcs})
add_dependencies(onnxruntime_pyop onnxruntime_graph onnxruntime_util)
onnxruntime_add_include_to_target(onnxruntime_pyop onnxruntime_common onnxruntime_util onnxruntime_graph onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
target_include_directories(onnxruntime_pyop PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
target_include_directories(onnxruntime_pyop PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_link_libraries(onnxruntime_pyop PRIVATE onnxruntime_util ${PYTHON_LIBRARIES})
